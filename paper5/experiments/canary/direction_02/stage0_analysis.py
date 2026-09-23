"""Budget-respecting policy replay and paired document-cluster inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

UPDATES = {"B0": 100, "B20": 240, "BSH": 380, "Ref100": 800}


def replay(scores: dict[int, dict[str, float]]) -> dict[str, Any]:
    """Freeze strategy choices with an explicit log of permitted S queries."""
    ids = sorted(scores[0])
    if len(ids) != 8 or set(scores) != {0, 20, 50, 100}:
        raise ValueError("Expected eight complete trajectories at 0/20/50/100")
    for values in scores.values():
        if sorted(values) != ids or not np.isfinite(list(values.values())).all():
            raise ValueError("Missing or nonfinite S loss")
    result = {}
    for strategy in UPDATES:
        queries = []

        def best(step: int, pool: list[str], count: int) -> list[str]:
            queries.extend({"candidate": name, "checkpoint": step} for name in pool)
            return sorted(pool, key=lambda name: (scores[step][name], name))[:count]

        if strategy == "B0":
            chosen = best(0, ids, 1)[0]
        elif strategy == "B20":
            chosen = best(20, ids, 1)[0]
        elif strategy == "Ref100":
            chosen = best(100, ids, 1)[0]
        else:
            four = best(20, ids, 4)
            two = best(50, four, 2)
            chosen = best(100, two, 1)[0]
        result[strategy] = {
            "selected": chosen,
            "endpoint": 100,
            "training_updates": UPDATES[strategy],
            "queries": queries,
        }
    return result


def paired_document_bootstrap(
    losses: np.ndarray,
    doc_ids: list[str],
    labels: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample documents jointly across all model rows; retain each entire cluster."""
    if (
        losses.ndim != 2
        or losses.shape[1] != len(doc_ids)
        or labels.shape != (len(doc_ids),)
    ):
        raise ValueError("Loss/document/label shapes do not align")
    if not np.isfinite(losses).all() or np.any(labels <= 0):
        raise ValueError("Nonfinite NLL or empty labels")
    docs = sorted(set(doc_ids))
    if len(docs) < 2:
        raise ValueError("At least two evaluation documents are required")
    totals = np.stack(
        [
            (
                losses[:, np.asarray(doc_ids) == doc]
                * labels[np.asarray(doc_ids) == doc]
            ).sum(axis=1)
            for doc in docs
        ],
        axis=1,
    )
    counts = np.asarray([labels[np.asarray(doc_ids) == doc].sum() for doc in docs])
    draws = np.random.default_rng(seed).integers(0, len(docs), (replicates, len(docs)))
    return (totals[:, draws].sum(axis=2) / counts[draws].sum(axis=1)).T


def interval(values: np.ndarray, point: float) -> dict[str, float]:
    """Paired percentile 95% interval, allowing signed selection differences."""
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"point": float(point), "lower95": float(lower), "upper95": float(upper)}


def classify(seed_results: dict[str, Any], margin: float) -> dict[str, Any]:
    """Keep final quality, problem evidence and simple-baseline results separate."""
    if set(seed_results) != {"17", "29"}:
        raise ValueError("Both fixed seeds are required for a scientific decision")
    values = list(seed_results.values())
    quality = all(row["quality_usable"] for row in values)
    if not quality:
        problem = "NOT_ASSESSED_QUALITY"
        overall = "QUALITY_INFEASIBLE"
    elif all(row["G0"]["upper95"] <= margin for row in values):
        problem = overall = "NO_MATERIAL_SELECTION_GAP"
    elif all(
        row["G0"]["point"] > margin and row["G0"]["lower95"] > 0 for row in values
    ):
        problem = overall = "PROBLEM_SIGNAL"
    else:
        problem = overall = "INCONCLUSIVE"
    sufficient = []
    for name in ("B20", "BSH"):
        if quality and all(
            row["strategies"][name]["difference_vs_ref"]["upper95"] <= margin
            and row["strategies"][name]["quality_pass"]
            and row["strategies"][name]["lower_cost"]
            for row in values
        ):
            sufficient.append(name)
    return {
        "engineering": "PASSED",
        "quality": "USABLE" if quality else "QUALITY_INFEASIBLE",
        "problem": problem,
        "verdict": overall,
        "simple_baseline": (
            "SIMPLE_BASELINE_SUFFICIENT" if sufficient else "NOT_ESTABLISHED"
        ),
        "sufficient_strategies": sufficient,
        "complex_method_authorized": False,
    }


def analyze(
    selection: dict[str, Any],
    s_losses: dict[str, Any],
    e_losses: dict[str, Any],
    documents: list[str],
    labels: list[int],
    cfg: dict[str, Any],
    unit_costs: dict[str, float],
) -> dict[str, Any]:
    """Analyze complete endpoints using the choices persisted before E access."""
    names = sorted(e_losses["17"])
    if len(names) != 8 or sorted(e_losses["29"]) != names:
        raise ValueError("Incomplete endpoints")
    arrays = [e_losses[seed][name] for seed in ("17", "29") for name in names]
    matrix = np.asarray([*arrays, e_losses["parent"]], dtype=float)
    weights = np.asarray(labels, dtype=float)
    means = np.average(matrix, axis=1, weights=weights)
    draws = paired_document_bootstrap(
        matrix, documents, weights, cfg["bootstrap_replicates"], cfg["bootstrap_seed"]
    )
    parent = float(means[-1])
    cutoff = np.log(cfg["proposed_quality_max_ppl_ratio"])
    seed_results = {}
    for offset, seed in enumerate(("17", "29")):
        start = offset * len(names)
        candidate_means = means[start : start + len(names)]
        candidate_draws = draws[:, start : start + len(names)]
        choices = selection[seed]
        ref = names.index(choices["Ref100"]["selected"])
        strategies = {}
        common = unit_costs["historical_preselection_estimate_seconds"]
        for strategy, record in choices.items():
            query_windows = len(record["queries"]) * cfg["windows"]["S"]
            constructions = 9 if strategy == "B0" else 8
            # Includes one endpoint E evaluation and persistence at policy stage boundaries.
            writes = {"B0": 1, "B20": 9, "BSH": 14, "Ref100": 8}[strategy]
            seconds = (
                common
                + record["training_updates"] * unit_costs["update_seconds"]
                + (query_windows + cfg["windows"]["E"])
                * unit_costs["eval_window_seconds"]
                + constructions * unit_costs["construction_seconds"]
                + writes * unit_costs["checkpoint_seconds"]
            )
            index = names.index(record["selected"])
            strategies[strategy] = {
                **record,
                "endpoint_e_nll": float(candidate_means[index]),
                "difference_vs_ref": interval(
                    candidate_draws[:, index] - candidate_draws[:, ref],
                    candidate_means[index] - candidate_means[ref],
                ),
                "hindsight_simple_regret": interval(
                    candidate_draws[:, index] - candidate_draws.min(axis=1),
                    candidate_means[index] - candidate_means.min(),
                ),
                "quality_pass": bool(candidate_means[index] - parent <= cutoff),
                "s_query_windows": query_windows,
                "training_input_tokens": record["training_updates"] * 4 * 512,
                "training_effective_labels": record["training_updates"] * 4 * 511,
                "standalone_seconds_estimate": seconds,
            }
        for strategy, row in strategies.items():
            row["lower_cost"] = bool(
                row["training_updates"] <= 0.5 * UPDATES["Ref100"]
                and row["standalone_seconds_estimate"]
                < strategies["Ref100"]["standalone_seconds_estimate"]
            )
        initial = np.asarray([s_losses[seed]["0"][name] for name in names])
        final = np.asarray([s_losses[seed]["100"][name] for name in names])
        rank0 = sorted(names, key=lambda name: (s_losses[seed]["0"][name], name))
        rank100 = sorted(names, key=lambda name: (s_losses[seed]["100"][name], name))
        seed_results[seed] = {
            "quality_candidate_count": int(np.sum(candidate_means - parent <= cutoff)),
            "quality_usable": bool(
                np.sum(candidate_means - parent <= cutoff)
                >= cfg["quality_min_candidates"]
                and candidate_means[ref] - parent <= cutoff
            ),
            "G0": strategies["B0"]["difference_vs_ref"],
            "strategies": strategies,
            "candidate_endpoint_e_nll": dict(zip(names, candidate_means.tolist())),
            "s_rank_0": rank0,
            "s_rank_100": rank100,
            "top1_retained": rank0[0] == rank100[0],
            "top3_overlap": len(set(rank0[:3]) & set(rank100[:3])) / 3,
            "spearman": (
                float(spearmanr(initial, final).statistic)
                if np.ptp(initial) and np.ptp(final)
                else None
            ),
            "kendall": (
                float(kendalltau(initial, final).statistic)
                if np.ptp(initial) and np.ptp(final)
                else None
            ),
            "s_damage_recovery_nll": dict(zip(names, (initial - final).tolist())),
        }
    return {
        "parent_e_nll": parent,
        "quality_excess_nll": float(cutoff),
        "seeds": seed_results,
        "decision": classify(seed_results, cfg["material_gap_nats"]),
        "bootstrap": {
            "unit": "document",
            "documents": len(set(documents)),
            "replicates": cfg["bootstrap_replicates"],
            "seed": cfg["bootstrap_seed"],
            "point_estimand": "label-token-weighted NLL",
            "seeds_pooled": False,
        },
        "simple_regret_limitation": "finite-candidate E hindsight reference; intervals descriptive",
        "cost_limitation": "policy replay and standalone estimates are not measured speedups",
    }
