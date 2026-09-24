"""CPU analysis of the fixed expanded set; original results remain separate."""

from __future__ import annotations

from collections import defaultdict
import csv
import math
import time
from typing import Any

import numpy as np

from . import run_competitor_coverage as coverage
from . import run_stage0 as runner
from . import stage0_analysis as analysis
from . import stage0_data as data


def rows_and_curves(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Validate exact identities and scores, keeping old and new rows identifiable."""
    combined = []
    s_scores = data.read_json(runner.OUT / "s_scores.json")
    e_losses = data.read_json(runner.OUT / "e_losses.json")
    for folder, names, parent in (
        (coverage.SOURCE, [f"a{i:02d}" for i in range(8)], True),
        (runner.OUT, ["a08", "a09"], False),
    ):
        with (folder / "per_window_nll.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = {
            ("S", str(seed), name, step, w)
            for seed in (17, 29)
            for name in names
            for step in (0, 20, 50, 100)
            for w in range(64)
        }
        expected |= {
            ("E", str(seed), name, 100, w)
            for seed in (17, 29)
            for name in names
            for w in range(64)
        }
        if parent:
            expected |= {("E", "parent", "parent", 0, w) for w in range(64)}
        seen = set()
        s_values = defaultdict(list)
        for row in rows:
            split, seed, name = row["split"], row["seed"], row["candidate"]
            step, w = int(row["checkpoint"]), int(row["window_id"])
            key = (split, seed, name, step, w)
            if key not in expected or key in seen:
                raise RuntimeError("Unexpected or duplicate measurement")
            seen.add(key)
            window = manifest["splits"][split]["windows"][w]
            value = float(row["nll"])
            if (
                not math.isfinite(value)
                or row["doc_id"] != window["doc_id"]
                or int(row["document_token_start"]) != window["document_token_start"]
                or int(row["labels"]) != window["labels"]
            ):
                raise RuntimeError("Measurement token/document mapping mismatch")
            if split == "S":
                s_values[(seed, name, str(step))].append(value)
            elif value != (
                e_losses["parent"][w] if name == "parent" else e_losses[seed][name][w]
            ):
                raise RuntimeError("E arrays differ from per-window rows")
        if seen != expected:
            raise RuntimeError("Missing measurements")
        for (seed, name, step), values in s_values.items():
            if not math.isclose(
                float(np.mean(values)), s_scores[seed][step][name], abs_tol=1e-12
            ):
                raise RuntimeError("S means differ from per-window rows")
        with (folder / "training_curve.csv").open(newline="") as handle:
            curves = list(csv.DictReader(handle))
        expected_updates = {
            (str(seed), name, step)
            for seed in (17, 29)
            for name in names
            for step in range(1, 101)
        }
        if (
            len(curves) != len(expected_updates)
            or {(row["seed"], row["candidate"], int(row["update"])) for row in curves}
            != expected_updates
        ):
            raise RuntimeError("Incomplete training curves")
        if any(
            not math.isfinite(float(row[k]))
            for row in curves
            for k in ("train_nll", "gradient_norm", "update_seconds")
        ):
            raise RuntimeError("Nonfinite training curve")
        combined.extend(rows)
    return combined


def analyze() -> None:
    """No E models are run here; intervals are explicitly exploratory descriptions."""
    started = time.monotonic()
    coverage.validate_frozen()
    selection = coverage.validate_selection()
    runtime = data.read_json(runner.OUT / "runtime.json")
    budget = data.read_json(runner.OUT / "gpu_budget.json")
    if (
        runtime["phase"] != "EVALUATED"
        or budget["phases"][-1]["status"] != "COMPLETE"
        or budget["spent_seconds"] > 14400
        or budget["phases"][-1]["elapsed_seconds"] > 3600
    ):
        raise RuntimeError("Incomplete or over-budget supplement")
    manifest = data.read_json(runner.OUT / "data_manifest.json")
    rows = rows_and_curves(manifest)
    scores = data.read_json(runner.OUT / "s_scores.json")
    windows = manifest["splits"]["E"]["windows"]
    result = analysis.analyze(
        selection["choices"],
        scores,
        data.read_json(runner.OUT / "e_losses.json"),
        [w["doc_id"] for w in windows],
        [w["labels"] for w in windows],
        runner.CFG,
        data.read_json(coverage.SOURCE / "pilot.json")["unit_costs"],
        policy_sizes=coverage.SIZES,
    )
    result.update(
        executed_code_sha=runtime["executed_code_sha"],
        base_archive_sha=coverage.BASE,
        interpretation="post-V2 exploratory coverage; E previously viewed and reused; intervals descriptive, not new confirmation",
        analysis_seconds=time.monotonic() - started,
    )
    result["bootstrap"]["confirmatory"] = False
    original = data.read_json(coverage.SOURCE / "metrics.json")
    data.write_json(runner.OUT / "metrics.json", result)
    data.write_json(
        runner.OUT / "original_vs_expanded.json",
        {
            "original_archive_sha": coverage.BASE,
            "original_metrics_sha256": data.file_hash(coverage.SOURCE / "metrics.json"),
            "original_eight": {k: original[k] for k in ("decision", "seeds")},
            "expanded_ten": {k: result[k] for k in ("decision", "seeds")},
            "independent_experiments": False,
            "reuse": "16 original trajectories plus exactly 4 new trajectories; not independent repeats",
        },
    )
    parent_s = data.read_json(coverage.SOURCE / "pilot.json")["parent_s_nll"]
    data.write_json(
        runner.OUT / "quality_by_checkpoint.json",
        {
            "threshold": 1.15,
            "parent_s_nll": parent_s,
            "parent_e_nll": result["parent_e_nll"],
            "parent_recovery_added": False,
            "s": {
                seed: {
                    step: {
                        name: {
                            "nll": loss,
                            "ppl_ratio": math.exp(loss - parent_s),
                            "quality_pass": math.exp(loss - parent_s) <= 1.15,
                        }
                        for name, loss in values.items()
                    }
                    for step, values in points.items()
                }
                for seed, points in scores.items()
            },
            "e": {
                seed: {
                    name: {
                        "nll": loss,
                        "ppl_ratio": math.exp(loss - result["parent_e_nll"]),
                        "quality_pass": math.exp(loss - result["parent_e_nll"]) <= 1.15,
                    }
                    for name, loss in entry["candidate_endpoint_e_nll"].items()
                }
                for seed, entry in result["seeds"].items()
            },
        },
    )
    groups = defaultdict(lambda: [0.0, 0, 0])
    for row in rows:
        runner.record_csv(runner.OUT / "expanded_per_window_nll.csv", row)
        key = tuple(
            row[k] for k in ("split", "seed", "candidate", "checkpoint", "doc_id")
        )
        groups[key][0] += float(row["nll"]) * int(row["labels"])
        groups[key][1] += int(row["labels"])
        groups[key][2] += 1
    for key, (total, labels, count) in sorted(groups.items()):
        runner.record_csv(
            runner.OUT / "per_document_nll.csv",
            {
                **dict(
                    zip(("split", "seed", "candidate", "checkpoint", "doc_id"), key)
                ),
                "windows": count,
                "labels": labels,
                "nll": total / labels,
            },
        )
    for seed, entry in result["seeds"].items():
        for name, policy in entry["strategies"].items():
            runner.record_csv(
                runner.OUT / "strategy_results.csv",
                {
                    "seed": seed,
                    "strategy": name,
                    "selected": policy["selected"],
                    "e_nll": policy["endpoint_e_nll"],
                    **policy["difference_vs_ref"],
                    "quality_pass": policy["quality_pass"],
                },
            )
            runner.record_csv(
                runner.OUT / "costs.csv",
                {
                    "seed": seed,
                    "strategy": name,
                    **{
                        k: policy[k]
                        for k in (
                            "training_updates",
                            "s_query_windows",
                            "training_input_tokens",
                            "training_effective_labels",
                            "standalone_seconds_estimate",
                        )
                    },
                    "endpoint_e_windows": 64,
                },
            )
    new = data.read_json(runner.OUT / "new_trajectories.json")
    data.write_json(
        runner.OUT / "audit_costs.json",
        {
            "prior_gpu_seconds": budget["spent_seconds"]
            - budget["phases"][-1]["elapsed_seconds"],
            "additional_gpu_seconds": budget["phases"][-1]["elapsed_seconds"],
            "cumulative_gpu_seconds": budget["spent_seconds"],
            "additional_limit_seconds": 3600,
            "cumulative_limit_seconds": 14400,
            "new_training_updates": 400,
            "reused_training_updates": 1600,
            "new_input_tokens": 819200,
            "new_effective_labels": 817600,
            "unique_t_windows": 400,
            "unique_t_input_tokens": 204800,
            "new_s_windows": 1024,
            "new_e_windows": 256,
            "training_update_seconds": sum(sum(t["update_seconds"]) for t in new),
            "s_evaluation_seconds": sum(
                sum(t["eval_window_seconds"]) * 64 for t in new
            ),
            "checkpoint_seconds": sum(sum(t["checkpoint_seconds"]) for t in new),
            "training_construction_seconds": sum(
                t["mapping"]["construction_seconds"] for t in new
            ),
            "endpoint_e_costs": runtime["evaluation_costs"],
            "cpu_analysis_including_io_seconds": time.monotonic() - started,
            "policy_estimates": "same original pilot unit costs; historical preselection and I/O included; not measured speedups",
        },
    )
    lines = [
        "# Exploratory strongest-competitor coverage",
        "",
        f"Base archive: `{coverage.BASE}`.",
        f"New execution code: `{runtime['executed_code_sha']}`; later result archive commit is distinct.",
        "",
        "Four new trajectories complete; sixteen original trajectories reused read-only.",
        "E was already viewed. Intervals describe this post-hoc internal analysis, not confirmation.",
        "",
        "## Original eight (unchanged)",
        "",
        str(original["decision"]),
        "",
        "## Expanded ten",
        "",
        str(result["decision"]),
        "",
        "| Seed | B0 | Ref100 | G0 [95% interval] | Quality |",
        "| --- | --- | --- | --- | --- |",
    ]
    for seed, entry in result["seeds"].items():
        g = entry["G0"]
        lines.append(
            f"| {seed} | {entry['strategies']['B0']['selected']} | {entry['strategies']['Ref100']['selected']} | {g['point']:.9f} [{g['lower95']:.9f}, {g['upper95']:.9f}] | {entry['quality_candidate_count']}/10 |"
        )
    lines += [
        "",
        "See metrics.json for ranking, recovery, simple regret and B20/BSH results; costs.csv separates policy information budgets and standalone estimates.",
        "The parent was not recovery-trained; no compression-benefit attribution is supported.",
        "Stop after this single supplement; no automatic candidates, complex predictors or new direction.",
    ]
    with (runner.OUT / "summary.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    if time.monotonic() - started > runner.CFG["cpu_seconds_limit"]:
        raise RuntimeError("CPU analysis exceeded authorized budget")
    coverage.validate_frozen()
    data.write_json(
        runner.OUT / "state.json",
        {
            "phase": "COMPLETE",
            "decision": result["decision"],
            "executed_code_sha": runtime["executed_code_sha"],
            "ended_at_utc": runner.now(),
            "next_experiment_authorized": False,
        },
    )
