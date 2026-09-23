"""CPU contract tests; no pretrained weights, scientific results or GPU work."""

import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from paper5.experiments.canary.direction_02 import stage0_analysis as analysis
from paper5.experiments.canary.direction_02 import stage0_data as data
from paper5.experiments.canary.direction_02 import stage0_model as model_code
from paper5.experiments.canary.direction_02 import run_stage0 as runner
from paper5.experiments.canary.direction_02.stage0_resources import choose_gpu
from paper5.experiments.canary.direction_02 import stage0_resources as resources


def tiny_model() -> Qwen2ForCausalLM:
    """Small random CPU model for structural and optimizer checks only."""
    torch.manual_seed(4)
    cfg = Qwen2Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        use_cache=False,
    )
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg)


def test_physical_removal_preserves_parameter_mapping_and_skips_removed() -> None:
    model = tiny_model()
    layers = list(model.model.layers)
    calls = []
    handles = [
        layer.register_forward_pre_hook(lambda _m, _x, i=i: calls.append(i))
        for i, layer in enumerate(layers)
    ]
    mapping = model_code.prune(model, [1, 3])
    tokens = torch.randint(0, 48, (1, 8))
    with torch.no_grad():
        model(input_ids=tokens, use_cache=False)
    for handle in handles:
        handle.remove()
    assert calls == [0, 2, 4]
    assert mapping["retained_original_indices"] == [0, 2, 4]
    assert model.config.num_hidden_layers == 3
    assert (
        mapping["base_parameters_before"] - mapping["base_parameters_after"]
        == mapping["removed_parameters"]
    )
    for new, old in enumerate([0, 2, 4]):
        assert model.model.layers[new] is layers[old]
        assert model.model.layers[new].self_attn.layer_idx == new


@pytest.mark.parametrize("removed", [[1], [1, 1], [-1, 2], [1, 5]])
def test_invalid_mapping_rejected(removed: list[int]) -> None:
    with pytest.raises(ValueError):
        model_code.prune(tiny_model(), removed)


def test_lora_update_base_freeze_zero_initial_a_gradient_and_reload(
    tmp_path: Path,
) -> None:
    model = tiny_model()
    model_code.prune(model, [1, 3])
    cfg = data.config()
    model_code.attach_lora(model, cfg, 17)
    initial = model_code.adapter_state(model)
    base = model_code.base_hash(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0)
    tokens = torch.arange(16).reshape(1, 16)
    model.train()
    loss = model_code.nll(model, tokens)
    loss.backward()
    grads_a = [
        p.grad for name, p in model.named_parameters() if name.endswith("lora_a")
    ]
    grads_b = [
        p.grad for name, p in model.named_parameters() if name.endswith("lora_b")
    ]
    assert all(g is not None and torch.count_nonzero(g) == 0 for g in grads_a)
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads_b)
    torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
    optimizer.step()
    assert model_code.base_hash(model) == base
    updated = model_code.adapter_state(model)
    assert any(not torch.equal(initial[name], updated[name]) for name in initial)
    torch.save(updated, tmp_path / "adapter.pt")
    model.eval()
    with torch.no_grad():
        expected = model_code.nll(model, tokens)
    second = tiny_model()
    model_code.prune(second, [1, 3])
    model_code.attach_lora(second, cfg, 29)
    model_code.load_adapter(
        second, torch.load(tmp_path / "adapter.pt", weights_only=True)
    )
    second.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model_code.nll(second, tokens), expected, rtol=0, atol=0
        )
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        model_code.load_adapter(second, {})


def test_same_structure_budget_same_adapter_capacity() -> None:
    capacities = []
    for pair in [[0, 1], [1, 4]]:
        model = tiny_model()
        model_code.prune(model, pair)
        capacities.append(model_code.attach_lora(model, data.config(), 17))
    assert capacities[0] == capacities[1] > 0


def toy_corpus() -> bytes:
    """Include subsections and duplicate articles without crossing split boundaries."""
    raw = "".join(
        f" = Article {i} = \n words {i} "
        + "token " * 50
        + "\n = = Section = = \n more words \n"
        for i in range(100)
    )
    return (raw + raw.split(" = Article 1 = ")[0]).encode()


def test_document_split_reproducibility_and_no_cross_document_windows() -> None:
    cfg = {
        "data_seed": 20260923,
        "sequence_length": 8,
        "windows": {"T": 30, "S": 10, "E": 10},
    }

    def tokenizer(text: str) -> list[int]:
        return list(range(len(text.split())))

    arrays, manifest = data.make_dataset(toy_corpus(), tokenizer, cfg)
    arrays2, manifest2 = data.make_dataset(toy_corpus(), tokenizer, cfg)
    assert manifest == manifest2
    memberships = []
    for split, spec in manifest["splits"].items():
        np.testing.assert_array_equal(arrays[split], arrays2[split])
        membership = {w["doc_id"] for w in spec["windows"]}
        memberships.append(membership)
        assert len(membership) == cfg["windows"][split]
        for row, window in zip(arrays[split], spec["windows"]):
            assert row.tolist() == list(
                range(window["document_token_start"], window["document_token_end"])
            )
    assert all(not memberships[i] & memberships[j] for i in range(3) for j in range(i))
    assert len(manifest["documents"]) == 100


def test_insufficient_data_and_unknown_document_format_stop() -> None:
    cfg = {
        "data_seed": 17,
        "sequence_length": 512,
        "windows": {"T": 400, "S": 64, "E": 64},
    }
    with pytest.raises(ValueError, match="Insufficient"):
        data.make_dataset(toy_corpus(), lambda text: [1] * 10, cfg)
    with pytest.raises(ValueError, match="boundaries"):
        data.documents(b"no article heading")


def test_candidate_choice_never_reads_validation_labels(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "first_layer", "second_layer", "nll"]
        )
        writer.writeheader()
        for i in range(12):
            for j in range(i + 1, 12):
                for _ in range(96):
                    writer.writerow(
                        dict(
                            split="calibration",
                            first_layer=i,
                            second_layer=j,
                            nll=2 + (i + j) / 100,
                        )
                    )
                writer.writerow(
                    dict(
                        split="validation",
                        first_layer=i,
                        second_layer=j,
                        nll="MUST_NOT_PARSE",
                    )
                )
    result = data.candidates(path, data.config())
    assert result["candidates"][0]["removed"] == [0, 1]
    assert len({tuple(row["removed"]) for row in result["candidates"]}) == 8
    assert data.candidates(path, data.config()) == result


def test_training_order_is_candidate_independent_unique_and_seeded() -> None:
    order = data.training_order(400, 17)
    assert order == data.training_order(400, 17)
    assert order != data.training_order(400, 29)
    assert sorted(order) == list(range(400))
    assert len(order) * 512 == 204800
    assert len(order) * 511 == 204400


def scores_fixture() -> dict[int, dict[str, float]]:
    return {
        step: {f"a{i:02d}": float(i) for i in range(8)} for step in (0, 20, 50, 100)
    }


def test_policy_replay_cannot_use_eliminated_later_scores() -> None:
    scores = scores_fixture()
    original = analysis.replay(scores)
    changed = copy.deepcopy(scores)
    changed[50]["a07"] = -1000
    changed[100]["a07"] = -1000
    changed[100]["a03"] = -900
    result = analysis.replay(changed)
    assert result["BSH"] == original["BSH"]
    assert result["B20"] == original["B20"]
    assert result["B0"] == original["B0"]
    assert result["Ref100"]["selected"] == "a07"
    assert {k: v["training_updates"] for k, v in result.items()} == analysis.UPDATES
    assert {k: len(v["queries"]) for k, v in result.items()} == {
        "B0": 8,
        "B20": 8,
        "BSH": 14,
        "Ref100": 8,
    }
    assert all(v["endpoint"] == 100 for v in result.values())


def test_bootstrap_preserves_document_clusters_and_pairing() -> None:
    base = np.array([1.0, 3.0, 10.0])
    losses = np.stack([base, base + 0.04])
    draws = analysis.paired_document_bootstrap(
        losses, ["one", "one", "two"], np.array([511, 511, 511]), 2000, 17
    )
    np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 0.04, atol=1e-14)
    # Whole-document sampling permits only all-one, mixed or all-two, not individual windows.
    assert set(np.round(draws[:, 0], 8)) == {2.0, round(14 / 3, 8), 10.0}
    with pytest.raises(ValueError):
        analysis.paired_document_bootstrap(losses, ["one"] * 3, np.ones(3), 10, 17)


def test_negative_g0_and_inconclusive_not_equivalence() -> None:
    def row(point: float, low: float, high: float) -> dict:
        return {
            "quality_usable": True,
            "G0": {"point": point, "lower95": low, "upper95": high},
            "strategies": {
                name: {
                    "difference_vs_ref": {"upper95": 0.03},
                    "quality_pass": True,
                    "lower_cost": True,
                }
                for name in ("B20", "BSH")
            },
        }

    result = analysis.classify(
        {"17": row(-0.03, -0.04, -0.02), "29": row(-0.01, -0.02, 0)}, 0.02
    )
    assert result["problem"] == "NO_MATERIAL_SELECTION_GAP"
    assert (
        analysis.classify(
            {"17": row(0.03, -0.01, 0.08), "29": row(0.04, -0.01, 0.08)}, 0.02
        )["problem"]
        == "INCONCLUSIVE"
    )
    signal = {"17": row(0.03, 0.01, 0.06), "29": row(0.04, 0.02, 0.08)}
    assert analysis.classify(signal, 0.02)["problem"] == "PROBLEM_SIGNAL"
    signal["17"]["quality_usable"] = False
    assert analysis.classify(signal, 0.02)["verdict"] == "QUALITY_INFEASIBLE"


def test_same_simple_strategy_must_pass_both_seeds() -> None:
    seeds = {
        str(seed): {
            "quality_usable": True,
            "G0": {"point": 0.04, "lower95": 0.01, "upper95": 0.1},
            "strategies": {
                name: {
                    "difference_vs_ref": {
                        "upper95": 0.01 if (seed == 17) == (name == "B20") else 0.03
                    },
                    "quality_pass": True,
                    "lower_cost": True,
                }
                for name in ("B20", "BSH")
            },
        }
        for seed in (17, 29)
    }
    assert analysis.classify(seeds, 0.02)["simple_baseline"] == "NOT_ESTABLISHED"
    seeds["29"]["strategies"]["B20"]["difference_vs_ref"]["upper95"] = 0.01
    assert analysis.classify(seeds, 0.02)["sufficient_strategies"] == ["B20"]


def test_e_seal_fails_before_even_opening_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE", tmp_path)
    with pytest.raises(RuntimeError, match="sealed"):
        runner.load_split("E", {})
    monkeypatch.setattr(runner, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.load_split("E", {}, e_unsealed=True)


def test_frozen_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    data.write_json(tmp_path / "frozen.json", {"value": 1})
    with pytest.raises(FileExistsError):
        data.write_json(tmp_path / "frozen.json", {"value": 2})
    assert data.read_json(tmp_path / "frozen.json") == {"value": 1}


def test_complete_analysis_keeps_seeds_separate_and_reports_simple_baseline() -> None:
    cfg = data.config()
    initial = {f"a{i:02d}": 2 + i / 100 for i in range(8)}
    terminal = {f"a{i:02d}": 2 + (7 - i) / 100 for i in range(8)}
    scores = {
        str(seed): {
            str(step): initial if step == 0 else terminal for step in cfg["checkpoints"]
        }
        for seed in cfg["training_seeds"]
    }
    choices = {
        seed: analysis.replay({int(k): v for k, v in values.items()})
        for seed, values in scores.items()
    }
    e = {
        str(seed): {name: [loss] * 64 for name, loss in terminal.items()}
        for seed in cfg["training_seeds"]
    }
    e["parent"] = [2.0] * 64
    result = analysis.analyze(
        choices,
        scores,
        e,
        [f"d{i}" for i in range(64)],
        [511] * 64,
        cfg,
        {
            "update_seconds": 1.0,
            "eval_window_seconds": 0.1,
            "construction_seconds": 1.0,
            "checkpoint_seconds": 0.1,
            "historical_preselection_estimate_seconds": 100.0,
        },
    )
    assert result["decision"]["problem"] == "PROBLEM_SIGNAL"
    assert result["decision"]["simple_baseline"] == "SIMPLE_BASELINE_SUFFICIENT"
    assert set(result["seeds"]) == {"17", "29"}
    assert result["seeds"]["17"]["G0"]["point"] == pytest.approx(0.07)
    assert result["seeds"]["29"]["quality_candidate_count"] == 8
    assert result["seeds"]["17"]["spearman"] == pytest.approx(-1)


def test_gpu_selection_checks_model_and_memory_reserve() -> None:
    rows = [
        {
            "index": 0,
            "name": "RTX 3050",
            "free_mib": 50000,
            "used_mib": 0,
            "utilization_percent": 0,
            "existing_compute_process_count": 0,
        },
        {
            "index": 1,
            "name": "NVIDIA A800",
            "free_mib": 30000,
            "used_mib": 50000,
            "utilization_percent": 80,
            "existing_compute_process_count": 1,
        },
        {
            "index": 2,
            "name": "NVIDIA A800",
            "free_mib": 70000,
            "used_mib": 10000,
            "utilization_percent": 10,
            "existing_compute_process_count": 1,
        },
    ]
    allocation = {"mode": "shared", "gpu_indices": [1, 2]}
    assert choose_gpu(rows, 24576, allocation)["index"] == 2
    with pytest.raises(RuntimeError):
        choose_gpu(rows, 75000, allocation)
    with pytest.raises(RuntimeError, match="allocation"):
        choose_gpu(rows, 24576)
    with pytest.raises(RuntimeError):
        choose_gpu(rows, 24576, {"mode": "exclusive", "gpu_indices": [1, 2]})


def test_idle_gpu_requires_permission_and_busy_gpu_requires_shared_mode() -> None:
    rows = [
        {
            "index": 0,
            "name": "A800",
            "free_mib": 47000,
            "used_mib": 34000,
            "utilization_percent": 100,
            "existing_compute_process_count": 1,
        },
        {
            "index": 1,
            "name": "A800",
            "free_mib": 81000,
            "used_mib": 10,
            "utilization_percent": 0,
            "existing_compute_process_count": 0,
        },
    ]
    grant = {"mode": "exclusive", "gpu_indices": [0, 1]}
    assert choose_gpu(rows, 24576, grant)["index"] == 1
    with pytest.raises(RuntimeError):
        choose_gpu(rows, 24576, {"mode": "exclusive", "gpu_indices": [0]})
    assert choose_gpu(rows, 24576, {"mode": "shared", "gpu_indices": [0]})["index"] == 0


def test_permission_record_must_cover_stage_task_and_expiry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("D2_GPU_PERMISSION_FILE", raising=False)
    with pytest.raises(RuntimeError, match="permission"):
        resources.permission("pilot")
    path = tmp_path / "grant.json"
    record = {
        "run_id": data.config()["run_id"],
        "mode": "shared",
        "gpu_indices": [0],
        "stages": ["pilot"],
        "evidence": "synthetic test allocation, not a real permit",
        "expires_at_utc": "2999-01-01T00:00:00+00:00",
    }
    data.write_json(path, record)
    monkeypatch.setenv("D2_GPU_PERMISSION_FILE", str(path))
    assert resources.permission("pilot") == record
    with pytest.raises(RuntimeError):
        resources.permission("formal")
    record["expires_at_utc"] = "2000-01-01T00:00:00+00:00"
    data.write_json(path, record, replace=True)
    with pytest.raises(RuntimeError, match="expired"):
        resources.permission("pilot")


def test_resource_recheck_and_duplicate_launch_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grant = {"mode": "exclusive", "gpu_indices": [0]}
    monkeypatch.setattr(resources, "permission", lambda stage: grant)
    monkeypatch.setattr(resources, "existing_task_processes", lambda: [123])
    with pytest.raises(RuntimeError, match="duplicate"):
        resources.admission("pilot")
    monkeypatch.setattr(resources, "existing_task_processes", lambda: [])
    idle = {
        "index": 0,
        "name": "A800",
        "free_mib": 81000,
        "used_mib": 10,
        "utilization_percent": 0,
        "existing_compute_process_count": 0,
    }
    busy = {**idle, "utilization_percent": 100, "existing_compute_process_count": 1}
    states = iter([[idle], [busy]])
    monkeypatch.setattr(resources, "inspect_gpus", lambda: next(states))
    with pytest.raises(RuntimeError):
        resources.admission("pilot")


def test_budget_rejects_duplicate_and_exhausted_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "CACHE", tmp_path)
    data.write_json(
        tmp_path / "gpu_ledger.json",
        {"spent_seconds": 10, "phases": [{"stage": "pilot"}]},
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        runner.GpuBudget("pilot")
    data.write_json(
        tmp_path / "gpu_ledger.json",
        {"spent_seconds": 14390, "phases": []},
        replace=True,
    )
    with pytest.raises(runner.BudgetStop):
        runner.GpuBudget("formal")


def test_freeze_rejects_failed_quality_without_relaxing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "validate_artifacts", lambda: None)
    data.write_json(
        tmp_path / "pilot.json",
        {
            "engineering_passed": True,
            "budget_feasible": True,
            "quality_proposal": {"formal_entry_quality_passed": False},
        },
    )
    data.write_json(tmp_path / "gpu_budget.json", {"phases": [{"status": "COMPLETE"}]})
    with pytest.raises(RuntimeError, match="gate"):
        runner.freeze()
    assert not (tmp_path / "formal_freeze.json").exists()


def test_missing_trajectory_prevents_e_unseal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "OUT", tmp_path)
    data.write_json(tmp_path / "selection.json", {})
    data.write_json(tmp_path / "trajectories.json", [])
    with pytest.raises(RuntimeError, match="Incomplete"):
        runner.validate_selection()
