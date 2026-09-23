"""Resume one authorized pilot checkpoint; S only, no formal Stage0 entry."""

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from . import run_stage0 as original
from . import stage0_data as data
from . import stage0_resources as resources

CONFIG = data.ROOT / "paper5/configs/canary/direction_02/recovery_capacity.json"
PROTOCOL = Path(__file__).with_name("RECOVERY_CAPACITY_PROTOCOL.md")
SPEC = data.read_json(CONFIG)
SOURCE_OUT, SOURCE_CACHE = original.OUT, original.CACHE
OUT = SOURCE_OUT.parent.parent / "recovery_capacity" / SPEC["run_id"]
CACHE = SOURCE_CACHE.parent / "recovery_capacity" / SPEC["run_id"]


def training_indices(order: list[int], completed: int) -> list[int]:
    """Continue the original order without repeating the consumed 40 windows."""
    if len(order) != 400 or len(set(order)) != 400 or not 10 <= completed < 100:
        raise ValueError("Invalid continuation order or update")
    return order[completed * 4 : (completed + 1) * 4]


def validate_checkpoint(saved: dict[str, Any]) -> None:
    """Bind the continuation to the exact pilot and complete AdamW state."""
    expected = {
        "step": 10,
        "seed": 17,
        "candidate": "a00",
        "executed_code_sha": SPEC["source_code_sha"],
        "config_sha256": SPEC["source_config_sha256"],
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Resume checkpoint provenance mismatch")
    if saved["training_order"] != data.training_order(400, 17):
        raise ValueError("Resume training order changed")
    if saved["mapping"]["removed_original_indices"] != [12, 15]:
        raise ValueError("Resume candidate mapping changed")
    groups = saved["optimizer"]["param_groups"]
    state = saved["optimizer"]["state"]
    if len(groups) != 1 or len(state) != len(saved["adapter"]):
        raise ValueError("Incomplete optimizer state")
    expected_group = {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }
    if any(groups[0].get(key) != value for key, value in expected_group.items()):
        raise ValueError("Optimizer recipe changed")
    if any(
        float(row["step"]) != 10 or not {"exp_avg", "exp_avg_sq"} <= row.keys()
        for row in state.values()
    ):
        raise ValueError("Optimizer is not at step 10")
    if "cpu_rng" not in saved or len(saved.get("cuda_rng", [])) != 1:
        raise ValueError("Incomplete RNG state")


def restore(model: Any, optimizer: Any, saved: dict[str, Any], *, cuda: bool) -> None:
    """Restore parameters, optimizer and RNG after constructing the same model."""
    import torch
    from .stage0_model import load_adapter

    load_adapter(model, saved["adapter"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    if cuda:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def load_ts(split: str, manifest: dict[str, Any]) -> np.ndarray:
    """This diagnostic never exposes the original E-unsealing option."""
    if split not in {"T", "S"}:
        raise ValueError("Only T and S are authorized")
    return original.load_split(split, manifest)


def preserved_files() -> dict[str, str]:
    """Fingerprint original results and scientific inputs, without opening E."""
    paths = list(SOURCE_OUT.glob("*")) + list(
        (data.ROOT / "paper5/results/canary/direction_01").rglob("*")
    )
    paths += [data.CONFIG, original.PROTOCOL, Path(original.__file__)]
    return {
        str(p.relative_to(data.ROOT)): data.file_hash(p)
        for p in sorted(paths)
        if p.is_file()
    }


def run(snapshot: Path) -> None:
    """Run exactly one checkpoint continuation under the original total budget."""
    import torch
    from . import stage0_model as implementation

    executed = original.code_sha()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "2"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise RuntimeError("Physical GPU 2 with PCI_BUS_ID ordering is required")
    if data.file_hash(data.CONFIG) != SPEC["source_config_sha256"]:
        raise RuntimeError("Original recipe changed")
    manifest, candidates = original.validate_artifacts()
    pilot = data.read_json(SOURCE_OUT / "pilot.json")
    checkpoint_path = SOURCE_CACHE / "pilot_adapter/17/a00/step_10.pt"
    if data.file_hash(checkpoint_path) != SPEC["source_checkpoint_sha256"]:
        raise RuntimeError("Original checkpoint changed")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    if candidates["candidates"][0]["id"] != "a00" or candidates["candidates"][0][
        "removed"
    ] != [12, 15]:
        raise RuntimeError("Original candidate manifest changed")
    train, selection = load_ts("T", manifest), load_ts("S", manifest)
    preserved = preserved_files()
    source_ledger_path = SOURCE_CACHE / "gpu_ledger.json"
    source_ledger_hash = data.file_hash(source_ledger_path)
    ledger = data.read_json(source_ledger_path)
    if ledger != data.read_json(SOURCE_OUT / "gpu_budget.json"):
        raise RuntimeError("Original GPU ledgers disagree")
    estimate = (
        1.5
        * (
            90 * pilot["unit_costs"]["update_seconds"]
            + 193 * pilot["unit_costs"]["eval_window_seconds"]
            + 4 * pilot["unit_costs"]["construction_seconds"]
        )
        + 300
    )
    if estimate > original.CFG["gpu_seconds_limit"] - ledger["spent_seconds"]:
        raise RuntimeError("Continuation does not fit remaining original budget")
    samples = []
    for index in range(3):
        gpu, grant, required = resources.admission("capacity", selected=2)
        if gpu["uuid"] != SPEC["gpu_uuid"] or grant["gpu_indices"] != [2]:
            raise RuntimeError("Wrong physical GPU or permission scope")
        samples.append(gpu)
        if index < 2:
            time.sleep(2)
    CACHE.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    data.write_json(CACHE / "gpu_ledger.json", ledger)
    data.write_json(
        OUT / "config_used.json", {"diagnostic": SPEC, "recipe": original.CFG}
    )
    data.write_json(OUT / "data_manifest.json", manifest)
    data.write_json(
        OUT / "candidate_manifest.json",
        {
            "candidate": candidates["candidates"][0],
            "source_manifest_sha256": data.file_hash(
                SOURCE_OUT / "candidate_manifest.json"
            ),
            "selection": "same already-piloted candidate; no new selection",
        },
    )
    data.write_json(
        OUT / "freeze.json",
        {
            "executed_code_sha": executed,
            "source_checkpoint_sha256": SPEC["source_checkpoint_sha256"],
            "protocol_sha256": data.file_hash(PROTOCOL),
            "config_sha256": data.file_hash(CONFIG),
            "source_data_manifest_sha256": data.file_hash(
                SOURCE_OUT / "data_manifest.json"
            ),
            "preserved_sha256": preserved,
            "source_gpu_ledger_sha256": source_ledger_hash,
            "training_order_sha256": data.digest(
                data.canonical(saved["training_order"])
            ),
            "source_code_sha": SPEC["source_code_sha"],
            "source_archive_sha": SPEC["source_archive_sha"],
            "e_accessed": False,
            "estimated_additional_gpu_seconds": estimate,
        },
    )
    data.write_json(
        OUT / "resource_preflight.json",
        {
            "samples": samples,
            "required_free_mib": required,
            "permission_mode": grant["mode"],
            "permission_stage": "capacity",
            "physical_gpu": 2,
            "gpu_uuid": gpu["uuid"],
            "explicit_user_shared_permission": True,
        },
    )
    # Only helper destinations change in this process; original files remain immutable.
    original.OUT, original.CACHE = OUT, CACHE
    budget = original.GpuBudget("capacity")
    status = "FAILED"
    completed = 10
    model = optimizer = None
    mapping = saved["mapping"]
    order = saved["training_order"]
    try:
        original.check_snapshot(snapshot)
        budget.check()
        model, mapping = original.load_model(snapshot, [12, 15], 17)
        if any(
            mapping[k] != v
            for k, v in saved["mapping"].items()
            if k != "construction_seconds"
        ):
            raise RuntimeError("Restored model mapping differs from pilot")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        restore(model, optimizer, saved, cuda=True)
        base_hash = implementation.base_hash(model)
        if base_hash != pilot["trajectory"]["base_sha256"]:
            raise RuntimeError("Restored frozen base differs from pilot")
        initial = implementation.adapter_state(model)
        values, _ = original.evaluate(
            model,
            selection[:1],
            manifest["splits"]["S"]["windows"][:1],
            "S",
            17,
            "a00",
            10,
            budget,
            "resume_check_nll.csv",
        )
        with (SOURCE_OUT / "pilot_nll.csv").open(newline="") as handle:
            expected = next(
                float(row["nll"])
                for row in csv.DictReader(handle)
                if row["candidate"] == "a00" and row["checkpoint"] == "10"
            )
        resume_difference = abs(values[0] - expected)
        if resume_difference > 1e-6:
            raise RuntimeError("Step-10 reload NLL differs from original")
        restore(model, optimizer, saved, cuda=True)
        torch.cuda.reset_peak_memory_stats()
        metrics = []
        for step in range(10, 100):
            budget.check()
            gpu = next(row for row in resources.inspect_gpus() if row["index"] == 2)
            if (
                gpu["uuid"] != SPEC["gpu_uuid"]
                or gpu["free_mib"] < original.CFG["memory_reserve_mib"]
            ):
                raise RuntimeError("GPU resource reserve no longer available")
            start = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for item in training_indices(order, step):
                budget.check()
                tokens = torch.from_numpy(train[item].copy()).unsqueeze(0).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = implementation.nll(model, tokens)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                losses.append(float(loss.detach()))
                (loss / 4).backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if any(
                p.grad is not None
                for name, p in model.named_parameters()
                if not name.endswith(("lora_a", "lora_b"))
            ):
                raise RuntimeError("Base received a gradient")
            optimizer.step()
            torch.cuda.synchronize()
            completed = step + 1
            original.record_csv(
                OUT / "training_curve.csv",
                {
                    "candidate": "a00",
                    "seed": 17,
                    "update": completed,
                    "train_nll": float(np.mean(losses)),
                    "gradient_norm": float(norm),
                    "update_seconds": time.monotonic() - start,
                    "cumulative_input_tokens": completed * 2048,
                    "cumulative_effective_labels": completed * 2044,
                },
            )
            if completed in SPEC["s_checkpoints"]:
                save_seconds = original.checkpoint(
                    model,
                    optimizer,
                    CACHE / f"step_{completed}.pt",
                    completed,
                    17,
                    "a00",
                    order,
                    mapping,
                    executed,
                )
                values, window_seconds = original.evaluate(
                    model,
                    selection,
                    manifest["splits"]["S"]["windows"],
                    "S",
                    17,
                    "a00",
                    completed,
                    budget,
                )
                score = float(np.mean(values))
                parent = pilot["parent_s_nll"]
                row = {
                    "step": completed,
                    "s_nll": score,
                    "s_ppl": math.exp(score),
                    "parent_s_nll": parent,
                    "nll_ratio": score / parent,
                    "ppl_ratio": math.exp(score - parent),
                    "quality_passed": math.exp(score - parent) <= 1.15,
                    "checkpoint_seconds": save_seconds,
                    "eval_window_seconds": window_seconds,
                }
                metrics.append(row)
                original.record_csv(OUT / "metrics.csv", row)
                print(
                    f"completed={completed} S_NLL={score:.9f} PPL_ratio={row['ppl_ratio']:.9f}",
                    flush=True,
                )
        current = implementation.adapter_state(model)
        delta = sum(float((current[k] - initial[k]).square().sum()) for k in current)
        if implementation.base_hash(model) != base_hash or delta <= 0:
            raise RuntimeError("Base mutated or adapter did not update")
        if (
            preserved_files() != preserved
            or data.file_hash(checkpoint_path) != SPEC["source_checkpoint_sha256"]
            or data.file_hash(source_ledger_path) != source_ledger_hash
        ):
            raise RuntimeError("Original artifacts changed")
        decision = (
            "EXPLORATORY_CAPACITY_FEASIBLE"
            if metrics[-1]["quality_passed"]
            else "EXPLORATORY_CAPACITY_QUALITY_NOT_MET"
        )
        data.write_json(
            OUT / "summary.json",
            {
                "status": decision,
                "exploratory": True,
                "executed_code_sha": executed,
                "source_code_sha": SPEC["source_code_sha"],
                "candidate": "a00",
                "seed": 17,
                "resumed_from_step": 10,
                "new_updates": 90,
                "endpoint": 100,
                "historical_s_scores": pilot["trajectory"]["s_scores"],
                "metrics": metrics,
                "quality_max_ppl_ratio": 1.15,
                "base_unchanged": True,
                "base_sha256": base_hash,
                "adapter_squared_change_from_step10": delta,
                "resume_absolute_nll_difference": resume_difference,
                "source_artifacts_unchanged": True,
                "e_accessed": False,
                "formal_started": False,
                "additional_input_tokens": 184320,
                "additional_effective_labels": 183960,
                "unique_training_windows_total": 400,
                "unique_training_windows_additional": 360,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "checkpoints_sha256": {
                    str(s): data.file_hash(CACHE / f"step_{s}.pt")
                    for s in [20, 50, 100]
                },
                "next_action": "STOP; any formal Stage0 requires renewed user approval",
            },
        )
        status = "COMPLETE"
    except BaseException as error:
        if model is not None and optimizer is not None:
            original.checkpoint(
                model,
                optimizer,
                CACHE / "interrupted.pt",
                completed,
                17,
                "a00",
                order,
                mapping,
                executed,
            )
        data.write_json(
            OUT / "failure.json",
            {
                "status": "STOPPED",
                "error_type": type(error).__name__,
                "completed_step": completed,
                "e_accessed": False,
            },
        )
        raise
    finally:
        budget.finish(status)
        data.write_json(
            OUT / "runtime.json",
            {
                "executed_code_sha": executed,
                "pid": os.getpid(),
                "status": status,
                "ended_at_utc": original.now(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_uuid": SPEC["gpu_uuid"],
                "physical_gpu": 2,
                "original_pilot_seconds": ledger["spent_seconds"],
                "diagnostic_seconds": budget.record["phases"][-1]["elapsed_seconds"],
                "cumulative_gpu_seconds": budget.record["spent_seconds"],
            },
        )


def main() -> None:
    """Hold the original task lock for the complete single diagnostic process."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    with ExitStack() as stack:
        lock = stack.enter_context((SOURCE_CACHE / "task.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.snapshot)


if __name__ == "__main__":
    main()
