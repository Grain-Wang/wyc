"""One post-V2 supplement: four new trajectories, read-only reuse of sixteen."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import math
import os
import shutil
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from . import run_stage0 as runner
from . import stage0_analysis as analysis
from . import stage0_data as data
from . import stage0_quality_v2 as v2

CONFIG = (
    data.ROOT / "paper5/configs/canary/direction_02/recovery_competitor_coverage.json"
)
PROTOCOL = Path(__file__).with_name("STAGE0_COMPETITOR_COVERAGE.md")
BASE = "123cd0ca2c7011a8bd1560aee47984954fe09a20"
SOURCE = v2.RESULTS / "recovery_stage0_v2/20260923_stage0_quality_v2_01"
NEW = [{"id": "a08", "removed": [10, 15]}, {"id": "a09", "removed": [10, 12]}]
SIZES = (10, 5, 2)


def activate() -> None:
    """Reuse the frozen scientific recipe with independent outputs and state."""
    cfg = data.read_json(CONFIG)
    original = data.read_json(v2.CONFIG)
    excluded = {"run_id", "design_version", "candidate_count", "coverage"}
    if {k: v for k, v in cfg.items() if k not in excluded} != {
        k: v for k, v in original.items() if k not in excluded
    }:
        raise RuntimeError("Scientific recipe differs from V2")
    if (
        cfg["run_id"] != "20260924_competitor_coverage_01"
        or cfg["design_version"] != "D2_STAGE0_COMPETITOR_COVERAGE_V1"
        or cfg["candidate_count"] != 10
        or cfg["coverage"]["new_candidates"] != NEW
        or cfg["coverage"]["base_archive_sha"] != BASE
        or cfg["coverage"]["additional_gpu_seconds_limit"] != 3600
        or cfg["coverage"]["policy_sizes"] != list(SIZES)
    ):
        raise RuntimeError("Unauthorized supplement scope")
    base = Path(
        os.environ.get("D2_CACHE", str(data.ROOT / "paper5/.cache/direction_02"))
    )
    runner.CFG = cfg
    runner.OUT = v2.RESULTS / "recovery_competitor_coverage" / cfg["run_id"]
    runner.CACHE = base / "recovery_competitor_coverage" / cfg["run_id"]
    runner.TOKEN_CACHE = base / original["quality_admission"]["source_run_id"]
    runner.EXEC_CONFIG, runner.EXEC_PROTOCOL = CONFIG, PROTOCOL
    runner.QUALITY_V2 = False


def old_cache() -> Path:
    """Original V2 checkpoint/ledger directory, always read-only."""
    return runner.TOKEN_CACHE.parent / "recovery_stage0_v2" / SOURCE.name


def verify_coverage() -> dict[str, Any]:
    """Reproduce calibration-only ranking and original deterministic sampling."""
    old = data.read_json(SOURCE / "candidate_manifest.json")
    if data.candidates(data.D1 / "per_window_nll.csv", data.config()) != old:
        raise RuntimeError("Original sampling rule does not reproduce manifest")
    selected = [r["removed"] for r in old["candidates"]]
    if (
        old["competitive_pool"][:3] != [[12, 15], [10, 15], [10, 12]]
        or selected[0] != [12, 15]
        or any(c["removed"] in selected for c in NEW)
    ):
        raise RuntimeError("Claimed coverage gap is false; stop")
    return {
        "base_archive_sha": BASE,
        "original_manifest_sha256": data.file_hash(SOURCE / "candidate_manifest.json"),
        "original_rule_reproduced": True,
        "competitive_pool": old["competitive_pool"],
        "candidates": old["candidates"] + NEW,
        "new_candidates": NEW,
        "layer_numbering": "original parent zero-based",
        "selection_basis": "fixed historical calibration ranks 2 and 3; post-V2 authorization",
        "e_previously_viewed": True,
    }


def projection() -> dict[str, Any]:
    """Bound four trajectories using archived measured units and unchanged reserves."""
    ledger = data.read_json(SOURCE / "gpu_budget.json")
    if (
        [p["stage"] for p in ledger["phases"]] != ["pilot", "capacity", "formal"]
        or any(p["status"] != "COMPLETE" for p in ledger["phases"])
        or not math.isclose(
            sum(p["elapsed_seconds"] for p in ledger["phases"]),
            ledger["spent_seconds"],
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("Incomplete or inconsistent previous GPU ledger")
    units = dict(data.read_json(SOURCE / "budget_projection.json")["unit_cost_bounds"])
    trajectories = data.read_json(SOURCE / "trajectories.json")
    for key in ("update_seconds", "eval_window_seconds", "checkpoint_seconds"):
        units[key] = max(units[key], *(max(t[key]) for t in trajectories))
    units["construction_seconds"] = max(
        units["construction_seconds"],
        *(t["mapping"]["construction_seconds"] for t in trajectories),
    )
    raw = (
        400 * units["update_seconds"]
        + 1280 * units["eval_window_seconds"]
        + 8 * units["construction_seconds"]
        + 16 * units["checkpoint_seconds"]
    )
    projected = (
        raw * runner.CFG["budget_safety_factor"]
        + runner.CFG["budget_fixed_reserve_seconds"]
    )
    limit = min(3600, 14400 - ledger["spent_seconds"])
    if projected > limit:
        raise RuntimeError(
            "Four complete trajectories/evaluation exceed authorized budget"
        )
    return {
        "prior_gpu_seconds": ledger["spent_seconds"],
        "projected_additional_gpu_seconds": projected,
        "projected_cumulative_gpu_seconds": ledger["spent_seconds"] + projected,
        "additional_limit_seconds": 3600,
        "cumulative_limit_seconds": 14400,
        "unit_cost_bounds": units,
        "counts": {
            "updates": 400,
            "s_windows": 1024,
            "e_windows": 256,
            "constructions": 8,
            "checkpoint_writes": 16,
        },
    }


def freeze() -> None:
    """Commit the independently versioned plan and unchanged historical inputs."""
    manifest = verify_coverage()
    budget = projection()
    preserved = {}
    roots = [v2.RESULTS.parent / "direction_01", v2.SOURCE, v2.CAPACITY, SOURCE]
    paths = [
        p
        for root in roots
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix != ".log"
    ]
    paths += [data.CONFIG, runner.PROTOCOL, v2.CONFIG, v2.PROTOCOL]
    for path in paths:
        relative = str(path.relative_to(data.ROOT))
        expected = subprocess.check_output(
            ["git", "show", f"{BASE}:{relative}"], cwd=data.ROOT
        )
        if expected != path.read_bytes():
            raise RuntimeError("Historical input differs from base archive")
        preserved[relative] = data.file_hash(path)
    runner.OUT.mkdir(parents=True, exist_ok=False)
    for name in ("data_manifest.json", "pilot.json"):
        shutil.copyfile(SOURCE / name, runner.OUT / name)
    shutil.copyfile(SOURCE / "gpu_budget.json", runner.OUT / "prior_gpu_budget.json")
    data.write_json(runner.OUT / "config_used.json", runner.CFG)
    data.write_json(runner.OUT / "candidate_manifest.json", manifest)
    data.write_json(runner.OUT / "budget_projection.json", budget)
    data.write_json(
        runner.OUT / "freeze.json",
        {
            "base_archive_sha": BASE,
            "frozen_at_utc": runner.now(),
            "e_previously_viewed": True,
            "new_model_forwards": 0,
            "files_sha256": {
                str(p.relative_to(data.ROOT)): data.file_hash(p)
                for p in [
                    CONFIG,
                    PROTOCOL,
                    *(
                        runner.OUT / name
                        for name in (
                            "config_used.json",
                            "candidate_manifest.json",
                            "data_manifest.json",
                            "prior_gpu_budget.json",
                            "budget_projection.json",
                        )
                    ),
                ]
            },
            "preserved_source_sha256": preserved,
        },
    )


def validate_frozen(executed: str | None = None) -> None:
    """Reject any historical evidence or frozen input mutation."""
    frozen = data.read_json(runner.OUT / "freeze.json")
    for group in ("files_sha256", "preserved_source_sha256"):
        for name, expected in frozen[group].items():
            if data.file_hash(data.ROOT / name) != expected:
                raise RuntimeError(f"Frozen input changed: {name}")
    if executed:
        path = runner.OUT / "freeze.json"
        committed = subprocess.check_output(
            ["git", "show", f"{executed}:{path.relative_to(data.ROOT)}"], cwd=data.ROOT
        )
        if committed != path.read_bytes():
            raise RuntimeError("Freeze must be in execution commit")


def validate_complete(
    complete: list[dict[str, Any]], *, endpoints: bool = True
) -> None:
    """No incomplete or surviving-only expanded set may reach E or analysis."""
    expected = {(seed, f"a{i:02d}") for seed in (17, 29) for i in range(10)}
    if (
        len(complete) != 20
        or {(t["seed"], t["candidate"]) for t in complete} != expected
    ):
        raise RuntimeError("Twenty complete trajectory records required")
    pool = {
        c["id"]: c["removed"]
        for c in data.read_json(runner.OUT / "candidate_manifest.json")["candidates"]
    }
    for t in complete:
        if (
            t["updates"] != 100
            or not t["base_unchanged"]
            or set(t["s_scores"]) != {"0", "20", "50", "100"}
        ):
            raise RuntimeError("Invalid trajectory endpoint")
        if t["mapping"]["removed_original_indices"] != pool[t["candidate"]]:
            raise RuntimeError("Candidate mapping mismatch")
        if t["mapping"]["adapter_parameters"] != 1011712:
            raise RuntimeError("Adapter capacity changed")
        if t["training_order_sha256"] != data.digest(
            data.canonical(data.training_order(400, t["seed"]))
        ):
            raise RuntimeError("Training order changed")
        if endpoints:
            cache = runner.CACHE if t["candidate"] in {"a08", "a09"} else old_cache()
            if (
                data.file_hash(
                    cache / "adapters" / str(t["seed"]) / t["candidate"] / "step_100.pt"
                )
                != t["endpoint_sha256"]
            ):
                raise RuntimeError("Endpoint missing or changed")


def validate_selection() -> dict[str, Any]:
    """Expanded S-only choices must be durable before any new E access."""
    record = data.read_json(runner.OUT / "selection.json")
    complete = data.read_json(runner.OUT / "trajectories.json")
    validate_complete(complete)
    scores = data.read_json(runner.OUT / "s_scores.json")
    for seed in ("17", "29"):
        for step in ("0", "20", "50", "100"):
            expected = {
                t["candidate"]: t["s_scores"][step]
                for t in complete
                if str(t["seed"]) == seed
            }
            if scores[seed][step] != expected:
                raise RuntimeError("Scores differ from actual completed trajectories")
    choices = {
        seed: analysis.replay(
            {int(k): v for k, v in points.items()}, policy_sizes=SIZES
        )
        for seed, points in scores.items()
    }
    if record["choices"] != choices or any(
        record[key] != data.file_hash(runner.OUT / file)
        for key, file in (
            ("trajectory_sha256", "trajectories.json"),
            ("s_scores_sha256", "s_scores.json"),
        )
    ):
        raise RuntimeError("Persisted choices/hash differ from S-only replay")
    return record


def open_e(manifest: dict[str, Any]) -> np.ndarray:
    """Validate the expanded barrier, then verify and read identical E tokens."""
    validate_selection()
    path = runner.TOKEN_CACHE / "E.sealed.npy"
    spec = manifest["splits"]["E"]
    if data.file_hash(path) != spec["npy_sha256"]:
        raise RuntimeError("E token file changed")
    values = np.load(path, allow_pickle=False)
    if (
        values.shape != (64, 512)
        or data.digest(values.tobytes()) != spec["token_data_sha256"]
    ):
        raise RuntimeError("E token coordinates changed")
    return values


class CoverageBudget(runner.GpuBudget):
    """Retain original cumulative accounting and add the one-hour phase cap."""

    def __init__(self) -> None:
        super().__init__("competitor_coverage")
        self.limit = min(self.limit, 3600)
        remaining = self.limit - (time.monotonic() - self.start)
        if remaining <= 30:
            raise runner.BudgetStop("Startup consumed additional budget")
        signal.setitimer(signal.ITIMER_REAL, remaining - 20)

    def check(self) -> None:
        if time.monotonic() >= getattr(self, "next_resource_check", 0):
            v2.check_live_resource()
            self.next_resource_check = time.monotonic() + 30
        super().check()


def preflight() -> None:
    """Reuse explicit permission checks, fixed GPU2, prior ledger and duplicate lock."""
    from . import stage0_resources as resources

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "2"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise RuntimeError("Only physical GPU2 is authorized")
    resources.CFG, resources.OUT, resources.CACHE = runner.CFG, runner.OUT, runner.CACHE
    if (old_cache() / "gpu_ledger.json").read_bytes() != (
        runner.OUT / "prior_gpu_budget.json"
    ).read_bytes():
        raise RuntimeError("Prior cumulative GPU budget changed")
    if (runner.CACHE / "gpu_ledger.json").exists():
        raise RuntimeError("Supplement already attempted; duplicate refused")
    samples = []
    for i in range(3):
        gpu, allocation, required = resources.admission(
            "competitor_coverage", selected=2
        )
        if (
            gpu["uuid"] != v2.GPU_UUID
            or allocation["gpu_indices"] != [2]
            or allocation["mode"] != "shared"
        ):
            raise RuntimeError("Physical GPU2 sharing permission mismatch")
        samples.append(gpu)
        if i < 2:
            time.sleep(2)
    data.write_json(
        runner.OUT / "resource_preflight.json",
        {
            "samples": samples,
            "required_free_mib": required,
            "permission_sha256": data.digest(data.canonical(allocation)),
            "mode": "shared",
            "other_processes_terminated": False,
        },
    )
    shutil.copyfile(
        runner.OUT / "prior_gpu_budget.json", runner.CACHE / "gpu_ledger.json"
    )


def execute(snapshot: Path, budget: CoverageBudget, executed: str) -> None:
    """Only the two fixed additions train; old numerical rows are read-only inputs."""
    import torch
    from .stage0_model import load_adapter

    manifest = data.read_json(runner.OUT / "data_manifest.json")
    train, selection = runner.load_split("T", manifest), runner.load_split(
        "S", manifest
    )
    new = []
    for seed in (17, 29):
        for candidate in NEW:
            budget.check()
            model, mapping = runner.load_model(snapshot, candidate["removed"], seed)
            result = runner.train_trajectory(
                model,
                candidate["id"],
                seed,
                100,
                train,
                selection,
                manifest["splits"]["S"]["windows"],
                mapping,
                budget,
                executed,
            )
            new.append(result)
            data.write_json(
                runner.CACHE / "trajectory_progress.json", new, replace=True
            )
            runner.release(model)
            del model
    data.write_json(runner.OUT / "new_trajectories.json", new)
    complete = data.read_json(SOURCE / "trajectories.json") + new
    validate_complete(complete)
    scores = data.read_json(SOURCE / "s_scores.json")
    for t in new:
        for step, loss in t["s_scores"].items():
            scores[str(t["seed"])][step][t["candidate"]] = loss
    data.write_json(runner.OUT / "trajectories.json", complete)
    data.write_json(runner.OUT / "s_scores.json", scores)
    data.write_json(
        runner.OUT / "selection.json",
        {
            "frozen_at_utc": runner.now(),
            "executed_code_sha": executed,
            "trajectory_sha256": data.file_hash(runner.OUT / "trajectories.json"),
            "s_scores_sha256": data.file_hash(runner.OUT / "s_scores.json"),
            "e_previously_viewed": True,
            "new_e_evaluations_before_freeze": 0,
            "choices": {
                seed: analysis.replay(
                    {int(k): v for k, v in points.items()}, policy_sizes=SIZES
                )
                for seed, points in scores.items()
            },
        },
    )
    evaluation = open_e(manifest)
    data.write_json(
        runner.OUT / "e_unsealed.json",
        {
            "at_utc": runner.now(),
            "selection_sha256": data.file_hash(runner.OUT / "selection.json"),
            "completed_new_trajectories": 4,
            "reused_trajectories": 16,
        },
    )
    e_losses = data.read_json(SOURCE / "e_losses.json")
    evaluation_costs = []
    for seed in (17, 29):
        for candidate in NEW:
            budget.check()
            model, mapping = runner.load_model(snapshot, candidate["removed"], seed)
            load_start = time.monotonic()
            state = torch.load(
                runner.CACHE / "adapters" / str(seed) / candidate["id"] / "step_100.pt",
                map_location="cpu",
                weights_only=True,
            )
            if (
                state["step"] != 100
                or state["seed"] != seed
                or state["candidate"] != candidate["id"]
                or state["executed_code_sha"] != executed
                or state["config_sha256"] != data.file_hash(CONFIG)
            ):
                raise RuntimeError("Endpoint recovery identity mismatch")
            load_adapter(model, state["adapter"])
            load_seconds = time.monotonic() - load_start
            values, duration = runner.evaluate(
                model,
                evaluation,
                manifest["splits"]["E"]["windows"],
                "E",
                seed,
                candidate["id"],
                100,
                budget,
            )
            e_losses[str(seed)][candidate["id"]] = values
            evaluation_costs.append(
                {
                    "seed": seed,
                    "candidate": candidate["id"],
                    "construction_seconds": mapping["construction_seconds"],
                    "adapter_load_seconds": load_seconds,
                    "e_evaluation_seconds": duration * 64,
                }
            )
            runner.release(model)
            del model
    data.write_json(runner.OUT / "e_losses.json", e_losses)
    data.write_json(
        runner.OUT / "runtime.json",
        {
            "phase": "EVALUATED",
            "executed_code_sha": executed,
            "base_archive_sha": BASE,
            "reused_execution_code_sha": data.read_json(SOURCE / "runtime.json")[
                "executed_code_sha"
            ],
            "design_version": runner.CFG["design_version"],
            "finished_at_utc": runner.now(),
            "gpu_name": torch.cuda.get_device_name(0),
            "physical_gpu": 2,
            "gpu_uuid": v2.GPU_UUID,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "numpy",
                    "scipy",
                    "tokenizers",
                    "safetensors",
                )
            },
            "new_trajectory_count": 4,
            "reused_trajectory_count": 16,
            "new_updates": 400,
            "new_s_forward_windows": 1024,
            "new_e_forward_windows": 256,
            "evaluation_costs": evaluation_costs,
        },
    )


def gpu_stage(snapshot: Path) -> None:
    """One attempt, protected by permission, frozen evidence and cumulative time."""
    executed = runner.code_sha()
    validate_frozen(executed)
    preflight()
    budget = CoverageBudget()
    status = "FAILED"
    try:
        import torch

        runner.check_snapshot(snapshot)
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or "A800" not in torch.cuda.get_device_name(0)
        ):
            raise RuntimeError("Exactly one authorized A800 required")
        torch.cuda.reset_peak_memory_stats()
        execute(snapshot, budget, executed)
        validate_frozen()
        status = "COMPLETE"
    except BaseException as exc:
        data.write_json(
            runner.OUT / "failure.json",
            {
                "error_type": type(exc).__name__,
                "status": (
                    "BUDGET_STOP"
                    if isinstance(exc, runner.BudgetStop)
                    else "RECOVERY_PIPELINE_FAILED"
                ),
                "scientific_decision": "NOT_ASSESSED",
                "executed_code_sha": executed,
                "at_utc": runner.now(),
            },
        )
        raise
    finally:
        budget.finish(status)


def main() -> None:
    """Separate freeze/GPU/CPU stages; no automatic retry or repeat trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "run", "analyze"), required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    activate()
    if args.stage == "freeze":
        freeze()
        return
    runner.CACHE.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (
            runner.CACHE / "task.lock",
            runner.TOKEN_CACHE / "task.lock",
            old_cache() / "task.lock",
        ):
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "run":
            if args.snapshot is None:
                parser.error("run requires pinned snapshot")
            gpu_stage(args.snapshot)
        else:
            from .stage0_coverage_analysis import analyze

            analyze()


if __name__ == "__main__":
    main()
