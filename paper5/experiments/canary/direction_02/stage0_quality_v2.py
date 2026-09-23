"""V2 admission and isolated paths; reuse the original Stage0 scientific pipeline."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from . import stage0_data as data

CONFIG = data.ROOT / "paper5/configs/canary/direction_02/recovery_stage0_v2.json"
PROTOCOL = Path(__file__).with_name("STAGE0_QUALITY_V2.md")
RESULTS = data.ROOT / "paper5/results/canary/direction_02"
SOURCE = RESULTS / "recovery_stage0/20260923_stage0_01"
CAPACITY = RESULTS / "recovery_capacity/20260923_a00_seed17_resume10"
BASE_ARCHIVE = "d44f48f49f80bb91d489eb80e2ab6ea0d08bc8b9"
GPU_UUID = "GPU-5c9c1a6d-52ff-000a-11e9-be3f1b36d43d"


def validate_recipe(cfg: dict[str, Any]) -> None:
    """Only run identity and the explicitly authorized admission metadata may differ."""
    old = data.config()
    excluded = {"run_id", "design_version", "quality_admission"}
    if {k: v for k, v in cfg.items() if k not in excluded} != {
        k: v for k, v in old.items() if k not in excluded
    }:
        raise RuntimeError("V2 cannot change the scientific recipe")
    if cfg["quality_admission"]["base_archive_sha"] != BASE_ARCHIVE:
        raise RuntimeError("Wrong capacity evidence archive")
    if (
        cfg["design_version"] != "D2_STAGE0_QUALITY_V2"
        or cfg["run_id"] != "20260923_stage0_quality_v2_01"
    ):
        raise RuntimeError("Unexpected V2 identity")


def activate(runner: ModuleType) -> None:
    """Select explicit V2 output paths without modifying the original recipe files."""
    cfg = data.read_json(CONFIG)
    validate_recipe(cfg)
    base = Path(
        os.environ.get("D2_CACHE", str(data.ROOT / "paper5/.cache/direction_02"))
    )
    runner.CFG = cfg
    runner.OUT = RESULTS / "recovery_stage0_v2" / cfg["run_id"]
    runner.CACHE = base / "recovery_stage0_v2" / cfg["run_id"]
    runner.TOKEN_CACHE = base / cfg["quality_admission"]["source_run_id"]
    runner.EXEC_CONFIG, runner.EXEC_PROTOCOL = CONFIG, PROTOCOL
    runner.QUALITY_V2 = True


def admission_evidence(pilot: dict[str, Any], capacity: dict[str, Any]) -> None:
    """Initial/pilot quality cannot veto V2; engineering and capacity still must pass."""
    if not pilot["engineering_passed"] or not pilot["trajectory"]["base_unchanged"]:
        raise RuntimeError("Engineering pilot failed")
    if (
        capacity["status"] != "EXPLORATORY_CAPACITY_FEASIBLE"
        or not capacity["base_unchanged"]
        or capacity["candidate"] != "a00"
        or capacity["seed"] != 17
        or capacity["endpoint"] != 100
        or capacity["e_accessed"]
        or capacity["formal_started"]
        or capacity["metrics"][-1]["ppl_ratio"] > 1.15
    ):
        raise RuntimeError("Missing authorized single-candidate capacity evidence")


def budget_projection(
    pilot: dict[str, Any],
    capacity: dict[str, Any],
    ledger: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Retain spent GPU time and conservative observed loading/evaluation overhead."""
    units = dict(pilot["unit_costs"])
    units["update_seconds"] = max(
        units["update_seconds"], ledger["phases"][-1]["elapsed_seconds"] / 90
    )
    units["eval_window_seconds"] = max(
        units["eval_window_seconds"],
        *(row["eval_window_seconds"] for row in capacity["metrics"]),
    )
    units["checkpoint_seconds"] = max(
        units["checkpoint_seconds"],
        *(row["checkpoint_seconds"] for row in capacity["metrics"]),
    )
    remaining = (
        1600 * units["update_seconds"]
        + 5184 * units["eval_window_seconds"]
        + 33 * units["construction_seconds"]
        + 64 * units["checkpoint_seconds"]
    )
    projected = (
        ledger["spent_seconds"]
        + cfg["budget_safety_factor"] * remaining
        + cfg["budget_fixed_reserve_seconds"]
    )
    if projected > cfg["gpu_seconds_limit"]:
        raise RuntimeError(
            "Complete sixteen-trajectory task exceeds remaining total budget"
        )
    return {
        "prior_gpu_seconds": ledger["spent_seconds"],
        "projected_total_gpu_seconds": projected,
        "hard_limit_seconds": cfg["gpu_seconds_limit"],
        "unit_cost_bounds": units,
        "counts": {
            "updates": 1600,
            "s_windows": 4096,
            "e_windows": 1088,
            "constructions": 33,
            "checkpoint_writes": 64,
        },
    }


def freeze_v2(runner: ModuleType) -> None:
    """Create a committed admission record bound to the already archived evidence."""
    pilot, capacity = data.read_json(SOURCE / "pilot.json"), data.read_json(
        CAPACITY / "summary.json"
    )
    ledger = data.read_json(CAPACITY / "gpu_budget.json")
    admission_evidence(pilot, capacity)
    if [p["stage"] for p in ledger["phases"]] != ["pilot", "capacity"] or any(
        p["status"] != "COMPLETE" for p in ledger["phases"]
    ):
        raise RuntimeError("Incomplete cumulative GPU evidence")
    if not math.isclose(
        sum(p["elapsed_seconds"] for p in ledger["phases"]),
        ledger["spent_seconds"],
        abs_tol=1e-9,
    ):
        raise RuntimeError("Cumulative GPU ledger inconsistent")
    preserved = {}
    for folder in [RESULTS.parent / "direction_01", SOURCE, CAPACITY]:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix != ".log":
                relative = str(path.relative_to(data.ROOT))
                expected = subprocess.check_output(
                    ["git", "show", f"{BASE_ARCHIVE}:{relative}"], cwd=data.ROOT
                )
                if expected != path.read_bytes():
                    raise RuntimeError("Original archived evidence changed")
                preserved[relative] = data.file_hash(path)
    projection = budget_projection(pilot, capacity, ledger, runner.CFG)
    runner.OUT.mkdir(parents=True, exist_ok=False)
    for name in [
        "data_manifest.json",
        "candidate_manifest.json",
        "preparation.json",
        "pilot.json",
    ]:
        shutil.copyfile(SOURCE / name, runner.OUT / name)
    data.write_json(runner.OUT / "config_used.json", runner.CFG)
    shutil.copyfile(CAPACITY / "gpu_budget.json", runner.OUT / "prior_gpu_budget.json")
    data.write_json(runner.OUT / "budget_projection.json", projection)
    data.write_json(
        runner.OUT / "formal_freeze.json",
        {
            "design_version": runner.CFG["design_version"],
            "base_archive_sha": BASE_ARCHIVE,
            "frozen_at_utc": runner.now(),
            "config_sha256": data.file_hash(CONFIG),
            "protocol_sha256": data.file_hash(PROTOCOL),
            "pilot_sha256": data.file_hash(runner.OUT / "pilot.json"),
            "data_manifest_sha256": data.file_hash(runner.OUT / "data_manifest.json"),
            "candidate_manifest_sha256": data.file_hash(
                runner.OUT / "candidate_manifest.json"
            ),
            "prior_gpu_budget_sha256": data.file_hash(
                runner.OUT / "prior_gpu_budget.json"
            ),
            "quality_max_ppl_ratio": 1.15,
            "e_accessed": False,
            "quality_admission": "engineering and capacity passed; quality judged at 100-step E",
            "observed_capacity_s_before_revision": True,
            "preserved_source_results_sha256": preserved,
        },
    )
    runner.validate_artifacts()


def check_live_resource() -> None:
    """Stay on physical GPU2 and retain reserve without affecting other processes."""
    from . import stage0_resources as resources

    gpu = next(row for row in resources.inspect_gpus() if row["index"] == 2)
    if gpu["uuid"] != GPU_UUID or gpu["free_mib"] < 4096:
        raise RuntimeError("Physical GPU2 changed or memory reserve is unavailable")


def prepare_gpu(runner: ModuleType, resources: ModuleType) -> None:
    """Use existing permission checks and inherit the unmodified completed ledger."""
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "2"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise RuntimeError("Only physical GPU2 with PCI bus ordering is authorized")
    # The CLI can run as __main__; bind resources to that same configured runner.
    resources.CFG, resources.OUT, resources.CACHE = runner.CFG, runner.OUT, runner.CACHE
    frozen = data.read_json(runner.OUT / "formal_freeze.json")
    for name, digest in frozen["preserved_source_results_sha256"].items():
        if data.file_hash(data.ROOT / name) != digest:
            raise RuntimeError("Original evidence changed before V2 execution")
    prior = runner.OUT / "prior_gpu_budget.json"
    actual = (
        runner.TOKEN_CACHE.parent
        / "recovery_capacity/20260923_a00_seed17_resume10/gpu_ledger.json"
    )
    if (
        data.file_hash(actual) != frozen["prior_gpu_budget_sha256"]
        or prior.read_bytes() != actual.read_bytes()
    ):
        raise RuntimeError("Cumulative GPU time changed; refusing to reset budget")
    if (runner.CACHE / "gpu_ledger.json").exists():
        raise RuntimeError("V2 GPU attempt already exists; duplicate launch refused")
    samples = []
    for index in range(3):
        gpu, allocation, required = resources.admission("formal", selected=2)
        if gpu["uuid"] != GPU_UUID or allocation["gpu_indices"] != [2]:
            raise RuntimeError("GPU permission does not match fixed physical GPU2")
        samples.append(gpu)
        if index < 2:
            time.sleep(2)
    data.write_json(
        runner.OUT / "formal_resource_preflight.json",
        {
            "samples": samples,
            "physical_gpu": 2,
            "gpu_uuid": GPU_UUID,
            "required_free_mib": required,
            "allocation_mode": allocation["mode"],
            "other_processes_terminated": False,
        },
    )
    shutil.copyfile(prior, runner.CACHE / "gpu_ledger.json")


def supplement_analysis(runner: ModuleType) -> None:
    """Expose all intermediate quality descriptively and separate measured costs."""
    scores = data.read_json(runner.OUT / "s_scores.json")
    metrics = data.read_json(runner.OUT / "metrics.json")
    trajectories = data.read_json(runner.OUT / "trajectories.json")
    pilot = data.read_json(SOURCE / "pilot.json")
    parent = pilot["parent_s_nll"]
    data.write_json(
        runner.OUT / "quality_by_checkpoint.json",
        {
            "s_parent_reference": "unmodified parent from original pilot, not retrained",
            "parent_s_nll": parent,
            "quality_max_ppl_ratio": 1.15,
            "s_quality": {
                seed: {
                    step: {
                        name: {
                            "nll": loss,
                            "ppl_ratio": math.exp(loss - parent),
                            "quality_passed": math.exp(loss - parent) <= 1.15,
                        }
                        for name, loss in values.items()
                    }
                    for step, values in points.items()
                }
                for seed, points in scores.items()
            },
            "e_final_quality": {
                seed: {
                    "quality_usable": row["quality_usable"],
                    "candidate_count": row["quality_candidate_count"],
                    "candidates": {
                        name: {
                            "nll": loss,
                            "ppl_ratio": math.exp(loss - metrics["parent_e_nll"]),
                            "quality_passed": math.exp(loss - metrics["parent_e_nll"])
                            <= 1.15,
                        }
                        for name, loss in row["candidate_endpoint_e_nll"].items()
                    },
                }
                for seed, row in metrics["seeds"].items()
            },
        },
    )
    budget = data.read_json(runner.OUT / "gpu_budget.json")
    data.write_json(
        runner.OUT / "audit_costs.json",
        {
            "prior_pilot_and_capacity_gpu_seconds": budget["spent_seconds"]
            - budget["phases"][-1]["elapsed_seconds"],
            "formal_gpu_seconds": budget["phases"][-1]["elapsed_seconds"],
            "cumulative_gpu_seconds": budget["spent_seconds"],
            "gpu_limit_seconds": 14400,
            "training_update_seconds": sum(
                sum(t["update_seconds"]) for t in trajectories
            ),
            "s_evaluation_seconds": sum(
                sum(t["eval_window_seconds"]) * 64 for t in trajectories
            ),
            "training_checkpoint_seconds": sum(
                sum(t["checkpoint_seconds"]) for t in trajectories
            ),
            "training_construction_seconds": sum(
                t["mapping"]["construction_seconds"] for t in trajectories
            ),
            "policy_estimate_basis": "unchanged original pilot unit costs; see copied pilot.json and metrics.json",
            "capacity_reused_as_formal_sample": False,
            "a00_same_seed_rerun_independent_replicate": False,
            "parent_recovery_training_added": False,
        },
    )
