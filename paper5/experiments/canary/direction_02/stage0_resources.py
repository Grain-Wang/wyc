"""Read GPU occupancy and select one A800 with explicit memory headroom."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import stage0_data as data
from .run_stage0 import CACHE, CFG, OUT


def permission(stage: str) -> dict[str, Any]:
    """Require a separately verified allocation record; capacity is not consent."""
    path = os.environ.get("D2_GPU_PERMISSION_FILE")
    if not path:
        raise RuntimeError(
            "GPU phase paused: verified server-use permission is missing"
        )
    record = data.read_json(Path(path))
    required = {"run_id", "mode", "gpu_indices", "stages", "evidence", "expires_at_utc"}
    if not required <= record.keys():
        raise RuntimeError("Incomplete GPU permission record")
    if (
        record["run_id"] != CFG["run_id"]
        or record["mode"] not in {"exclusive", "shared"}
        or stage not in record["stages"]
        or not record["gpu_indices"]
        or not record["evidence"].strip()
    ):
        raise RuntimeError("GPU permission scope does not cover this task and stage")
    if any(type(index) is not int or index < 0 for index in record["gpu_indices"]):
        raise RuntimeError("Invalid permitted GPU indices")
    expiry = datetime.fromisoformat(record["expires_at_utc"])
    if expiry.tzinfo is None or expiry <= datetime.now(UTC):
        raise RuntimeError("GPU permission record expired or lacks timezone")
    return record


def choose_gpu(
    rows: list[dict[str, Any]],
    required_free_mib: int,
    allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend only explicitly permitted resources; never infer sharing consent."""
    if allocation is None or allocation.get("mode") not in {"exclusive", "shared"}:
        raise RuntimeError("GPU phase paused: no explicit allocation mode")
    eligible = [
        row
        for row in rows
        if "A800" in row["name"]
        and row["free_mib"] >= required_free_mib
        and row["index"] in allocation["gpu_indices"]
        and (
            allocation["mode"] == "shared"
            or (
                row["existing_compute_process_count"] == 0
                and row["used_mib"] < 1024
                and row["utilization_percent"] < 5
            )
        )
    ]
    if not eligible:
        raise RuntimeError("No A800 has required measured peak plus memory reserve")
    return min(
        eligible,
        key=lambda row: (row["utilization_percent"], row["used_mib"], row["index"]),
    )


def existing_task_processes() -> list[int]:
    """Find own D2 GPU jobs, excluding the current process and its launch ancestry."""
    excluded = {os.getpid()}
    current = os.getpid()
    while current > 1:
        try:
            status = Path(f"/proc/{current}/status").read_text()
            current = int(
                next(
                    line.split()[1]
                    for line in status.splitlines()
                    if line.startswith("PPid:")
                )
            )
            excluded.add(current)
        except (OSError, StopIteration):
            break
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            launcher = any(
                arg.endswith("direction_02/launch_stage0.sh") for arg in args
            )
            runner = "paper5.experiments.canary.direction_02.run_stage0" in args
            gpu_phase = any(arg in {"pilot", "formal"} for arg in args)
            if (launcher or runner) and gpu_phase:
                matches.append(int(entry.name))
        except OSError:
            continue
    return sorted(matches)


def admission(
    stage: str, *, selected: int | None = None, wait_idle: bool = False
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Recheck permission, duplicate jobs and live resources immediately before use."""
    allocation = permission(stage)
    if existing_task_processes():
        raise RuntimeError("An existing D2 GPU job was found; duplicate launch refused")
    required = CFG["initial_free_memory_mib"]
    if stage == "formal":
        pilot = data.read_json(OUT / "pilot.json")
        required = max(
            required, int(pilot["peak_reserved_mib"]) + CFG["memory_reserve_mib"]
        )
    allowed = dict(allocation)
    if selected is not None:
        if selected not in allocation["gpu_indices"]:
            raise RuntimeError("Selected device outside allocation")
        allowed["gpu_indices"] = [selected]
    first = choose_gpu(inspect_gpus(), required, allowed)
    allowed["gpu_indices"] = [first["index"]]
    # Exclusive use requires four clean samples spanning 90 seconds; no busy waiting.
    samples = 3 if wait_idle and allocation["mode"] == "exclusive" else 1
    for _ in range(samples):
        if wait_idle and allocation["mode"] == "exclusive":
            time.sleep(30)
        permission(stage)
        if existing_task_processes():
            raise RuntimeError("D2 job appeared during resource checks")
        first = choose_gpu(inspect_gpus(), required, allowed)
    return first, allocation, required


def inspect_gpus() -> list[dict[str, Any]]:
    """Read utilization and aggregate compute occupancy without personal details."""
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return [
        {
            "index": int(row[0].strip()),
            "uuid": row[1].strip(),
            "name": row[2].strip(),
            "used_mib": int(row[3].strip()),
            "free_mib": int(row[4].strip()),
            "utilization_percent": int(row[5].strip()),
            "existing_compute_process_count": sum(
                p.strip() == row[1].strip() for p in processes
            ),
        }
        for row in csv.reader(io.StringIO(raw))
    ]


def main() -> None:
    """Print one GPU index for the launcher and retain sanitized resource metadata."""
    stage = os.environ["D2_STAGE"]
    second, allocation, required = admission(stage, wait_idle=True)
    data.write_json(
        OUT / f"{stage}_resource_preflight.json",
        {
            **second,
            "minimum_free_mib": required,
            "reserve_mib": CFG["memory_reserve_mib"],
            "other_processes_terminated": False,
            "allocation_mode": allocation["mode"],
            "permission_record_sha256": data.digest(data.canonical(allocation)),
        },
    )
    ledger = (
        data.read_json(CACHE / "gpu_ledger.json")
        if (CACHE / "gpu_ledger.json").exists()
        else {"spent_seconds": 0}
    )
    remaining = min(
        CFG["gpu_seconds_limit"] - ledger["spent_seconds"],
        CFG["pilot_seconds_limit"] if stage == "pilot" else CFG["gpu_seconds_limit"],
    )
    if remaining < 60:
        raise RuntimeError("GPU time budget exhausted")
    print(f"{second['index']} {int(remaining) - 5}")


if __name__ == "__main__":
    main()
