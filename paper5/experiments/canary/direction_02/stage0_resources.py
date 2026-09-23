"""Read GPU occupancy and select one A800 with explicit memory headroom."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from typing import Any

from . import stage0_data as data
from .run_stage0 import CACHE, CFG, OUT


def choose_gpu(rows: list[dict[str, Any]], required_free_mib: int) -> dict[str, Any]:
    """Select the least occupied eligible A800; other processes are untouched."""
    eligible = [
        row
        for row in rows
        if "A800" in row["name"] and row["free_mib"] >= required_free_mib
    ]
    if not eligible:
        raise RuntimeError("No A800 has required measured peak plus memory reserve")
    return min(
        eligible,
        key=lambda row: (row["used_mib"], row["utilization_percent"], row["index"]),
    )


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
    required = CFG["initial_free_memory_mib"]
    if stage == "formal":
        pilot = data.read_json(OUT / "pilot.json")
        required = max(
            required, int(pilot["peak_reserved_mib"]) + CFG["memory_reserve_mib"]
        )
    first = choose_gpu(inspect_gpus(), required)
    second = next(row for row in inspect_gpus() if row["index"] == first["index"])
    if second["free_mib"] < required:
        raise RuntimeError("GPU memory changed before admission")
    data.write_json(
        OUT / f"{stage}_resource_preflight.json",
        {
            **second,
            "minimum_free_mib": required,
            "reserve_mib": CFG["memory_reserve_mib"],
            "other_processes_terminated": False,
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
