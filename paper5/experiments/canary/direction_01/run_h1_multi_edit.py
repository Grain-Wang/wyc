"""One fixed-budget multi-edit diagnostic; prepare, evaluate, then CPU analyze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

from paper5.analysis import review_h1_small as history
from paper5.experiments.canary.direction_01 import h1_isolation as isolation
from paper5.experiments.canary.direction_01 import run_h1_small as h1

ROOT = history.ROOT
DESIGN = "abf29124a7acfa0c2b16575b7183468d0ef138df"
PROTOCOL = "paper5/experiments/canary/direction_01/H1_multi_edit_protocol.md"
CONFIG = ROOT / "paper5/configs/canary/direction_01/h1_multi_edit.yaml"
CONFIG_HASH = "ea1ff7cef82bb18d95e44ed5ebd875c8f35798aa87ba43b4a77d854496d19e2c"
OUTPUT = ROOT / "paper5/results/canary/direction_01/H1_multi_edit"
CACHE = ROOT / "paper5/.cache/h1_multi_edit"
ISOLATION_PROTOCOL = (
    ROOT / "paper5/experiments/canary/direction_01/H1_multi_edit_isolation_v2.md"
)
ISOLATION_RECORD = ISOLATION_PROTOCOL.with_suffix(".json")
MANIFEST_HASH = "7c40b1a1963d76e99144d7b27608fbfc6e68f1831e244f332e217a9c704021b9"
ARCHIVE_HASHES = {
    "config_used.yaml": "e164d26699c8f8e8180acec8440d6d74465f6cf7c41cff268807d7a2cdd1cdd6",
    "interaction_heatmap.svg": "8b77afaf93ed9843fa3f821bff6aa884dd8a50c1be96084d87bf5508a6545699",
    "metrics.json": "9744a0827579042b3c36026b06fc162dee3b793f49846d51b8d5b4983edd39a2",
    "per_window_nll.csv": "c0b2f17b4320af7664d9e58ba974f8dbeb3662fbf10a9b74826e6bb9185e5c86",
    "results.csv": "e0cafb3d4859005e7273aaa1abbecb098fa591c8f6a7c0e992353357411e7e17",
    "summary.md": "e42697f804b1424ece6195a149a2291bca8a5cef9cfe63bfa3f8a789efda0284",
}
Architecture = tuple[int, ...]


def digest(value: bytes) -> str:
    """SHA256 of bytes, including canonical manifests and little-endian tensors."""
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    """Serialize without whitespace, allowing no nonfinite numbers."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def configuration() -> dict[str, Any]:
    """Reject even a one-field change to the approved execution contract."""
    content = CONFIG.read_bytes()
    if digest(content) != CONFIG_HASH:
        raise ValueError("Frozen multi-edit configuration changed")
    return yaml.safe_load(content)


def file_digest(path: Path) -> str:
    """Hash large cached weights without loading them into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name: str, value: Any) -> None:
    """Write a small new-run JSON record without touching historical results."""
    (OUTPUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read(name: str) -> Any:
    """Read a previously frozen new-run artifact."""
    return json.loads((OUTPUT / name).read_text())


def git(*args: str) -> str:
    """Read local Git state only; this runner never accesses a Git remote."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def archive() -> tuple[dict[Architecture, np.ndarray], dict[str, Any]]:
    """Validate immutable historical evidence; return only calibration labels."""
    for name, expected in ARCHIVE_HASHES.items():
        if digest(history.archive_bytes(history.RESULTS + name)) != expected:
            raise ValueError("Historical Git evidence hash mismatch")
        if digest((ROOT / history.RESULTS / name).read_bytes()) != expected:
            raise ValueError("Historical worktree evidence was changed")
    metadata = json.loads(history.archive_bytes(history.RESULTS + "metrics.json"))[
        "metadata"
    ]
    windows = history.load_windows(
        history.archive_bytes(history.RESULTS + "per_window_nll.csv").decode(),
        metadata["selected_source_window_indices"],
    )
    return windows["calibration"], metadata


def candidate_plan(calibration: dict[Architecture, np.ndarray]) -> dict[str, Any]:
    """Freeze all 90 candidates and predictions using historical calibration only."""
    effects = history.effects(calibration)
    manifest = history.candidate_manifest(effects)
    if digest(canonical(manifest)) != MANIFEST_HASH:
        raise ValueError("Candidate manifest differs from approved design")
    panels = {}
    for k in (4, 6, 8):
        for stratum in ("global", "low"):
            arches = sorted(manifest[str(k)][stratum])
            s1, s2 = scores(arches, effects)
            prefix = sorted(
                range(15),
                key=lambda i: (
                    digest(
                        f"20260921:{k}:{stratum}:".encode()
                        + ",".join(map(str, arches[i])).encode()
                    ),
                    arches[i],
                ),
            )
            panels[f"{k}_{stratum}"] = {
                "architectures": arches,
                "s1": s1.tolist(),
                "s2": s2.tolist(),
                "s1_order": history.order(s1),
                "s2_order": history.order(s2),
                "direct_prefix_order": prefix,
            }
    return {"candidate_manifest": manifest, "panels": panels}


def scores(
    arches: list[Architecture], effects: dict[Architecture, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum singles plus each measured calibration pair residual exactly once."""
    s1 = np.array([sum(effects[(i,)] for i in a) for a in arches])
    s2 = s1 + np.array(
        [
            sum(
                effects[(i, j)] - effects[(i,)] - effects[(j,)]
                for i, j in itertools.combinations(a, 2)
            )
            for a in arches
        ]
    )
    return s1, s2


def holdout_indices(count: int, old: list[int]) -> list[int]:
    """Apply precisely the approved exclusion and hash order; never backfill."""
    if len(old) != 64 or len(set(old)) != 64 or not all(0 <= j < count for j in old):
        raise ValueError("Invalid V_old manifest")
    excluded = {j + d for j in old for d in (-1, 0, 1)}
    eligible = [j for j in range(count) if j not in excluded]
    if len(eligible) < 64:
        raise ValueError("Fewer than 64 eligible V_new windows; stop")
    return sorted(
        eligible, key=lambda j: (digest(f"20260922:holdout:{j}".encode()), j)
    )[:64]


def choices(panel: dict[str, Any], calibration: np.ndarray) -> dict[str, int]:
    """Expose only each policy's allowed queries; no validation argument exists."""
    if calibration.shape != (15,) or not np.isfinite(calibration).all():
        raise ValueError("Incomplete candidate calibration")

    def choose(indices: list[int]) -> int:
        return min(indices, key=lambda i: (calibration[i], i))

    selected = {}
    for score in ("s1", "s2"):
        ordered = panel[f"{score}_order"]
        selected[score] = ordered[0]
        selected[f"{score}_top5"] = choose(ordered[:5])
    for budget in (1, 3, 5, 10, 11, 15):
        selected[f"direct_{budget}"] = choose(panel["direct_prefix_order"][:budget])
    return selected


def losses(
    means: np.ndarray, selected: int, reference: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed calibration-reference G can be negative; finite-panel R cannot."""
    return means[..., selected] - means[..., reference], means[
        ..., selected
    ] - means.min(axis=-1)


def interval(values: np.ndarray, adjusted: bool = False) -> list[float]:
    """Frozen percentile intervals; adjustment is only for three primary G values."""
    return np.percentile(
        values, [0.833333, 99.166667] if adjusted else [2.5, 97.5]
    ).tolist()


def decision(panels: dict[str, Any], complete: bool = True) -> dict[str, Any]:
    """Apply conservative approved gates; missing evidence never becomes support."""
    if not complete:
        return {"verdict": "ENGINEERING_INCOMPLETE", "h2": False}
    low = [panels[f"{k}_low"] for k in (4, 6, 8)]
    qualifying = []
    for k, panel in zip((4, 6, 8), low):
        additive = panel["strategies"]["s1"]
        stable = (
            panel["practical"]
            and additive["G_adjusted_interval"][0] > 0.02
            and min(panel["confirmation_slice_G"]) > 0.02
            and min(row["s1"]["G"] for row in panel["calibration_shards"]) > 0.02
        )
        cheap_fail = all(
            panel["strategies"][name]["G_interval"][0] > 0.02
            for name in ("s1_top5", "s2", "direct_5", "direct_11")
        )
        if (
            stable
            and cheap_fail
            and panel["strategies"]["direct_15"]["R_interval"][1] <= 0.02
        ):
            qualifying.append(k)
    viable = [p for p in low if p["practical"]]

    def solved(row: dict[str, Any]) -> bool:
        return row["G_interval"][1] <= 0.02 and row["R_interval"][1] <= 0.02

    if len(qualifying) >= 2:
        verdict = "REQUEST_H2_FEASIBILITY_REVIEW_ONLY"
    elif not viable or all(solved(p["strategies"]["s1"]) for p in viable):
        verdict = "STOP_COMPLEX_METHOD_INVESTMENT"
    elif all(
        solved(p["strategies"]["s1"])
        or (
            p["strategies"]["s1"]["G_adjusted_interval"][0] > 0.02
            and any(
                solved(p["strategies"][name])
                for name in ("s1_top5", "s2", "direct_5", "direct_11")
            )
        )
        for p in viable
    ):
        verdict = "STOP_COMPLEX_METHOD_INVESTMENT"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "qualifying_k": qualifying,
        "h2": False,
        "original_h1_verdict": "INCONCLUSIVE",
    }


class WindowBudget:
    """Charge attempted forwards before execution; checks also consume budget."""

    def __init__(self) -> None:
        self.measurement = 0
        self.safety = 0

    def charge(self, count: int, safety: bool = False) -> None:
        """Reject any expansion, including excess safety checks."""
        if count <= 0:
            raise ValueError("Invalid window count")
        if safety:
            self.safety += count
        else:
            self.measurement += count
        if (
            self.measurement > 14464
            or self.safety > 16
            or self.measurement + self.safety > 14480
        ):
            raise RuntimeError("Architecture-window budget exceeded")


def prepare() -> None:
    """CPU-only manifests and pinned offline cache checks, before any forward."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer

    config = configuration()
    if OUTPUT.exists() or CACHE.exists():
        raise FileExistsError("Existing run/cache; do not restart or overwrite")
    if git("branch", "--show-current") != "paper5" or git(
        "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("Require a clean tracked paper5 worktree")
    protocol = subprocess.check_output(
        ["git", "show", f"{DESIGN}:{PROTOCOL}"], cwd=ROOT
    )
    if protocol != (ROOT / PROTOCOL).read_bytes():
        raise ValueError("Approved protocol changed")
    cal, metadata = archive()
    plan = candidate_plan(cal)
    audit = json.loads(ISOLATION_RECORD.read_text())
    if audit["version"] != isolation.VERSION:
        raise ValueError("Isolation protocol version mismatch")
    snapshot = (
        h1.H1_CACHE_ROOT
        / "huggingface/models--Qwen--Qwen2.5-1.5B/snapshots"
        / h1.EXPECTED_REVISION
    )
    names = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    )
    if not all((snapshot / name).is_file() for name in names):
        raise FileNotFoundError("Pinned model cache incomplete; no download allowed")
    cache_hashes = {name: file_digest(snapshot / name) for name in names}
    weights_hash = (snapshot / "model.safetensors").resolve().name
    if len(weights_hash) != 64 or cache_hashes["model.safetensors"] != weights_hash:
        raise ValueError("Cached weights do not match their pinned LFS object")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    for name, expected in audit["tokenizer_replay"]["H1_small_and_V2"][
        "tokenizer_files_sha256"
    ].items():
        if cache_hashes[name] != expected:
            raise ValueError("Audited tokenizer changed")
    source = metadata["selected_source_window_indices"]
    tensors, manifest = {}, {}
    for split, historical, filename in (
        ("calibration", "calibration", "train.txt"),
        ("V_new", "validation", "valid.txt"),
    ):
        path = h1._data_root() / filename
        if not path.is_file():
            raise FileNotFoundError("Required local data missing; no download allowed")
        expected = h1.EXPECTED_DATASET[historical]
        text = h1._checked_text(path, expected["url"], expected["sha256"], path)
        ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt", verbose=False
        ).input_ids[0]
        count = ids.numel() // 512
        if split == "V_new":
            coordinate = {
                "source_sha256": expected["sha256"],
                "token_count": ids.numel(),
                "full_token_stream_sha256": digest(ids.numpy().astype("<i8").tobytes()),
                "tokenization": {
                    "decode": "utf-8",
                    "add_special_tokens": False,
                    "truncation": False,
                    "padding": False,
                    "chat_template": False,
                },
            }
            if (
                coordinate != audit["coordinate"]
                or source["validation"] != audit["V_old"]
            ):
                raise ValueError("Audited source coordinate or V_old changed")
            v2 = isolation.select_windows(
                count, source["validation"], audit["historical_exposures"], coordinate
            )
            if (
                v2 != audit["selection"]
                or holdout_indices(count, source["validation"])
                != audit["original_V_new"]
            ):
                raise ValueError("Frozen isolation selection changed")
        selected = source["calibration"] if split == "calibration" else v2["selected"]
        chunks = ids[: count * 512].reshape(count, 512)
        tensors[split] = chunks[selected].contiguous()
        manifest[split] = {
            "source_window_indices": selected,
            "source_sha256": expected["sha256"],
            "complete_chunk_count": count,
            "tensor_sha256": digest(tensors[split].numpy().astype("<i8").tobytes()),
            "full_token_stream_sha256": digest(ids.numpy().astype("<i8").tobytes()),
        }
        if split == "V_new":
            if (
                manifest[split]["tensor_sha256"]
                != audit["confirmation_tensor_sha256"]["V2"]
            ):
                raise ValueError("Frozen V2 token tensor mismatch")
            sanity = torch.randperm(
                ids.numel() // 128, generator=torch.Generator().manual_seed(20260919)
            )[:2].tolist()
            if set(j // 4 for j in sanity) & set(selected):
                raise ValueError("V_new overlaps previously observed H1 sanity tokens")
            manifest[split]["sanity_128_indices"] = sanity
    OUTPUT.mkdir(parents=True)
    CACHE.mkdir(parents=True)
    torch.save(tensors, CACHE / "sequences.pt")
    config.update(
        {
            "executed_code_commit": git("rev-parse", "HEAD"),
            "input_archive_sha256": ARCHIVE_HASHES,
            "isolation_version": isolation.VERSION,
            "isolation_protocol_sha256": file_digest(ISOLATION_PROTOCOL),
            "isolation_record_sha256": file_digest(ISOLATION_RECORD),
        }
    )
    (OUTPUT / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save("predictions.json", plan)
    save(
        "manifest.json",
        {
            "windows": manifest,
            "tokenizer_revision": h1.EXPECTED_REVISION,
            "candidate_manifest_sha256": MANIFEST_HASH,
            "predictions_sha256": digest(canonical(plan)),
            "protocol_sha256": digest(protocol),
            "cached_snapshot_files_sha256": cache_hashes,
            "V_old_source_window_indices": source["validation"],
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "isolation_version": isolation.VERSION,
            "isolation_protocol_sha256": file_digest(ISOLATION_PROTOCOL),
            "isolation_record_sha256": file_digest(ISOLATION_RECORD),
            "isolation_audit": audit,
        },
    )
    save(
        "state.json",
        {"phase": "PREPARED", "executed_code_commit": config["executed_code_commit"]},
    )
    print(
        json.dumps({"phase": "PREPARED", "candidates": 90, "windows": [96, 64]}),
        flush=True,
    )


def evaluate() -> None:
    """One offline single-GPU execution, C selections frozen before V_new labels."""
    configuration()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from paper5.experiments.canary.direction_01.smoke.core import (
        locate_decoder_layers,
        max_repeated_logit_difference,
    )
    from paper5.experiments.canary.direction_01.smoke.run_smoke import (
        set_reproducibility,
    )

    state = read("state.json")
    if (
        state["phase"] != "PREPARED"
        or git("rev-parse", "HEAD") != state["executed_code_commit"]
    ):
        raise ValueError("Run already attempted or executed code changed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked worktree must remain clean")
    cal, _ = archive()
    plan, manifest = read("predictions.json"), read("manifest.json")
    if (
        manifest["isolation_version"] != isolation.VERSION
        or manifest["isolation_protocol_sha256"] != file_digest(ISOLATION_PROTOCOL)
        or manifest["isolation_record_sha256"] != file_digest(ISOLATION_RECORD)
        or manifest["isolation_audit"] != json.loads(ISOLATION_RECORD.read_text())
        or manifest["windows"]["V_new"]["source_window_indices"]
        != manifest["isolation_audit"]["selection"]["selected"]
    ):
        raise ValueError("Frozen V2 isolation record changed")
    if (
        canonical(candidate_plan(cal)) != canonical(plan)
        or digest(canonical(plan)) != manifest["predictions_sha256"]
    ):
        raise ValueError("Frozen predictions changed")
    tokens = torch.load(CACHE / "sequences.pt", map_location="cpu", weights_only=True)
    for split, tensor in tokens.items():
        if (
            digest(tensor.numpy().astype("<i8").tobytes())
            != manifest["windows"][split]["tensor_sha256"]
        ):
            raise ValueError("Frozen token tensor changed")
    state.update(
        {
            "phase": "RUNNING",
            "pid": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    save("state.json", state)
    started = time.monotonic()
    signal.alarm(7200)
    budget = WindowBudget()
    device, selected_gpu, gpu_name = h1._require_a800()
    set_reproducibility(20260918)
    snapshot = (
        h1.H1_CACHE_ROOT
        / "huggingface/models--Qwen--Qwen2.5-1.5B/snapshots"
        / h1.EXPECTED_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if len(locate_decoder_layers(model)) != 28:
        raise ValueError("Model block count mismatch")
    if model.dtype != torch.bfloat16 or model.config._attn_implementation != "eager":
        raise ValueError("Model precision or attention contract mismatch")

    def measure(
        architecture: Architecture, batch: Any, safety: bool = False
    ) -> np.ndarray:
        budget.charge(len(batch), safety)
        return h1._evaluate_architecture(model, batch, architecture, device)

    example = tokens["calibration"][:1]
    parent = float(measure((), example, True)[0])
    if abs(parent - float(cal[()][0])) > 1e-5:
        raise ValueError("Cached parent differs from the historical calibration check")
    no_edit = float(measure((), example, True)[0])
    first = tuple(plan["panels"]["4_global"]["architectures"][0])
    forward = float(measure(first, example, True)[0])
    backward = float(measure(tuple(reversed(first)), example, True)[0])
    restored = float(measure((), example, True)[0])
    budget.charge(2, True)
    repeated = max_repeated_logit_difference(model, example, device)
    if (
        max(abs(no_edit - parent), abs(restored - parent), abs(forward - backward))
        > 1e-7
        or not np.isfinite(repeated)
        or repeated > 1e-5
    ):
        raise ValueError("Intervention safety check failed")
    calibration = {}
    with (OUTPUT / "per_window_nll.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "k",
                "stratum",
                "architecture",
                "sample_index",
                "source_window_index",
                "nll",
            ],
        )
        writer.writeheader()

        def record(
            split: str, key: str, arch: Architecture, values: np.ndarray
        ) -> None:
            for i, value in enumerate(values):
                writer.writerow(
                    {
                        "split": split,
                        "k": len(arch),
                        "stratum": key.split("_")[1] if arch else "parent",
                        "architecture": ",".join(map(str, arch)),
                        "sample_index": i,
                        "source_window_index": manifest["windows"][split][
                            "source_window_indices"
                        ][i],
                        "nll": float(value),
                    }
                )
            handle.flush()

        for key, panel in plan["panels"].items():
            values = []
            for arch in panel["architectures"]:
                nll = measure(tuple(arch), tokens["calibration"])
                values.append(nll.tolist())
                record("calibration", key, tuple(arch), nll)
                print(
                    json.dumps(
                        {"phase": "calibration", "panel": key, "completed": len(values)}
                    ),
                    flush=True,
                )
            calibration[key] = np.array(values)
        selected = {
            key: choices(panel, calibration[key].mean(axis=1))
            for key, panel in plan["panels"].items()
        }
        save(
            "calibration_choices.json",
            {
                "choices": selected,
                "frozen_at_utc": datetime.now(UTC).isoformat(),
                "predictions_sha256": manifest["predictions_sha256"],
            },
        )
        # No V_new forward occurs before the persisted selection record above.
        record("V_new", "0_parent", (), measure((), tokens["V_new"]))
        for key, panel in plan["panels"].items():
            for i, arch in enumerate(panel["architectures"], 1):
                record("V_new", key, tuple(arch), measure(tuple(arch), tokens["V_new"]))
                print(
                    json.dumps({"phase": "V_new", "panel": key, "completed": i}),
                    flush=True,
                )
    final_parent = float(measure((), example, True)[0])
    if abs(final_parent - parent) > 1e-7 or budget.measurement != 14464:
        raise ValueError("Restoration or completeness check failed")
    metadata = {
        **state,
        "phase": "EVALUATED",
        "selected_gpu": selected_gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": gpu_name,
        "model_revision": h1.EXPECTED_REVISION,
        "isolation_version": isolation.VERSION,
        "model_loading": "pinned_local_snapshot_only",
        "gpu_runtime_seconds": time.monotonic() - started,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "measurement_windows": budget.measurement,
        "safety_windows": budget.safety,
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "checks": {
            "no_edit": True,
            "order_independent_k4": True,
            "restoration": True,
            "repeated_logit_difference": repeated,
            "complete": True,
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "per_window_sha256": digest((OUTPUT / "per_window_nll.csv").read_bytes()),
    }
    model = None
    torch.cuda.empty_cache()
    signal.alarm(0)
    save("runtime.json", metadata)
    save("state.json", metadata)


def prediction_metrics(
    predicted: np.ndarray, actual: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorized point/bootstrap prediction error and ranking, within one panel."""
    actual = np.atleast_2d(actual)
    error = actual - predicted

    def correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
        denom = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
        return np.divide(
            (a * b).sum(axis=-1), denom, out=np.zeros(actual.shape[0]), where=denom != 0
        )

    result = {
        "mae": np.abs(error).mean(axis=1),
        "rmse": np.sqrt((error * error).mean(axis=1)),
        "pearson": correlations(predicted, actual),
        "spearman": correlations(
            stats.rankdata(predicted), stats.rankdata(actual, axis=1)
        ),
    }
    i, j = np.triu_indices(actual.shape[1], k=1)
    p_sign = np.sign(predicted[i] - predicted[j])
    a_sign = np.sign(actual[:, i] - actual[:, j])
    denom = np.sqrt(np.count_nonzero(p_sign) * np.count_nonzero(a_sign, axis=1))
    result["kendall_tau_b"] = np.divide(
        (a_sign * p_sign).sum(axis=1),
        denom,
        out=np.zeros(actual.shape[0]),
        where=denom != 0,
    )
    p_order = np.argsort(predicted, kind="stable")
    a_order = np.argsort(actual, axis=1, kind="stable")
    for k in (1, 3, 5, 10, 15):
        result[f"top_{k}_overlap_fraction"] = (
            np.isin(a_order[:, :k], p_order[:k]).sum(axis=1) / k
        )
    return result


def load_measurements(
    plan: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Check every architecture/window identity, not just total row counts."""
    cal = {key: np.full((15, 96), np.nan) for key in plan["panels"]}
    val = {key: np.full((15, 64), np.nan) for key in plan["panels"]}
    parent = np.full(64, np.nan)
    with (OUTPUT / "per_window_nll.csv").open() as handle:
        for row in csv.DictReader(handle):
            split, sample = row["split"], int(row["sample_index"])
            if split not in ("calibration", "V_new"):
                raise ValueError("Unexpected split")
            indices = manifest["windows"][split]["source_window_indices"]
            if (
                not 0 <= sample < len(indices)
                or int(row["source_window_index"]) != indices[sample]
            ):
                raise ValueError("Unpaired source windows")
            arch = (
                list(map(int, row["architecture"].split(",")))
                if row["architecture"]
                else []
            )
            if not arch:
                if split != "V_new" or row["stratum"] != "parent" or row["k"] != "0":
                    raise ValueError("Unexpected new parent evaluation")
                array = parent
            else:
                key = row["k"] + "_" + row["stratum"]
                if len(arch) != int(row["k"]):
                    raise ValueError("Invalid edit count")
                position = plan["panels"][key]["architectures"].index(arch)
                array = (cal if split == "calibration" else val)[key][position]
            if not np.isnan(array[sample]):
                raise ValueError("Duplicate architecture/window row")
            array[sample] = float(row["nll"])
    if not all(np.isfinite(a).all() for a in [parent, *cal.values(), *val.values()]):
        raise ValueError("Incomplete/nonfinite candidate data; no survivor analysis")
    return cal, val, parent


def analyze() -> None:
    """CPU-only paired analysis; archived inputs and pre-V_new choices stay fixed."""
    signal.alarm(1800)
    started = time.monotonic()
    state = read("state.json")
    if state["phase"] != "EVALUATED":
        raise ValueError("Require a complete evaluation; no automatic rerun")
    plan, manifest = read("predictions.json"), read("manifest.json")
    historical, _ = archive()
    if (
        digest((OUTPUT / "per_window_nll.csv").read_bytes())
        != state["per_window_sha256"]
    ):
        raise ValueError("Per-window evidence changed")
    if canonical(candidate_plan(historical)) != canonical(plan):
        raise ValueError("Predictions differ from frozen calibration information")
    cal, val, parent = load_measurements(plan, manifest)
    selected = read("calibration_choices.json")["choices"]
    if selected != {
        key: choices(panel, cal[key].mean(axis=1))
        for key, panel in plan["panels"].items()
    }:
        raise ValueError("Persisted calibration selections disagree")
    keys = list(plan["panels"])
    matrix = np.concatenate([parent[None, :], *(val[key] for key in keys)])
    # One shared draw across all 91 rows, including all k values, for every replicate.
    sampled = history.paired_resamples(matrix, 20000, 20260923)
    panels, results, strategies = {}, [], []
    historical_parent = float(historical[()].mean())
    for p, key in enumerate(keys):
        panel = plan["panels"][key]
        arches = [tuple(a) for a in panel["architectures"]]
        c_means, v_means = cal[key].mean(axis=1), val[key].mean(axis=1)
        c_delta, v_delta = c_means - historical_parent, v_means - parent.mean()
        rep = sampled[:, 1 + p * 15 : 1 + (p + 1) * 15]
        rep_delta = rep - sampled[:, :1]
        reference = selected[key]["direct_15"]
        entry = {
            "practical": bool(
                np.count_nonzero(v_delta <= 0.30) >= 3 and v_delta[reference] <= 0.30
            ),
            "quality_candidate_count": int(np.count_nonzero(v_delta <= 0.30)),
            "calibration_reference": arches[reference],
            "calibration_reference_validation_delta": float(v_delta[reference]),
            "validation_hindsight_best": arches[history.order(v_means)[0]],
            "strategies": {},
            "prediction_metrics": {},
            "shortlists": {},
        }
        for name, choice in selected[key].items():
            g, r = losses(v_means, choice, reference)
            bg, br = losses(rep, choice, reference)
            result = {
                "selected_architecture": arches[choice],
                "s1_prediction": panel["s1"][choice],
                "s2_prediction": panel["s2"][choice],
                "calibration_delta": float(c_delta[choice]),
                "validation_delta": float(v_delta[choice]),
                "G": float(g),
                "R": float(r),
                "G_interval": interval(bg),
                "R_interval": interval(br),
            }
            if name == "s1":
                result["G_adjusted_interval"] = interval(bg, True)
            entry["strategies"][name] = result
            strategies.append(
                {
                    "panel": key,
                    "strategy": name,
                    "architecture": ",".join(map(str, arches[choice])),
                    "calibration_delta": float(c_delta[choice]),
                    "validation_delta": float(v_delta[choice]),
                    "G": float(g),
                    "G_lower95": interval(bg)[0],
                    "G_upper95": interval(bg)[1],
                    "R": float(r),
                    "R_lower95": interval(br)[0],
                    "R_upper95": interval(br)[1],
                }
            )
        for score in ("s1", "s2"):
            prediction = np.array(panel[score])
            point = prediction_metrics(prediction, v_delta)
            reps = prediction_metrics(prediction, rep_delta)
            entry["prediction_metrics"][score] = {
                name: {"point": float(value[0]), "interval95": interval(reps[name])}
                for name, value in point.items()
            }
            entry["prediction_metrics"][score]["calibration_point"] = {
                name: float(value[0])
                for name, value in prediction_metrics(prediction, c_delta).items()
            }
            entry["shortlists"][score] = {}
            for count in (1, 3, 5, 10, 15):
                shortlist = panel[f"{score}_order"][:count]
                hindsight = min(shortlist, key=lambda i: (v_means[i], i))
                deploy = min(shortlist, key=lambda i: (c_means[i], i))
                entry["shortlists"][score][str(count)] = {
                    "hindsight_architecture": arches[hindsight],
                    "hindsight_delta": float(v_delta[hindsight]),
                    "hindsight_regret": float(v_delta[hindsight] - v_delta.min()),
                    "calibration_rerank_architecture": arches[deploy],
                    "calibration_rerank_delta": float(v_delta[deploy]),
                    "calibration_rerank_regret": float(v_delta[deploy] - v_delta.min()),
                    "deployment_budget_status": (
                        "formal"
                        if count in (1, 5, 15)
                        else "descriptive_additional_query_cost"
                    ),
                }
        entry["confirmation_slice_G"] = [
            float(
                val[key][selected[key]["s1"], start : start + 32].mean()
                - val[key][reference, start : start + 32].mean()
            )
            for start in (0, 32)
        ]
        entry["calibration_shards"] = []
        for shard in range(3):
            features = history.effects(
                {
                    a: values[shard * 32 : (shard + 1) * 32]
                    for a, values in historical.items()
                }
            )
            shard_scores = scores(arches, features)
            record = {}
            for score, values in zip(("s1", "s2"), shard_scores):
                chosen = history.order(values)[0]
                g, r = losses(v_means, chosen, reference)
                record[score] = {
                    "selected_architecture": arches[chosen],
                    "G": float(g),
                    "R": float(r),
                }
            entry["calibration_shards"].append(record)
        panels[key] = entry
        for i, a in enumerate(arches):
            results.append(
                {
                    "panel": key,
                    "architecture": ",".join(map(str, a)),
                    "s1": panel["s1"][i],
                    "s2": panel["s2"][i],
                    "calibration_nll": float(c_means[i]),
                    "calibration_delta": float(c_delta[i]),
                    "validation_nll": float(v_means[i]),
                    "validation_delta": float(v_delta[i]),
                }
            )
    costs = [
        {
            "strategy": name,
            "cold_calibration_units_all_six_panels": units,
            "cold_calibration_windows": units * 96,
            "incremental_candidate_units_history_sunk": extra,
        }
        for name, units, extra in (
            ("s1", 13, 0),
            ("s1_top5", 43, 30),
            ("s2", 79, 0),
            ("s2_top5", 109, 30),
            *((f"direct_{b}", 13 + 6 * b, 6 * b) for b in (1, 3, 5, 10, 11, 15)),
        )
    ]
    decision_record = decision(panels)
    metrics = {
        "isolation_version": isolation.VERSION,
        "decision": decision_record,
        "panels": panels,
        "bootstrap": {
            "replicates": 20000,
            "seed": 20260923,
            "unit": "paired_V_new_window_all_91_architectures",
            "primary_low_G_percentiles": [0.833333, 99.166667],
        },
        "costs": {
            "historical_all_windows": 12640,
            "historical_calibration_used_windows": 7584,
            "new_calibration_audit_windows": 8640,
            "new_validation_windows": 5824,
            "measurement_windows": 14464,
            "safety_windows": state["safety_windows"],
            "total_input_tokens": (14464 + state["safety_windows"]) * 512,
            "pair_information_extra_windows": 6336,
            "full_direct_conditional_units": 91,
            "full_direct_inclusive_units": 103,
        },
        "cpu_analysis_seconds": time.monotonic() - started,
    }
    h1.write_csv(OUTPUT / "results.csv", results)
    h1.write_csv(OUTPUT / "strategy_results.csv", strategies)
    h1.write_csv(OUTPUT / "costs.csv", costs)
    save("metrics.json", metrics)
    lines = [
        "# H1 multi-edit fixed-budget diagnostic",
        "",
        "Isolation version: H1_multi_edit_isolation_v2. 确认集：项目内隔离后的内部确认集。",
        "",
        f"Decision: **{decision_record['verdict']}**. Original H1-small: **INCONCLUSIVE**, unchanged. H2 was not run.",
        "",
        "All 90 candidates completed C=96 and V_new=64; k=2 reused without new forwards.",
        "",
        "## Low panels (primary)",
        "",
        "| k | Practical / quality count | Additive G (adjusted interval) | Additive R | Best simple result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in (4, 6, 8):
        panel = panels[f"{k}_low"]
        a = panel["strategies"]["s1"]
        best_name = min(
            ("s1_top5", "s2", "direct_5", "direct_11"),
            key=lambda name: panel["strategies"][name]["R"],
        )
        best = panel["strategies"][best_name]
        lines.append(
            f"| {k} | {panel['practical']} / {panel['quality_candidate_count']} | {a['G']:.6f} {a['G_adjusted_interval']} | {a['R']:.6f} | {best_name}: G={best['G']:.6f}, R={best['R']:.6f} (post-hoc descriptive, not a new strategy) |"
        )
    lines.extend(
        [
            "",
            "## Scope and uncertainty",
            "",
            "All global/low panels, candidate losses, numerical errors, ranks, Top-k, calibration choices, confirmation halves and calibration-shard sensitivities are retained in metrics.json and CSVs. No damaged candidate was dropped. Only practical low panels enter the continuation decision. G uses the fixed calibration reference, can be negative, and differs from numerical error or rank disagreement. R uses the finite 15-candidate V_new hindsight minimum, not the true search-space optimum or an independent test. Bootstrap is paired across every architecture and k; minimum reselection makes R intervals descriptive. Document dependence can weaken nominal coverage. Low pools are additive-defined and not representative of all NAS candidates.",
            "",
            "s1/s2 scores were frozen before new calibration labels; policy choices before V_new. Source hashes, pinned tokenizer, token tensor hashes and excluded old windows are in manifest.json. H1 sanity short windows were checked for token overlap. V_new is internal confirmation, not a fresh corpus; the plan was authored after V_old and a historical different-model 0.5B smoke experiment on the same corpus.",
            "",
            "## Cost and decision",
            "",
            "costs.csv counts candidate construction (parent + 12 singles) for every cold-start policy. Quadratic scores cost 66 additional historical pair evaluations (6,336 windows), not free information. Additive Top-5 and direct-5 both cost 43 cold units; quadratic Top-1 and direct-11 cost 79; quadratic Top-5 costs 109 versus full direct's 103. With history sunk, both Top-5 rerankers and direct-5 cost 30 new units across six panels. Full-C audit labels are not free policy queries. The full direct reference leaves only 24 units above the 79-unit continuation gate. Hook intervention still computes original blocks, so no structural speedup is established.",
            "",
            f"Exact approved design: `{DESIGN}`; executed code: `{state['executed_code_commit']}`.",
            f"GPU {state['selected_gpu']} ({state['gpu_name']}), model phase {state['gpu_runtime_seconds']:.3f} s, peak allocated {state['peak_allocated_memory_mib']:.3f} MiB; CPU analysis {metrics['cpu_analysis_seconds']:.3f} s.",
            "",
            f"The approved gate yields {decision_record['verdict']}; qualifying k: {decision_record['qualifying_k']}. This supports no transport-method claim and authorizes no H2 execution. Stop after this single diagnostic.",
            "",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines))
    save(
        "state.json",
        {
            **state,
            "phase": "COMPLETE",
            "cpu_analysis_seconds": metrics["cpu_analysis_seconds"],
        },
    )
    signal.alarm(0)
    print(json.dumps(decision_record), flush=True)


def main() -> None:
    """Separate CPU preparation/analysis from the one authorized model execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "evaluate", "analyze"), required=True
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Launch from the independent repository root")
    try:
        if args.stage == "prepare":
            prepare()
        elif args.stage == "evaluate":
            evaluate()
        else:
            analyze()
    except BaseException as error:
        if OUTPUT.exists():
            save(
                "failure.json",
                {
                    "stage": args.stage,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "at_utc": datetime.now(UTC).isoformat(),
                    "no_automatic_retry": True,
                },
            )
        raise


if __name__ == "__main__":
    main()
