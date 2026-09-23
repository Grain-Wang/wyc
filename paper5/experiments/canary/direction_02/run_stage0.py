"""One bounded D2 Stage0: prepare, pilot, freeze, formal trajectories, analyze."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import gc
import importlib.metadata
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import stage0_analysis as analysis
from . import stage0_data as data

CFG = data.config()
OUT = data.ROOT / "paper5/results/canary/direction_02/recovery_stage0" / CFG["run_id"]
CACHE = (
    Path(os.environ.get("D2_CACHE", str(data.ROOT / "paper5/.cache/direction_02")))
    / CFG["run_id"]
)
PROTOCOL = data.ROOT / "paper5/experiments/canary/direction_02/STAGE0_PROTOCOL.md"


def now() -> str:
    """UTC timestamp for small, portable provenance records."""
    return datetime.now(UTC).isoformat()


def code_sha() -> str:
    """Require committed, clean code without touching any Git state."""
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=data.ROOT
    ):
        raise RuntimeError("Tracked worktree is dirty; commit reviewed code first")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=data.ROOT, text=True
    ).strip()


def record_csv(path: Path, row: dict[str, Any]) -> None:
    """Append an auditable numerical row and flush it before continuing."""
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def prepare(source: Path, tokenizer_path: Path) -> None:
    """Prepare token files locally; no model weights, GPU or losses are loaded."""
    from transformers import AutoTokenizer

    raw = source.read_bytes()
    if data.digest(raw) != CFG["source_sha256"]:
        raise ValueError("Source file differs from the authorized cached train text")
    expected = data.read_json(
        data.ROOT / "paper5/results/canary/direction_01/H1_multi_edit/manifest.json"
    )["cached_snapshot_files_sha256"]
    hashes = {
        name: data.file_hash(tokenizer_path / name)
        for name in expected
        if name != "model.safetensors"
    }
    if any(hashes[name] != expected[name] for name in hashes):
        raise ValueError("Pinned tokenizer/config file mismatch")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    arrays, manifest = data.make_dataset(
        raw, lambda text: tok(text, add_special_tokens=False)["input_ids"], CFG
    )
    candidate_manifest = data.candidates(data.D1 / "per_window_nll.csv", CFG)
    CACHE.mkdir(parents=True, exist_ok=True)
    for split, array in arrays.items():
        filename = f"{split}.npy" if split != "E" else "E.sealed.npy"
        with (CACHE / filename).open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        manifest["splits"][split]["npy_sha256"] = data.file_hash(CACHE / filename)
    manifest.update(
        tokenizer_revision=CFG["model_revision"],
        tokenizer_files_sha256=hashes,
        config_sha256=data.file_hash(data.CONFIG),
        protocol_sha256=data.file_hash(PROTOCOL),
        source_name="PyTorch examples WikiText-2 train",
        source_revision="content-addressed source SHA256",
        source_url="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        dataset_choice="authorized existing-train fallback; no ready small C4/FineWeb subset",
        unknown_marker_count=raw.count(b"<unk>"),
    )
    data.write_json(OUT / "config_used.json", CFG)
    data.write_json(OUT / "data_manifest.json", manifest)
    data.write_json(OUT / "candidate_manifest.json", candidate_manifest)
    data.write_json(
        OUT / "preparation.json",
        {
            "prepared_at_utc": now(),
            "model_forwards": 0,
            "status": "PREPARED_NOT_PILOTED",
            "data_manifest_sha256": data.file_hash(OUT / "data_manifest.json"),
            "candidate_manifest_sha256": data.file_hash(
                OUT / "candidate_manifest.json"
            ),
        },
    )


def validate_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject changed prepared data, candidates, recipe or protocol."""
    manifest = data.read_json(OUT / "data_manifest.json")
    candidate_manifest = data.read_json(OUT / "candidate_manifest.json")
    preparation = data.read_json(OUT / "preparation.json")
    if data.read_json(OUT / "config_used.json") != CFG:
        raise ValueError("Prepared recipe changed")
    for key, path in [("config_sha256", data.CONFIG), ("protocol_sha256", PROTOCOL)]:
        if manifest[key] != data.file_hash(path):
            raise ValueError("Prepared protocol/config hash changed")
    for name in ("data_manifest", "candidate_manifest"):
        if preparation[f"{name}_sha256"] != data.file_hash(OUT / f"{name}.json"):
            raise ValueError("Prepared manifest changed")
    return manifest, candidate_manifest


def load_split(
    split: str, manifest: dict[str, Any], *, e_unsealed: bool = False
) -> np.ndarray:
    """Never open E before complete trajectories and persisted strategy choices."""
    if split == "E":
        if not e_unsealed:
            raise RuntimeError("E is sealed")
        validate_selection()
    filename = f"{split}.npy" if split != "E" else "E.sealed.npy"
    path = CACHE / filename
    spec = manifest["splits"][split]
    if data.file_hash(path) != spec["npy_sha256"]:
        raise ValueError("Token file hash mismatch")
    array = np.load(path, allow_pickle=False)
    if (
        array.shape != (CFG["windows"][split], 512)
        or data.digest(array.tobytes()) != spec["token_data_sha256"]
    ):
        raise ValueError("Token coordinates/hash mismatch")
    return array


def validate_selection() -> dict[str, Any]:
    """Require the complete endpoint ledger, choices and their exact hashes."""
    record = data.read_json(OUT / "selection.json")
    complete = data.read_json(OUT / "trajectories.json")
    expected = {(seed, f"a{i:02d}") for seed in CFG["training_seeds"] for i in range(8)}
    if {(row["seed"], row["candidate"]) for row in complete} != expected or len(
        complete
    ) != 16:
        raise RuntimeError("Incomplete trajectory set; cannot unseal E")
    for row in complete:
        if row["updates"] != 100 or not row["base_unchanged"]:
            raise RuntimeError("Invalid completed trajectory")
        if (
            data.file_hash(
                CACHE / "adapters" / str(row["seed"]) / row["candidate"] / "step_100.pt"
            )
            != row["endpoint_sha256"]
        ):
            raise RuntimeError("Missing or changed endpoint adapter")
    if record["trajectory_sha256"] != data.file_hash(OUT / "trajectories.json"):
        raise RuntimeError("Choices no longer bound to trajectories")
    if record["s_scores_sha256"] != data.file_hash(OUT / "s_scores.json"):
        raise RuntimeError("S scores changed after strategy freeze")
    s = data.read_json(OUT / "s_scores.json")
    expected_choices = {
        seed: analysis.replay({int(step): values for step, values in scores.items()})
        for seed, scores in s.items()
    }
    if record["choices"] != expected_choices:
        raise RuntimeError("Strategy replay differs from persisted choices")
    return record


class BudgetStop(RuntimeError):
    """Stop early enough to retain a partial checkpoint before the hard deadline."""


class GpuBudget:
    """Persistent cumulative wall time over GPU phases; no silent repeat runs."""

    def __init__(self, stage: str) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / "gpu_ledger.json"
        self.record = (
            data.read_json(self.path)
            if self.path.exists()
            else {"spent_seconds": 0.0, "phases": []}
        )
        if any(row["stage"] == stage for row in self.record["phases"]):
            raise RuntimeError(
                "This GPU phase was already attempted; no duplicate launch"
            )
        # Count Python import / snapshot verification overhead measured by the launcher.
        launch_age = max(
            0.0,
            time.time()
            - float(os.environ.get("D2_GPU_PHASE_STARTED_UNIX", time.time())),
        )
        self.start = time.monotonic() - launch_age
        self.stage = stage
        self.limit = min(
            CFG["gpu_seconds_limit"] - self.record["spent_seconds"],
            (
                CFG["pilot_seconds_limit"]
                if stage == "pilot"
                else CFG["gpu_seconds_limit"]
            ),
        )
        if self.limit <= 30:
            raise BudgetStop("Insufficient remaining GPU budget")
        self.record["phases"].append(
            {"stage": stage, "started_at_utc": now(), "status": "RUNNING"}
        )
        data.write_json(self.path, self.record, replace=True)
        signal.signal(signal.SIGALRM, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        if self.limit - launch_age <= 30:
            raise BudgetStop("Startup consumed remaining GPU budget")
        signal.setitimer(signal.ITIMER_REAL, self.limit - launch_age - 20)

    def _stop(self, _signum: int, _frame: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise BudgetStop("Cumulative GPU deadline or termination signal")

    def check(self) -> None:
        """Check before each forward; leave checkpoint-save margin."""
        if time.monotonic() - self.start >= self.limit - 30:
            raise BudgetStop("Insufficient time for another update/evaluation")
        data.write_json(
            CACHE / "heartbeat.json",
            {
                "pid": os.getpid(),
                "stage": self.stage,
                "utc": now(),
                "current_phase_elapsed_seconds": time.monotonic() - self.start,
                "previous_gpu_seconds": self.record["spent_seconds"],
            },
            replace=True,
        )

    def finish(self, status: str) -> None:
        """Persist all spent time, including failed phases and loading/I/O."""
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - self.start
        self.record["spent_seconds"] += elapsed
        self.record["phases"][-1].update(
            status=status, elapsed_seconds=elapsed, ended_at_utc=now()
        )
        data.write_json(self.path, self.record, replace=True)
        data.write_json(OUT / "gpu_budget.json", self.record, replace=True)


def check_snapshot(snapshot: Path) -> None:
    """Read-only verification of pinned shared D1 weights and tokenizer."""
    expected = data.read_json(
        data.ROOT / "paper5/results/canary/direction_01/H1_multi_edit/manifest.json"
    )["cached_snapshot_files_sha256"]
    if any(
        data.file_hash(snapshot / name) != value for name, value in expected.items()
    ):
        raise RuntimeError("Pinned shared model snapshot changed")


def load_model(
    snapshot: Path, removed: list[int] | None, seed: int | None
) -> tuple[Any, dict[str, Any]]:
    """Load pristine weights offline, then remove blocks and create new LoRA."""
    import torch
    from transformers import AutoModelForCausalLM
    from . import stage0_model as implementation

    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    if model.config.num_hidden_layers != 28 or model.config.model_type != "qwen2":
        raise RuntimeError("Unexpected parent architecture")
    model.config.use_cache = False
    model.requires_grad_(False)
    mapping = (
        implementation.prune(model, removed)
        if removed is not None
        else {"parent": True}
    )
    if seed is not None:
        mapping["adapter_parameters"] = implementation.attach_lora(model, CFG, seed)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    torch.cuda.synchronize()
    mapping["construction_seconds"] = time.monotonic() - start
    return model, mapping


def release(model: Any) -> None:
    """Move a completed model off GPU without touching shared cached weights."""
    import torch

    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(
    model: Any,
    array: np.ndarray,
    windows: list[dict[str, Any]],
    split: str,
    seed: int | str,
    candidate: str,
    step: int,
    budget: GpuBudget,
    filename: str = "per_window_nll.csv",
) -> tuple[list[float], float]:
    """Evaluate full windows in eval mode and retain document/label identities."""
    import torch
    from .stage0_model import nll

    model.eval()
    start = time.monotonic()
    losses = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tokens, window in zip(array, windows, strict=True):
            budget.check()
            value = float(
                nll(
                    model, torch.from_numpy(tokens.copy()).unsqueeze(0).to("cuda")
                ).item()
            )
            if not np.isfinite(value):
                raise RuntimeError("Nonfinite evaluation loss")
            losses.append(value)
            record_csv(
                OUT / filename,
                {
                    "split": split,
                    "seed": seed,
                    "candidate": candidate,
                    "checkpoint": step,
                    "window_id": window["window_id"],
                    "doc_id": window["doc_id"],
                    "document_token_start": window["document_token_start"],
                    "labels": window["labels"],
                    "nll": value,
                },
            )
    return losses, (time.monotonic() - start) / len(losses)


def checkpoint(
    model: Any,
    optimizer: Any,
    path: Path,
    step: int,
    seed: int,
    candidate: str,
    order: list[int],
    mapping: dict[str, Any],
    executed: str,
) -> float:
    """Save adapters, optimizer and RNG only; refuse to overwrite a checkpoint."""
    import torch
    from .stage0_model import adapter_state

    start = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(
            {
                "adapter": adapter_state(model),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "seed": seed,
                "candidate": candidate,
                "training_order": order,
                "mapping": mapping,
                "executed_code_sha": executed,
                "config_sha256": data.file_hash(data.CONFIG),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            handle,
        )
    return time.monotonic() - start


def train_trajectory(
    model: Any,
    candidate: str,
    seed: int,
    updates: int,
    train: np.ndarray,
    selection: np.ndarray,
    windows: list[dict[str, Any]],
    mapping: dict[str, Any],
    budget: GpuBudget,
    executed: str,
    *,
    pilot: bool = False,
) -> dict[str, Any]:
    """One optimizer trajectory; intermediate checkpoints never restart training."""
    import torch
    from . import stage0_model as implementation

    order = data.training_order(len(train), seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or len(order) < updates * 4:
        raise RuntimeError("Missing adapters or insufficient unique T windows")
    optimizer = torch.optim.AdamW(params, lr=CFG["learning_rate"], weight_decay=0.0)
    original_hash = implementation.base_hash(model)
    initial = implementation.adapter_state(model)
    folder = CACHE / ("pilot_adapter" if pilot else "adapters") / str(seed) / candidate
    times = []
    checkpoint_times = []
    eval_times = []
    scores = {}
    completed = 0
    before_loss = None
    step = 0
    try:
        for step in range(updates + 1):
            if (
                step == 0
                or (not pilot and step in CFG["checkpoints"])
                or step == updates
            ):
                checkpoint_times.append(
                    checkpoint(
                        model,
                        optimizer,
                        folder / f"step_{step}.pt",
                        step,
                        seed,
                        candidate,
                        order,
                        mapping,
                        executed,
                    )
                )
                values, duration = evaluate(
                    model,
                    selection,
                    windows,
                    "S",
                    seed,
                    candidate,
                    step,
                    budget,
                    "pilot_nll.csv" if pilot else "per_window_nll.csv",
                )
                scores[str(step)] = float(np.mean(values))
                eval_times.append(duration)
            if step == updates:
                break
            budget.check()
            start = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(4):
                budget.check()
                tokens = (
                    torch.from_numpy(train[order[step * 4 + micro]].copy())
                    .unsqueeze(0)
                    .to("cuda")
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = implementation.nll(model, tokens)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                losses.append(float(loss.detach().item()))
                (loss / 4).backward()
            norm = torch.nn.utils.clip_grad_norm_(
                params, CFG["gradient_clip"], error_if_nonfinite=True
            )
            if any(
                p.grad is not None
                for name, p in model.named_parameters()
                if not name.endswith(("lora_a", "lora_b"))
            ):
                raise RuntimeError("Base received a gradient")
            if step == 0:
                before_loss = float(np.mean(losses))
                if float(norm.item()) <= 0:
                    raise RuntimeError("No adapter gradient at first update")
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            completed = step + 1
            record_csv(
                OUT / ("pilot_training_curve.csv" if pilot else "training_curve.csv"),
                {
                    "seed": seed,
                    "candidate": candidate,
                    "update": step + 1,
                    "train_nll": float(np.mean(losses)),
                    "gradient_norm": float(norm.item()),
                    "input_tokens_cumulative": (step + 1) * 4 * 512,
                    "effective_labels_cumulative": (step + 1) * 4 * 511,
                    "update_seconds": times[-1],
                },
            )
    except BaseException:
        checkpoint(
            model,
            optimizer,
            folder / "interrupted.pt",
            completed,
            seed,
            candidate,
            order,
            mapping,
            executed,
        )
        raise
    current = implementation.adapter_state(model)
    delta = sum(
        float((current[name] - initial[name]).square().sum()) for name in current
    )
    if implementation.base_hash(model) != original_hash or delta <= 0:
        raise RuntimeError("Base mutated or adapter did not update")
    return {
        "candidate": candidate,
        "seed": seed,
        "updates": updates,
        "s_scores": scores,
        "base_unchanged": True,
        "base_sha256": original_hash,
        "adapter_squared_change": delta,
        "first_train_loss": before_loss,
        "update_seconds": times,
        "checkpoint_seconds": checkpoint_times,
        "eval_window_seconds": eval_times,
        "input_tokens": updates * 4 * 512,
        "effective_labels": updates * 4 * 511,
        "unique_windows": len(set(order[: updates * 4])),
        "training_order_sha256": data.digest(data.canonical(order)),
        "mapping": mapping,
        "endpoint_sha256": data.file_hash(folder / f"step_{updates}.pt"),
    }


def pilot(snapshot: Path, budget: GpuBudget, executed: str) -> None:
    """Run one ten-update feasibility trajectory without opening E."""
    import torch
    from .stage0_model import load_adapter, nll, verify_forward_order

    manifest, pool = validate_artifacts()
    train = load_split("T", manifest)
    selection = load_split("S", manifest)
    windows = manifest["splits"]["S"]["windows"]
    parent, _ = load_model(snapshot, None, None)
    values, parent_time = evaluate(
        parent, selection, windows, "S", "parent", "parent", 0, budget, "pilot_nll.csv"
    )
    parent_nll = float(np.mean(values))
    release(parent)
    del parent
    initial_scores = {}
    mappings = {}
    for candidate in pool["candidates"]:
        model, mapping = load_model(snapshot, candidate["removed"], CFG["pilot_seed"])
        verify_forward_order(
            model, torch.from_numpy(train[0].copy()).unsqueeze(0).to("cuda")
        )
        values, _ = evaluate(
            model,
            selection,
            windows,
            "S",
            "initial",
            candidate["id"],
            0,
            budget,
            "pilot_nll.csv",
        )
        initial_scores[candidate["id"]] = float(np.mean(values))
        mappings[candidate["id"]] = mapping
        release(model)
        del model
    if len({m["adapter_parameters"] for m in mappings.values()}) != 1:
        raise RuntimeError("Unequal adapter capacity")
    chosen = pool["candidates"][0]
    model, mapping = load_model(snapshot, chosen["removed"], CFG["pilot_seed"])
    trajectory = train_trajectory(
        model,
        chosen["id"],
        CFG["pilot_seed"],
        CFG["pilot_updates"],
        train,
        selection,
        windows,
        mapping,
        budget,
        executed,
        pilot=True,
    )
    model.eval()
    tokens = torch.from_numpy(selection[0].copy()).unsqueeze(0).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = float(nll(model, tokens).item())
    release(model)
    del model
    reloaded, _ = load_model(snapshot, chosen["removed"], CFG["pilot_seed"])
    state = torch.load(
        CACHE / "pilot_adapter" / str(CFG["pilot_seed"]) / chosen["id"] / "step_10.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_adapter(reloaded, state["adapter"])
    reloaded.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = float(nll(reloaded, tokens).item())
    if abs(actual - expected) > 1e-6:
        raise RuntimeError("Pilot save/reload changes evaluation NLL")
    release(reloaded)
    del reloaded
    cutoff = float(np.log(CFG["proposed_quality_max_ppl_ratio"]))
    eligible = [
        name for name, loss in initial_scores.items() if loss - parent_nll <= cutoff
    ]
    update_seconds = float(max(trajectory["update_seconds"]))
    forward_seconds = float(max([parent_time, *trajectory["eval_window_seconds"]]))
    construction = float(max(m["construction_seconds"] for m in mappings.values()))
    io_seconds = float(max(trajectory["checkpoint_seconds"]))
    spent = budget.record["spent_seconds"] + time.monotonic() - budget.start
    remaining_estimate = (
        1600 * update_seconds
        + (16 * 4 * 64 + 17 * 64) * forward_seconds
        + 33 * construction
        + 64 * io_seconds
    )
    projection = (
        spent
        + CFG["budget_safety_factor"] * remaining_estimate
        + CFG["budget_fixed_reserve_seconds"]
    )
    quality_ok = (
        len(eligible) >= 2 and trajectory["s_scores"]["10"] - parent_nll <= cutoff
    )
    data.write_json(
        OUT / "pilot.json",
        {
            "phase": "PILOT_COMPLETE",
            "pilot_executed_code_sha": executed,
            "ended_at_utc": now(),
            "engineering_passed": True,
            "save_reload_absolute_nll_difference": abs(actual - expected),
            "parent_s_nll": parent_nll,
            "initial_s_nll": initial_scores,
            "candidate_mappings": mappings,
            "quality_proposal": {
                "max_ppl_ratio": CFG["proposed_quality_max_ppl_ratio"],
                "max_excess_nll": cutoff,
                "source": "Stage0 working tolerance, not literature fact",
                "initial_eligible": eligible,
                "formal_entry_quality_passed": quality_ok,
            },
            "trajectory": trajectory,
            "projected_total_gpu_seconds": projection,
            "budget_feasible": projection <= CFG["gpu_seconds_limit"],
            "unit_costs": {
                "update_seconds": update_seconds,
                "eval_window_seconds": forward_seconds,
                "construction_seconds": construction,
                "checkpoint_seconds": io_seconds,
                "historical_preselection_estimate_seconds": (66 * 96 + 96)
                * forward_seconds,
            },
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "e_accessed": False,
        },
    )


def freeze() -> None:
    """After pilot review, bind the feasible recipe without changing any thresholds."""
    validate_artifacts()
    result = data.read_json(OUT / "pilot.json")
    ledger = data.read_json(OUT / "gpu_budget.json")
    if (
        not result["engineering_passed"]
        or not result["budget_feasible"]
        or not result["quality_proposal"]["formal_entry_quality_passed"]
    ):
        raise RuntimeError(
            "Pilot engineering, quality or complete-budget gate did not pass"
        )
    if ledger["phases"][-1]["status"] != "COMPLETE":
        raise RuntimeError("Pilot GPU phase did not finish")
    data.write_json(
        OUT / "formal_freeze.json",
        {
            "design_version": CFG["design_version"],
            "frozen_at_utc": now(),
            "config_sha256": data.file_hash(data.CONFIG),
            "protocol_sha256": data.file_hash(PROTOCOL),
            "pilot_sha256": data.file_hash(OUT / "pilot.json"),
            "quality_max_ppl_ratio": CFG["proposed_quality_max_ppl_ratio"],
            "data_manifest_sha256": data.file_hash(OUT / "data_manifest.json"),
            "candidate_manifest_sha256": data.file_hash(
                OUT / "candidate_manifest.json"
            ),
            "gpu_budget_sha256": data.file_hash(OUT / "gpu_budget.json"),
            "e_accessed": False,
            "quality_reason": "At least two initial S candidates and the pilot endpoint meet the prespecified relative-parent tolerance; final quality remains to be evaluated on E.",
        },
    )


def formal(snapshot: Path, budget: GpuBudget, executed: str) -> None:
    """Complete 16 trajectories, persist choices, then and only then open E."""
    import torch
    from .stage0_model import load_adapter

    manifest, pool = validate_artifacts()
    frozen = data.read_json(OUT / "formal_freeze.json")
    for key, path in [
        ("config_sha256", data.CONFIG),
        ("protocol_sha256", PROTOCOL),
        ("pilot_sha256", OUT / "pilot.json"),
        ("data_manifest_sha256", OUT / "data_manifest.json"),
        ("candidate_manifest_sha256", OUT / "candidate_manifest.json"),
    ]:
        if frozen[key] != data.file_hash(path):
            raise RuntimeError("Formal freeze changed")
    # Freeze must be a tracked committed object, not a local afterthought.
    committed = subprocess.check_output(
        [
            "git",
            "show",
            f"{executed}:{(OUT / 'formal_freeze.json').relative_to(data.ROOT)}",
        ],
        cwd=data.ROOT,
    )
    if committed != (OUT / "formal_freeze.json").read_bytes():
        raise RuntimeError("Formal freeze not committed in executed revision")
    train = load_split("T", manifest)
    selection = load_split("S", manifest)
    complete = []
    s_scores = {}
    capacity = None
    for seed in CFG["training_seeds"]:
        s_scores[str(seed)] = {str(step): {} for step in CFG["checkpoints"]}
        for candidate in pool["candidates"]:
            budget.check()
            model, mapping = load_model(snapshot, candidate["removed"], seed)
            capacity = capacity or mapping["adapter_parameters"]
            if mapping["adapter_parameters"] != capacity:
                raise RuntimeError("Unequal adapter capacity")
            result = train_trajectory(
                model,
                candidate["id"],
                seed,
                CFG["updates"],
                train,
                selection,
                manifest["splits"]["S"]["windows"],
                mapping,
                budget,
                executed,
            )
            complete.append(result)
            data.write_json(CACHE / "trajectory_progress.json", complete, replace=True)
            for step, loss in result["s_scores"].items():
                s_scores[str(seed)][step][candidate["id"]] = loss
            release(model)
            del model
    data.write_json(OUT / "trajectories.json", complete)
    data.write_json(OUT / "s_scores.json", s_scores)
    choices = {
        seed: analysis.replay({int(step): row for step, row in scores.items()})
        for seed, scores in s_scores.items()
    }
    data.write_json(
        OUT / "selection.json",
        {
            "frozen_at_utc": now(),
            "executed_code_sha": executed,
            "trajectory_sha256": data.file_hash(OUT / "trajectories.json"),
            "s_scores_sha256": data.file_hash(OUT / "s_scores.json"),
            "e_previously_opened": False,
            "choices": choices,
        },
    )
    evaluation = load_split("E", manifest, e_unsealed=True)
    data.write_json(
        OUT / "e_unsealed.json",
        {
            "at_utc": now(),
            "selection_sha256": data.file_hash(OUT / "selection.json"),
            "completed_trajectories": 16,
        },
    )
    e_losses: dict[str, Any] = {}
    for seed in CFG["training_seeds"]:
        e_losses[str(seed)] = {}
        for candidate in pool["candidates"]:
            model, _ = load_model(snapshot, candidate["removed"], seed)
            state = torch.load(
                CACHE / "adapters" / str(seed) / candidate["id"] / "step_100.pt",
                map_location="cpu",
                weights_only=True,
            )
            load_adapter(model, state["adapter"])
            values, _ = evaluate(
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
            release(model)
            del model
    parent, _ = load_model(snapshot, None, None)
    values, _ = evaluate(
        parent,
        evaluation,
        manifest["splits"]["E"]["windows"],
        "E",
        "parent",
        "parent",
        0,
        budget,
    )
    e_losses["parent"] = values
    release(parent)
    del parent
    data.write_json(OUT / "e_losses.json", e_losses)
    data.write_json(
        OUT / "runtime.json",
        {
            "phase": "EVALUATED",
            "executed_code_sha": executed,
            "design_version": CFG["design_version"],
            "gpu_name": torch.cuda.get_device_name(0),
            "selected_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "finished_at_utc": now(),
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
            "trajectory_count": 16,
            "training_updates": 1600,
            "s_forward_windows": 4096,
            "e_forward_windows": 1088,
        },
    )


def analyze() -> None:
    """CPU-only statistical analysis of completed, unchanged measurements."""
    started = time.monotonic()
    selection = validate_selection()
    runtime = data.read_json(OUT / "runtime.json")
    if runtime["phase"] != "EVALUATED":
        raise RuntimeError("Formal evaluation incomplete")
    manifest = data.read_json(OUT / "data_manifest.json")
    complete_rows = validate_measurement_rows(manifest)
    windows = manifest["splits"]["E"]["windows"]
    result = analysis.analyze(
        selection["choices"],
        data.read_json(OUT / "s_scores.json"),
        data.read_json(OUT / "e_losses.json"),
        [w["doc_id"] for w in windows],
        [w["labels"] for w in windows],
        CFG,
        data.read_json(OUT / "pilot.json")["unit_costs"],
    )
    elapsed = time.monotonic() - started
    if elapsed > CFG["cpu_seconds_limit"]:
        raise RuntimeError("CPU analysis budget exceeded")
    result.update(
        executed_code_sha=runtime["executed_code_sha"],
        analysis_seconds=elapsed,
        interpretation="internal exploratory CE/LoRA probe; not FHRD reproduction or a paper claim",
    )
    result["measured_audit_cost"] = {
        "gpu_elapsed_seconds_including_pilot": data.read_json(OUT / "gpu_budget.json")[
            "spent_seconds"
        ],
        "formal_training_updates": 1600,
        "formal_input_tokens": 1600 * 4 * 512,
        "formal_effective_labels": 1600 * 4 * 511,
        "formal_s_e_measurement_rows": len(complete_rows),
        "cpu_analysis_seconds": elapsed,
    }
    data.write_json(OUT / "metrics.json", result)
    aggregates: dict[tuple[str, ...], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    for row in complete_rows:
        key = tuple(
            row[k] for k in ("split", "seed", "candidate", "checkpoint", "doc_id")
        )
        aggregates[key][0] += float(row["nll"]) * int(row["labels"])
        aggregates[key][1] += int(row["labels"])
        aggregates[key][2] += 1
    for key, (total, labels, count) in sorted(aggregates.items()):
        record_csv(
            OUT / "per_document_nll.csv",
            {
                **dict(
                    zip(("split", "seed", "candidate", "checkpoint", "doc_id"), key)
                ),
                "windows": int(count),
                "labels": int(labels),
                "nll": total / labels,
            },
        )
    lines = [
        "# D2-Stage0 result",
        "",
        f"Decision: {result['decision']}",
        "",
        f"Executed code: `{runtime['executed_code_sha']}`. Design: {CFG['design_version']}.",
        "The archive commit is a later Git commit, not the executed model code.",
        "",
        "| Seed | G0 | 95% document interval | Quality candidates |",
        "| --- | ---: | --- | ---: |",
    ]
    for seed, row in result["seeds"].items():
        g = row["G0"]
        lines.append(
            f"| {seed} | {g['point']:.6f} | [{g['lower95']:.6f}, {g['upper95']:.6f}] | {row['quality_candidate_count']} |"
        )
        for strategy, policy in row["strategies"].items():
            record_csv(
                OUT / "strategy_results.csv",
                {
                    "seed": seed,
                    "strategy": strategy,
                    "selected": policy["selected"],
                    "e_nll": policy["endpoint_e_nll"],
                    **policy["difference_vs_ref"],
                    "quality_pass": policy["quality_pass"],
                },
            )
            record_csv(
                OUT / "costs.csv",
                {
                    "seed": seed,
                    "strategy": strategy,
                    "training_updates": policy["training_updates"],
                    "s_query_windows": policy["s_query_windows"],
                    "training_input_tokens": policy["training_input_tokens"],
                    "training_effective_labels": policy["training_effective_labels"],
                    "standalone_seconds_estimate": policy[
                        "standalone_seconds_estimate"
                    ],
                },
            )
    lines += [
        "",
        "Full trajectories are an audit cost; policy replay is not measured speedup.",
        "Quality, problem signal and simple-baseline sufficiency are separate in metrics.json.",
        "No H2, FHRD/RSC or other direction was run. Stop after this probe.",
    ]
    with (OUT / "summary.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    data.write_json(
        OUT / "state.json",
        {
            "phase": "COMPLETE",
            "decision": result["decision"],
            "executed_code_sha": runtime["executed_code_sha"],
            "ended_at_utc": now(),
        },
    )


def validate_measurement_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Reject incomplete, duplicate or mismapped candidate/window measurements."""
    names = [f"a{i:02d}" for i in range(8)]
    expected = {
        ("S", str(seed), name, step, window)
        for seed in CFG["training_seeds"]
        for name in names
        for step in CFG["checkpoints"]
        for window in range(64)
    }
    expected |= {
        ("E", str(seed), name, 100, window)
        for seed in CFG["training_seeds"]
        for name in names
        for window in range(64)
    }
    expected |= {("E", "parent", "parent", 0, window) for window in range(64)}
    with (OUT / "per_window_nll.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = []
    for row in rows:
        key = (
            row["split"],
            row["seed"],
            row["candidate"],
            int(row["checkpoint"]),
            int(row["window_id"]),
        )
        actual.append(key)
        if key not in expected:
            raise RuntimeError("Unexpected measurement identity")
        window = manifest["splits"][row["split"]]["windows"][key[-1]]
        if (
            row["doc_id"] != window["doc_id"]
            or int(row["document_token_start"]) != window["document_token_start"]
            or int(row["labels"]) != window["labels"]
            or not np.isfinite(float(row["nll"]))
        ):
            raise RuntimeError(
                "Measurement document, label or finite-value check failed"
            )
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("Duplicate or missing measurement rows")
    with (OUT / "training_curve.csv").open(newline="") as handle:
        curve = list(csv.DictReader(handle))
    expected_updates = {
        (str(seed), name, step)
        for seed in CFG["training_seeds"]
        for name in names
        for step in range(1, 101)
    }
    if (
        len(curve) != 1600
        or {(row["seed"], row["candidate"], int(row["update"])) for row in curve}
        != expected_updates
    ):
        raise RuntimeError("Incomplete training curve")
    return rows


def gpu_stage(stage: str, snapshot: Path) -> None:
    """Execute one unique single-A800 phase, preserving failure/time evidence."""
    from .stage0_resources import admission

    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected.isdigit():
        raise RuntimeError("One explicitly selected GPU index is required")
    # Guard direct Python entry as well as the detached launcher; no CUDA context yet.
    admission(stage, selected=int(selected))
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "A800" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("Exactly one explicitly selected A800 is required")
    executed = code_sha()
    validate_artifacts()
    check_snapshot(snapshot)
    budget = GpuBudget(stage)
    status = "FAILED"
    try:
        torch.cuda.reset_peak_memory_stats()
        (pilot if stage == "pilot" else formal)(snapshot, budget, executed)
        status = "COMPLETE"
    except BaseException as exc:
        data.write_json(
            OUT / f"{stage}_failure.json",
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "scientific_decision": "NOT_ASSESSED",
                "status": (
                    "BUDGET_STOP"
                    if isinstance(exc, BudgetStop)
                    else "RECOVERY_PIPELINE_FAILED"
                ),
                "executed_code_sha": executed,
                "at_utc": now(),
            },
            replace=True,
        )
        raise
    finally:
        budget.finish(status)


def main() -> None:
    """Explicit phase CLI; formal cannot bypass the post-pilot committed freeze."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "pilot", "freeze", "formal", "analyze"),
        required=True,
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            if args.source is None or args.tokenizer is None:
                parser.error("prepare requires --source and --tokenizer")
            prepare(args.source, args.tokenizer)
        elif args.stage == "freeze":
            freeze()
        elif args.stage == "analyze":
            analyze()
        else:
            if args.snapshot is None:
                parser.error("GPU stages require --snapshot")
            gpu_stage(args.stage, args.snapshot)


if __name__ == "__main__":
    main()
