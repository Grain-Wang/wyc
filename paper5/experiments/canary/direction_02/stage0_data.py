"""Document-disjoint Stage0 preparation; no model forward or validation-label use."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "paper5/configs/canary/direction_02/recovery_stage0.json"
D1 = ROOT / "paper5/results/canary/direction_01/H1_small"


def digest(data: bytes) -> str:
    """Return SHA256 of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    """Canonical finite JSON for frozen identities."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def file_hash(path: Path) -> str:
    """Stream a file hash without retaining model weights in RAM."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text())


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    """Write finite JSON, rejecting accidental replacement of frozen evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not replace:
        with path.open("x") as handle:
            handle.write(content)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)


def config() -> dict[str, Any]:
    """Read the single experiment configuration."""
    return read_json(CONFIG)


def documents(raw: bytes) -> list[dict[str, Any]]:
    """Parse whole WikiText articles with byte coordinates and content IDs."""
    starts: list[int] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        if re.fullmatch(r"= [^=]+ =", line.decode("utf-8").strip()):
            starts.append(offset)
        offset += len(line)
    if not starts or raw[: starts[0]].strip():
        raise ValueError("Unrecognized article boundaries or nonempty preamble")
    result = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(raw)
        content = raw[start:end]
        # Whitespace-only duplicates must not be assigned to different splits.
        identity = digest(" ".join(content.decode("utf-8").split()).encode())
        result.append(
            {
                "doc_id": identity,
                "source_start_byte": start,
                "source_end_byte": end,
                "source_document_index": index,
                "text": content.decode("utf-8"),
            }
        )
    return result


def split_name(doc_id: str, seed: int) -> str:
    """Assign an entire normalized document without consulting its losses."""
    bucket = int(digest(f"{seed}:split:{doc_id}".encode()), 16) % 100
    return "T" if bucket < 70 else "S" if bucket < 85 else "E"


def make_dataset(
    raw: bytes, tokenize: Callable[[str], list[int]], cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Freeze disjoint windows without concatenating separate documents."""
    seq = cfg["sequence_length"]
    seed = cfg["data_seed"]
    groups: dict[str, dict[str, list[tuple[dict[str, Any], list[int]]]]] = {
        key: {} for key in ("T", "S", "E")
    }
    doc_records = []
    seen: set[str] = set()
    for doc in documents(raw):
        identity = doc["doc_id"]
        if identity in seen:
            continue
        seen.add(identity)
        split = split_name(identity, seed)
        tokens = tokenize(doc["text"])
        record = {key: value for key, value in doc.items() if key != "text"}
        record.update(
            split=split, token_count=len(tokens), discarded_tail=len(tokens) % seq
        )
        doc_records.append(record)
        windows = []
        for start in range(0, len(tokens) - seq + 1, seq):
            window = {
                "doc_id": identity,
                "document_token_start": start,
                "document_token_end": start + seq,
                "labels": seq - 1,
            }
            windows.append((window, tokens[start : start + seq]))
        windows.sort(
            key=lambda pair: digest(
                f"{seed}:window:{identity}:{pair[0]['document_token_start']}".encode()
            )
        )
        if windows:
            groups[split][identity] = windows
    arrays = {}
    lists = {}
    for split, wanted in cfg["windows"].items():
        available = groups[split]
        ids = sorted(
            available, key=lambda name: digest(f"{seed}:document:{name}".encode())
        )
        ordered = [
            available[name][offset]
            for offset in range(max((len(v) for v in available.values()), default=0))
            for name in ids
            if offset < len(available[name])
        ]
        if len(ordered) < wanted:
            raise ValueError(f"Insufficient {split} windows: {len(ordered)} < {wanted}")
        selected = ordered[:wanted]
        arrays[split] = np.asarray([pair[1] for pair in selected], dtype="<i8")
        lists[split] = {
            "available_windows": len(ordered),
            "selected_document_count": len({pair[0]["doc_id"] for pair in selected}),
            "windows": [
                dict(pair[0], window_id=index) for index, pair in enumerate(selected)
            ],
            "input_tokens": wanted * seq,
            "effective_labels": wanted * (seq - 1),
            "token_data_sha256": digest(arrays[split].tobytes()),
        }
    memberships = [
        {w["doc_id"] for w in lists[split]["windows"]} for split in ("T", "S", "E")
    ]
    if any(memberships[i] & memberships[j] for i in range(3) for j in range(i)):
        raise ValueError("Cross-split document contamination")
    return arrays, {
        "source_sha256": digest(raw),
        "documents": doc_records,
        "splits": lists,
        "data_seed": seed,
        "sequence_length": seq,
        "add_special_tokens": False,
        "historical_exposure": "D1 train calibration reused; internal exploratory evaluation",
    }


def candidates(csv_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Preselect eight architectures strictly from historical calibration rows."""
    values: dict[tuple[int, int], list[float]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "calibration":
                continue  # Do not even parse historical validation NLL values.
            if row["first_layer"] and row["second_layer"]:
                pair = (int(row["first_layer"]), int(row["second_layer"]))
                values[pair].append(float(row["nll"]))
    if len(values) != 66 or any(
        len(v) != 96 or not np.isfinite(v).all() for v in values.values()
    ):
        raise ValueError("Expected 66 finite historical calibration pairs × 96 windows")
    ranking = sorted(values, key=lambda pair: (float(np.mean(values[pair])), pair))
    pool = ranking[: cfg["competitive_pool_size"]]
    others = sorted(
        pool[1:],
        key=lambda pair: digest(
            f"{cfg['candidate_seed']}:candidate:".encode() + canonical(pair)
        ),
    )
    selected = [pool[0], *others[: cfg["candidate_count"] - 1]]
    if len(set(selected)) != 8:
        raise ValueError("Candidate selection is not eight unique pairs")
    return {
        "source_sha256": file_hash(csv_path),
        "source_commit": cfg["d1_archive_commit"],
        "used_split": "calibration",
        "excluded_split": "validation",
        "competitive_pool": [list(pair) for pair in pool],
        "historical_candidate_windows": 66 * 96,
        "historical_parent_windows": 96,
        "candidate_seed": cfg["candidate_seed"],
        "candidates": [
            {
                "id": f"a{index:02d}",
                "removed": list(pair),
                "historical_calibration_nll": float(np.mean(values[pair])),
            }
            for index, pair in enumerate(selected)
        ],
    }


def training_order(count: int, seed: int) -> list[int]:
    """The same seed uses the same entire no-replacement T order in every model."""
    return np.random.default_rng(seed).permutation(count).tolist()
