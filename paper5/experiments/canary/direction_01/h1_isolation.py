"""CPU-only source-token isolation for the once-authorized V2 amendment."""

from __future__ import annotations

import hashlib
from typing import Any

VERSION = "H1_multi_edit_isolation_v2"


def overlap_length(first: tuple[int, int], second: tuple[int, int]) -> int:
    """Intersect half-open source-token ranges, never integer chunk labels."""
    if any(start < 0 or stop <= start for start, stop in (first, second)):
        raise ValueError("Invalid source-token range")
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def select_windows(
    count: int,
    old: list[int],
    exposures: list[dict[str, Any]],
    coordinate: dict[str, Any],
) -> dict[str, Any]:
    """Exclude verified exposures plus neighbors, then apply the original hash."""
    if len(old) != 64 or len(set(old)) != 64 or not all(0 <= j < count for j in old):
        raise ValueError("Invalid V_old manifest")
    reasons: dict[int, list[str]] = {}

    def exclude(center: int, reason: str) -> None:
        for j in (center - 1, center, center + 1):
            if 0 <= j < count:
                label = reason if j == center else f"neighbor_of_{center}: {reason}"
                reasons.setdefault(j, []).append(label)

    for j in old:
        exclude(j, "V_old")
    overlaps = []
    for exposure in exposures:
        if exposure["coordinate"] != coordinate:
            raise ValueError(
                "Incompatible source/token coordinate; explicit mapping required"
            )
        start, stop = exposure["source_token_range"]
        if start < 0 or stop <= start or stop > coordinate["token_count"]:
            raise ValueError("Invalid historical exposure range")
        for j in range(start // 512, min(count, (stop - 1) // 512 + 1)):
            shared = overlap_length((start, stop), (512 * j, 512 * (j + 1)))
            if shared:
                exclude(j, exposure["id"])
                overlaps.append(
                    {"exposure": exposure["id"], "window": j, "tokens": shared}
                )
    eligible = [j for j in range(count) if j not in reasons]
    if len(eligible) < 64:
        raise ValueError("Fewer than 64 eligible V2 windows; stop")
    selected = sorted(
        eligible,
        key=lambda j: (hashlib.sha256(f"20260922:holdout:{j}".encode()).hexdigest(), j),
    )[:64]
    return {
        "version": VERSION,
        "selected": selected,
        "eligible_count": len(eligible),
        "exclusions": {str(j): sorted(set(reasons[j])) for j in sorted(reasons)},
        "overlaps": overlaps,
    }
