"""Descriptive smoke-only interaction analysis and artifact writers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return zero-based average ranks with deterministic tie handling."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = (cursor + end - 1) / 2.0
        for index, _ in indexed[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Pearson correlation, returning zero for a constant vector."""
    if len(left) != len(right) or not left:
        raise ValueError("Correlation inputs must be non-empty and equally sized.")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman correlation using average ranks."""
    return pearson(_average_ranks(left), _average_ranks(right))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Kendall tau-b for small smoke-test candidate sets."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError(
            "Kendall inputs must be equally sized with at least two items."
        )
    concordant = 0
    discordant = 0
    left_ties = 0
    right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_sign = (left[first] > left[second]) - (left[first] < left[second])
            right_sign = (right[first] > right[second]) - (right[first] < right[second])
            if left_sign == 0 and right_sign == 0:
                continue
            if left_sign == 0:
                left_ties += 1
            elif right_sign == 0:
                right_ties += 1
            elif left_sign == right_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties) * (concordant + discordant + right_ties)
    )
    if denominator == 0.0:
        return 0.0
    return (concordant - discordant) / denominator


def summarize_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize additive prediction behavior without making an H1 verdict."""
    additive = [float(row["additive_delta_nll"]) for row in rows]
    actual = [float(row["actual_delta_nll"]) for row in rows]
    interactions = [float(row["interaction_nll"]) for row in rows]
    relative = [float(row["relative_abs_interaction"]) for row in rows]
    absolute_errors = [
        abs(prediction - truth) for prediction, truth in zip(additive, actual)
    ]
    return {
        "pair_count": len(rows),
        "pearson_additive_vs_actual": pearson(additive, actual),
        "spearman_additive_vs_actual": spearman(additive, actual),
        "kendall_tau_b_additive_vs_actual": kendall_tau_b(additive, actual),
        "additive_mae": sum(absolute_errors) / len(absolute_errors),
        "mean_abs_interaction": sum(abs(value) for value in interactions)
        / len(interactions),
        "fraction_relative_abs_interaction_gt_0_10": sum(
            value > 0.10 for value in relative
        )
        / len(relative),
        "amplification_count": sum(value > 0.0 for value in interactions),
        "cancellation_count": sum(value < 0.0 for value in interactions),
        "zero_count": sum(value == 0.0 for value in interactions),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to a deterministic UTF-8 CSV file."""
    if not rows:
        raise ValueError("Cannot write an empty smoke-test CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_interaction_svg(
    path: Path, rows: Sequence[Mapping[str, Any]], title: str
) -> None:
    """Write a dependency-free SVG of additive versus measured pair damage."""
    width = 640
    height = 480
    margin = 60
    additive = [float(row["additive_delta_nll"]) for row in rows]
    actual = [float(row["actual_delta_nll"]) for row in rows]
    low = min(additive + actual)
    high = max(additive + actual)
    span = max(high - low, 1e-12)

    def x_coord(value: float) -> float:
        return margin + (value - low) / span * (width - 2 * margin)

    def y_coord(value: float) -> float:
        return height - margin - (value - low) / span * (height - 2 * margin)

    circles = "\n".join(
        f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="4" fill="#2563eb" />'
        for x, y in zip(additive, actual)
    )
    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">{escaped_title}</text>
<line x1="{x_coord(low):.2f}" y1="{y_coord(low):.2f}" x2="{x_coord(high):.2f}" y2="{y_coord(high):.2f}" stroke="#9ca3af" stroke-dasharray="5,5" />
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black" />
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black" />
<text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-family="sans-serif" font-size="13">Additive delta NLL</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="sans-serif" font-size="13">Measured pair delta NLL</text>
{circles}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_summary(
    path: Path,
    metadata: Mapping[str, Any],
    checks: Mapping[str, Any],
    analysis: Mapping[str, Any],
    pipeline_status: str,
) -> None:
    """Write the smoke summary with an explicit scientific non-claim."""
    lines = [
        "# Direction 1 Local Smoke Test",
        "",
        "> **THIS IS NOT A FORMAL SCIENTIFIC CANARY RESULT.**",
        ">",
        "> All measurements below are **SMOKE TEST / PRELIMINARY** and must not be used to claim H1 or H2.",
        "",
        f"## Pipeline status: {pipeline_status}",
        "",
        f"- Model: `{metadata['model_id']}` at `{metadata['resolved_revision']}`",
        f"- Device: `{metadata['gpu_name']}`",
        "- OOM: `False`",
        f"- Edit family: `{metadata['edit_family']}`",
        f"- Unique single edits: {metadata['single_edit_count']}",
        f"- Unique pairs: {metadata['pair_count']}",
        f"- Calibration: {metadata['calibration_sequences']} × {metadata['sequence_length']} tokens",
        f"- Validation: {metadata['validation_sequences']} × {metadata['sequence_length']} tokens",
        f"- Peak allocated GPU memory: {metadata['peak_gpu_memory_mib']:.2f} MiB",
        f"- Runtime: {metadata['runtime_seconds']:.2f} seconds",
        "",
        "## Pipeline checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "## Descriptive pipeline outputs",
            "",
            "These values only demonstrate that additive-analysis code ran end to end. The sample is too small and the hardware/protocol are not the formal A800 design.",
            "",
        ]
    )
    for split, values in analysis.items():
        lines.append(f"### {split}")
        lines.append("")
        for name, value in values.items():
            lines.append(f"- `{name}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## H2 status",
            "",
            "H2 runtime validation deferred to A800. Only a toy-tensor JVP shape/value unit test was run locally.",
            "",
            "## Scientific verdict",
            "",
            "No H1 or H2 verdict is permitted from this smoke test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
