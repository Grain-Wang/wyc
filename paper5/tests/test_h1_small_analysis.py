"""Check H1-small predictors against a known synthetic pair interaction."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper5.experiments.canary.direction_01.run_h1_small import (
    analyze_h1_small,
    load_h1_config,
    write_interaction_heatmap,
)

LAYERS = [1, 3, 6, 8, 10, 12, 15, 17, 19, 21, 24, 26]


def synthetic_windows(
    count: int,
    interaction_offset: float = 0.1,
    single_scale: float = 1.0,
) -> dict[tuple[int, ...], np.ndarray]:
    """Make fixed windows with controlled single and pair effects."""
    base = 3.0 + np.arange(count, dtype=np.float64) * 0.0001
    singles = {
        layer: single_scale * (0.02 + layer * 0.003 + (2**index) * 1e-7)
        for index, layer in enumerate(LAYERS)
    }
    windows = {(): base}
    for layer in LAYERS:
        windows[(layer,)] = base + singles[layer]
    for first, second in itertools.combinations(LAYERS, 2):
        windows[(first, second)] = (
            base + singles[first] + singles[second] + interaction_offset
        )
    return windows


class H1SmallAnalysisTest(unittest.TestCase):
    """Exercise held-out baselines and required H1 output shape."""

    def test_constant_interaction_is_not_a_ranking_failure(self) -> None:
        """A stable offset changes additive error without changing pair order."""
        rows, metrics = analyze_h1_small(
            synthetic_windows(96),
            synthetic_windows(64),
            LAYERS,
            calibration_shards=3,
            bootstrap_replicates=20,
            seed=7,
        )
        self.assertEqual(len(rows), 66)
        self.assertAlmostEqual(
            metrics["interaction_distribution_nll"]["mean_absolute"], 0.1
        )
        self.assertAlmostEqual(metrics["baseline_metrics"]["additive"]["mae"], 0.1)
        self.assertLess(metrics["baseline_metrics"]["mean_interaction"]["mae"], 1e-12)
        self.assertLess(metrics["baseline_metrics"]["linear_regression"]["mae"], 1e-12)
        self.assertAlmostEqual(metrics["baseline_metrics"]["additive"]["spearman"], 1.0)
        self.assertEqual(metrics["protocol_point_verdict"], "INCONCLUSIVE")
        self.assertEqual(metrics["h1_verdict"], "INCONCLUSIVE")
        self.assertFalse(metrics["stable_confirmation"])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "interaction_heatmap.svg"
            write_interaction_heatmap(image_path, rows, LAYERS)
            self.assertIn(
                "H1-small interaction", image_path.read_text(encoding="utf-8")
            )

    def test_signed_pair_residual_uses_same_split_single_effects(self) -> None:
        """Opposite pair effects appear with the correct signs and NLL units."""
        calibration = synthetic_windows(96)
        validation = synthetic_windows(64)
        validation[(1, 3)] = validation[(1, 3)] + 0.2
        validation[(24, 26)] = validation[(24, 26)] - 0.3
        rows, metrics = analyze_h1_small(
            calibration,
            validation,
            LAYERS,
            calibration_shards=3,
            bootstrap_replicates=20,
            seed=11,
        )
        by_pair = {(row["first_layer"], row["second_layer"]): row for row in rows}
        self.assertAlmostEqual(by_pair[(1, 3)]["interaction_nll"], 0.3)
        self.assertAlmostEqual(by_pair[(24, 26)]["interaction_nll"], -0.2)
        self.assertEqual(metrics["interaction_distribution_nll"]["negative_count"], 1)

    def test_learned_baselines_fit_calibration_and_use_validation_features(
        self,
    ) -> None:
        """Validation pair labels cannot affect either learned predictor."""
        calibration = synthetic_windows(96, interaction_offset=0.1)
        validation = synthetic_windows(64, interaction_offset=0.3, single_scale=2.0)
        rows, metrics = analyze_h1_small(
            calibration,
            validation,
            LAYERS,
            calibration_shards=3,
            bootstrap_replicates=20,
            seed=13,
        )
        self.assertEqual(metrics["split_contract"]["fit_split"], "calibration")
        self.assertFalse(
            metrics["split_contract"]["validation_pair_labels_used_for_fitting"]
        )
        self.assertAlmostEqual(metrics["mean_calibration_interaction_nll"], 0.1)
        self.assertAlmostEqual(
            metrics["baseline_metrics"]["mean_interaction"]["mae"], 0.2
        )
        self.assertAlmostEqual(
            metrics["baseline_metrics"]["linear_regression"]["mae"], 0.2
        )
        first = rows[0]
        self.assertAlmostEqual(
            first["additive_prediction_nll"],
            first["delta_first_nll"] + first["delta_second_nll"],
        )
        self.assertAlmostEqual(
            first["mean_interaction_prediction_nll"],
            first["additive_prediction_nll"] + 0.1,
        )

    def test_config_locks_protocol_scale_and_output(self) -> None:
        """The executable config fixes all 12 edits, 66 pairs, and output path."""
        config_path = (
            Path(__file__).parents[1]
            / "configs"
            / "canary"
            / "direction_01"
            / "h1_config.yaml"
        )
        config = load_h1_config(config_path)
        self.assertEqual(config["edit"]["layer_indices"], LAYERS)
        self.assertEqual(config["pair_sampling"]["pair_count"], 66)
        self.assertEqual(
            config["output_directory"],
            "paper5/results/canary/direction_01/H1_small",
        )
        self.assertEqual(config["baselines"]["fit_split"], "calibration")
        self.assertEqual(config["baselines"]["evaluation_split"], "validation")


if __name__ == "__main__":
    unittest.main()
