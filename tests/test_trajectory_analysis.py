import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.trajectory_analysis import (
    AnalysisSettings,
    analyze_run,
    normalize_estimate,
    prepare_trajectories,
    resample_trajectory,
    run_analysis,
    rollout_window_metrics,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrajectoryAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.run = self.runs / "synthetic_model_20260825_000000"
        self.run.mkdir(parents=True)
        _write_json(self.run / "threshold.json", {"threshold": 100})
        _write_json(self.run / "config.json", {"model": "synthetic", "count": 6})
        _write_json(self.run / "trajectories.json", {
            "baseline": [
                [50, 100],
                [100, 100],
                None,
                [1],
                ["bad", 2],
                [50, 2000],
            ],
            "below_good": [
                [200, 150, 100],
                [50, 75, 90],
                [100, 100],
            ],
            "above_good": [
                [200, 250, 300],
                [50, 100, 150],
                [100, 110],
                [100, 2000],
            ],
        })
        # Deliberately mirrors shipped runs: conditioned final estimates absent.
        _write_json(self.run / "estimates.json", {
            "baseline": [100, None, 200, None, None, 2000],
        })
        self.settings = AnalysisSettings(
            seed=17,
            grid_points=20,
            bootstrap_resamples=30,
            confidence=0.90,
            figure_dpi=40,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_normalization_resampling_and_window_drift(self) -> None:
        self.assertEqual(normalize_estimate(150, 100), 0.5)
        np.testing.assert_allclose(resample_trajectory([0, 10], 3), [0, 5, 10])
        records, _ = prepare_trajectories([[50, 150]], 100)
        metrics = rollout_window_metrics(
            records[0], 100, grid_points=10, window_fraction=0.2
        )
        self.assertAlmostEqual(metrics["early_mean_estimate"], 50 + 100 / 18)
        self.assertAlmostEqual(metrics["late_mean_estimate"], 150 - 100 / 18)
        self.assertGreater(metrics["within_trajectory_drift_mean_normalized"], 0)
        self.assertEqual(metrics["reasoning_trajectory_endpoint"], 150)

    def test_quality_counts_strata_and_explicit_filtered_mrf(self) -> None:
        result = analyze_run(self.run, self.settings)
        quality = result["trajectory_quality"]["baseline"]
        self.assertEqual(quality["total_raw"], 6)
        self.assertEqual(quality["valid_unfiltered"], 3)
        self.assertEqual(quality["invalid"], 3)
        self.assertEqual(quality["outlier_among_valid"], 1)
        self.assertEqual(quality["valid_after_10x_filter"], 2)
        self.assertEqual(quality["stratum_counts_unfiltered"]["start_below"], 2)
        self.assertEqual(quality["stratum_counts_unfiltered"]["start_equal"], 1)

        mrf = result["trajectory_analysis"]["mrf"]
        self.assertIn("starter_compatible_unfiltered", mrf)
        self.assertIn("robustness_10x_filtered", mrf)
        unfiltered = mrf["starter_compatible_unfiltered"]["by_stratum"]["pooled"]
        filtered = mrf["robustness_10x_filtered"]["by_stratum"]["pooled"]
        self.assertEqual(unfiltered["n_above"], 4)
        self.assertEqual(filtered["n_above"], 3)
        self.assertIsNotNone(unfiltered["motivated_reasoning_factor"])
        self.assertIsNotNone(unfiltered["bootstrap_difference_in_medians"])

        modes = result["trajectory_analysis"]["filter_modes"]
        for stratum in ("pooled", "start_above", "start_below", "start_equal"):
            self.assertIn(stratum, modes["unfiltered"]["baseline"])

    def test_missing_conditioned_final_estimates_never_use_endpoint(self) -> None:
        result = analyze_run(self.run, self.settings)
        final = result["visible_final_estimate_distribution"]
        self.assertEqual(final["source"], "estimates.json only")
        for condition in ("below_good", "above_good"):
            artifact = final["conditions"][condition]
            self.assertEqual(artifact["artifact_status"], "missing_condition_artifact")
            self.assertFalse(artifact["available"])
            self.assertFalse(artifact["fallback_used"])
        self.assertIsNone(final["above_minus_below_difference_in_medians"])

        endpoints = result["trajectory_analysis"]["reasoning_trajectory_endpoint_distribution"]
        self.assertIn("not visible final answer", endpoints["label"])
        self.assertGreater(endpoints["conditions"]["above_good"]["n"], 0)

    def test_output_is_separate_reproducible_and_never_overwritten(self) -> None:
        before = {path.name: _digest(path) for path in self.run.iterdir() if path.is_file()}
        output = self.root / "derived" / "E02_test"
        paths = run_analysis(
            self.run, output, self.settings, runs_root=self.runs
        )
        for path in paths.values():
            self.assertTrue(path.is_file())
        analysis = json.loads(paths["analysis"].read_text(encoding="utf-8"))
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        self.assertEqual(analysis["settings"]["seed"], 17)
        self.assertEqual(provenance["source_run"], str(self.run.resolve()))
        self.assertIn("trajectories.json", provenance["input_artifacts"])
        after = {path.name: _digest(path) for path in self.run.iterdir() if path.is_file()}
        self.assertEqual(before, after)

        with self.assertRaises(FileExistsError):
            run_analysis(self.run, output, self.settings, runs_root=self.runs)
        with self.assertRaises(ValueError):
            run_analysis(
                self.run,
                self.runs / "derived_should_be_rejected",
                self.settings,
                runs_root=self.runs,
            )


if __name__ == "__main__":
    unittest.main()
