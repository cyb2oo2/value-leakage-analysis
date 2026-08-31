from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.side_mechanics import (
    analyze_run,
    classify_rollout,
    condition_favored_of,
    gap_change_of,
    permutation_delta,
    placebo_grid,
    run_analysis,
    side_of,
    summarize_condition,
    toward_threshold_of,
    wilson_interval,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _classified(values, threshold=100.0, condition="above_good", index=0):
    return classify_rollout(
        values,
        threshold,
        condition,
        rollout_index=index,
        outlier_10x=False,
    )


class SideMechanicUnitTests(unittest.TestCase):
    def test_wilson_is_bounded_and_symmetric_at_one_half(self) -> None:
        emptyish = wilson_interval(0, 10)
        self.assertEqual(emptyish.p, 0.0)
        self.assertGreaterEqual(emptyish.low, 0.0)
        self.assertLess(emptyish.high, 0.4)
        mid = wilson_interval(50, 100)
        self.assertAlmostEqual(mid.p, 0.5)
        self.assertAlmostEqual(mid.low + mid.high, 1.0, places=6)
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)

    def test_side_revision_and_toward_definitions(self) -> None:
        self.assertEqual(side_of(41_000_001, 41_000_000), "above")
        self.assertEqual(side_of(41_000_000, 41_000_000), "equal")
        self.assertEqual(side_of(40_999_999, 41_000_000), "below")
        self.assertEqual(toward_threshold_of("below", "up"), "toward")
        self.assertEqual(toward_threshold_of("above", "down"), "toward")
        self.assertEqual(toward_threshold_of("below", "down"), "away")
        self.assertEqual(toward_threshold_of("equal", "up"), "away")
        self.assertEqual(toward_threshold_of("above", "none"), "none")
        self.assertEqual(condition_favored_of("above_good", "up"), "favored")
        self.assertEqual(condition_favored_of("below_good", "down"), "favored")
        self.assertEqual(condition_favored_of("above_good", "down"), "opposed")
        self.assertEqual(condition_favored_of("baseline", "up"), "not_applicable")

    def test_gap_change_distinguishes_overshoot_from_seeking(self) -> None:
        self.assertEqual(gap_change_of(80, 50, 41), "shrink")
        self.assertEqual(gap_change_of(80, 1, 41), "grow")
        toward_overshoot = _classified([80, 1], threshold=41)
        self.assertEqual(toward_overshoot.toward_threshold, "toward")
        self.assertEqual(toward_overshoot.gap_change, "grow")
        self.assertTrue(toward_overshoot.crossed)
        self.assertAlmostEqual(toward_overshoot.first_cross_frac or -1, 1.0)

    def test_first_cross_fraction_uses_the_first_side_change(self) -> None:
        row = _classified([10, 10, 150, 90], threshold=100)
        self.assertAlmostEqual(row.first_cross_frac or -1, 2 / 3)

    def test_summarize_tracks_toward_not_favored(self) -> None:
        rows = [
            _classified([50, 90], condition="below_good", index=0),
            _classified([150, 110], condition="below_good", index=1),
            _classified([150, 200], condition="below_good", index=2),
        ]
        summary = summarize_condition(rows, "below_good")
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["p_toward_given_directional"]["k"], 2)
        self.assertEqual(summary["p_favored_given_directional"]["k"], 1)
        self.assertEqual(summary["p_last_favored_side"]["k"], 1)

    def test_permutation_detects_separated_groups(self) -> None:
        result = permutation_delta(
            [True] * 12,
            [False] * 12,
            n_perm=400,
            seed=7,
        )
        self.assertEqual(result["observed"], 1.0)
        self.assertLess(result["p_two_sided"], 0.01)

    def test_permutation_is_null_when_labels_are_exchangeable(self) -> None:
        flags = [True, False, True, False, True, False]
        result = permutation_delta(flags, flags, n_perm=300, seed=11)
        self.assertEqual(result["observed"], 0.0)
        self.assertGreater(result["p_two_sided"], 0.5)

    def test_up_permutation_detects_condition_linked_revision(self) -> None:
        result = permutation_delta(
            [True] * 8,
            [False] * 8,
            n_perm=400,
            seed=3,
        )
        self.assertEqual(result["observed"], 1.0)
        self.assertLess(result["p_two_sided"], 0.01)

    def test_placebo_gap_shrink_peaks_at_the_true_threshold(self) -> None:
        rows = [
            _classified([70, 55], threshold=50, condition="baseline", index=0),
            _classified([30, 45], threshold=50, condition="baseline", index=1),
            _classified([80, 60], threshold=50, condition="baseline", index=2),
            _classified([20, 40], threshold=50, condition="baseline", index=3),
        ]
        grid = {item["multiplier"]: item for item in placebo_grid({"baseline": rows}, 50.0)}
        true = grid[1.0]["conditions"]["baseline"]["p_gap_shrunk"]["p"]
        far = grid[4.0]["conditions"]["baseline"]["p_gap_shrunk"]["p"]
        self.assertEqual(true, 1.0)
        self.assertLess(far, true)


class SideMechanicRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.run = self.runs / "synthetic_qwen"
        self.run.mkdir(parents=True)
        _write_json(self.run / "threshold.json", {"threshold": 100})
        _write_json(
            self.run / "config.json",
            {"model": "qwen3.5-122b-a10b", "model_id": "mock/qwen", "count": 6},
        )
        _write_json(self.run / "factor.json", {"motivated_reasoning_factor": 0.02})
        _write_json(
            self.run / "trajectories.json",
            {
                "baseline": [[40, 70], [160, 130], [90, 95], [2000, 2000]],
                "below_good": [[40, 80], [160, 120], [30, 60]],
                "above_good": [[40, 85], [170, 125], [200, 150]],
            },
        )
        dummy = {"rows": []}
        for name in ("baseline", "below_good", "above_good"):
            _write_json(self.run / f"{name}.json", dummy)
        _write_json(self.run / "estimates.json", {"baseline": [70, 130, 95, 2000]})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_analyze_run_drops_10x_outliers_and_reports_controls(self) -> None:
        result = analyze_run(self.run, n_perm=80, perm_seed=5)
        self.assertEqual(result["conditions"]["baseline"]["n"], 3)
        self.assertNotIn(3, [row["rollout_index"] for row in result["rollouts"]["baseline"]])
        self.assertGreater(result["conditions"]["baseline"]["p_toward_given_directional"]["p"], 0.9)
        self.assertIn("permutation_delta_early", result)
        self.assertEqual(len(result["placebo_threshold"]), 5)

    def test_run_analysis_writes_bundle_outside_runs(self) -> None:
        out = self.root / "figures" / "side_mechanics_test"
        result = run_analysis(
            runs_root=self.runs,
            output_dir=out,
            n_perm=40,
            perm_seed=9,
        )
        self.assertEqual(result["n_models"], 1)
        for name in (
            "analysis.json",
            "REPORT.md",
            "provenance.json",
            "toward_threshold_by_model.png",
            "qwen_first_last_sides.png",
            "qwen_favored_vs_toward.png",
            "qwen_placebo_threshold.png",
            "mrf_vs_toward.png",
        ):
            self.assertTrue((out / name).is_file(), name)
        with self.assertRaises(ValueError):
            run_analysis(runs_root=self.runs, output_dir=self.runs / "nested", n_perm=10)
        with self.assertRaises(FileExistsError):
            run_analysis(runs_root=self.runs, output_dir=out, n_perm=10)


if __name__ == "__main__":
    unittest.main()
