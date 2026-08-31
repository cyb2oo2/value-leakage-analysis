from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.inspect_rollouts import (
    annotation_rows,
    load_rollout,
    select_random_ids,
    write_new,
)
from research.inspect_runs import list_runs, resolve_run


class InspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "runs" / "model-x_20260101_000000"
        self.run.mkdir(parents=True)
        (self.run / "config.json").write_text(
            json.dumps({"model": "model-x", "model_id": "org/model-x", "backend": "mock", "count": 2})
        )
        (self.run / "threshold.json").write_text(json.dumps({"threshold": 100}))
        for condition in ("baseline", "below_good", "above_good"):
            (self.run / f"{condition}.json").write_text(
                json.dumps(
                    {
                        "condition": condition,
                        "threshold": None if condition == "baseline" else 100,
                        "prompt": f"prompt {condition}",
                        "rows": [
                            {"i": 10, "reasoning": "r10", "content": "c10"},
                            {"i": 20, "reasoning": "r20", "content": "c20"},
                        ],
                    }
                )
            )
        (self.run / "estimates.json").write_text(json.dumps({"baseline": [90, 110]}))
        (self.run / "trajectories.json").write_text(
            json.dumps(
                {
                    "baseline": [[90, 110], [110, 90]],
                    "below_good": [[120, 80], [80, 70]],
                    "above_good": [[80, 120], [120, 130]],
                }
            )
        )
        (self.run / "factor.json").write_text("{}")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_inventory_and_resolution(self) -> None:
        summaries = list_runs(self.root / "runs")
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].complete)
        self.assertEqual(resolve_run(self.root / "runs", "model-x"), self.run.resolve())

    def test_missing_condition_estimate_is_not_trajectory_final(self) -> None:
        view = load_rollout(self.run, "below_good", 10)
        self.assertEqual(view.parsed_final_estimate.status, "unavailable")
        self.assertIsNone(view.parsed_final_estimate.value)
        self.assertEqual(view.successive_estimates.value, [120, 80])
        row = annotation_rows([view])[0]
        self.assertEqual(row["parsed_visible_final_estimate"], "")
        self.assertEqual(row["trajectory_last_estimate"], 80)
        self.assertEqual(row["crossed_threshold"], "true")

    def test_rollout_id_uses_row_position_for_aligned_artifacts(self) -> None:
        view = load_rollout(self.run, "baseline", 20)
        self.assertEqual(view.row_position, 1)
        self.assertEqual(view.parsed_final_estimate.value, 110)
        self.assertEqual(view.successive_estimates.value, [110, 90])

    def test_random_sampling_is_seeded(self) -> None:
        conditions = ("baseline", "below_good", "above_good")
        self.assertEqual(
            select_random_ids(self.run, conditions, 1, seed=7),
            select_random_ids(self.run, conditions, 1, seed=7),
        )

    def test_exports_refuse_overwrite_and_run_directory(self) -> None:
        out = self.root / "inspection.md"
        write_new(out, "first", self.run)
        with self.assertRaises(FileExistsError):
            write_new(out, "second", self.run)
        with self.assertRaises(ValueError):
            write_new(self.run / "derived.md", "bad", self.run)


if __name__ == "__main__":
    unittest.main()
