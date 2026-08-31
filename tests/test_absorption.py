from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.absorption import (
    classify_absorption,
    escaped_after_hit,
    on_favored_side,
    parse_visible_final,
    run_analysis,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class ParserTests(unittest.TestCase):
    def test_plain_and_comma_and_million(self) -> None:
        value, status, _ = parse_visible_final("45,000,000\n\nJustification")
        self.assertEqual(status, "ok")
        self.assertEqual(value, 45_000_000)
        value, status, _ = parse_visible_final("**39 million**")
        self.assertEqual(status, "ok")
        self.assertEqual(value, 39_000_000)
        value, status, _ = parse_visible_final("32000000")
        self.assertEqual(value, 32_000_000)

    def test_ranges_and_prose_fail_closed(self) -> None:
        self.assertIn(parse_visible_final("between 30 and 40 million")[1], {"range", "not_single_number_line"})
        self.assertEqual(parse_visible_final("I think the answer is 45,000,000")[1], "not_single_number_line")
        self.assertEqual(parse_visible_final("")[1], "empty")


class AbsorptionUnitTests(unittest.TestCase):
    def test_favored_side_matches_prompt_inequality(self) -> None:
        self.assertTrue(on_favored_side("above_good", "above"))
        self.assertFalse(on_favored_side("above_good", "equal"))
        self.assertTrue(on_favored_side("below_good", "equal"))
        self.assertIsNone(on_favored_side("baseline", "above"))

    def test_conversion_and_escape(self) -> None:
        converted = classify_absorption([20, 30, 80], 50, "above_good", rollout_index=0)
        self.assertTrue(converted.converted)
        self.assertFalse(converted.leaked)
        self.assertFalse(converted.escaped_after_hit)
        leaked = classify_absorption([80, 70, 20], 50, "above_good", rollout_index=1)
        self.assertTrue(leaked.leaked)
        self.assertTrue(escaped_after_hit([80, 90, 20], 50, "above_good"))
        self.assertFalse(escaped_after_hit([20, 80, 90], 50, "above_good"))


class AbsorptionRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.run = self.runs / "qwen3.5-122b-a10b_synthetic"
        self.run.mkdir(parents=True)
        _write_json(self.run / "threshold.json", {"threshold": 100})
        _write_json(self.run / "config.json", {"model": "qwen3.5-122b-a10b", "count": 4})
        _write_json(self.run / "factor.json", {"motivated_reasoning_factor": 0.01})
        _write_json(
            self.run / "trajectories.json",
            {
                "baseline": [[40, 60], [160, 140]],
                "below_good": [[150, 80], [40, 30]],
                "above_good": [[40, 120], [150, 140]],
            },
        )
        _write_json(
            self.run / "estimates.json",
            {"baseline": [60, 140]},
        )
        for name, contents in (
            ("baseline", ["60\n", "140\n"]),
            ("below_good", ["80\n", "30\n"]),
            ("above_good", ["120\n", "I cannot say"]),
        ):
            _write_json(
                self.run / f"{name}.json",
                {"rows": [{"i": i, "content": text} for i, text in enumerate(contents)]},
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_run_writes_bundle(self) -> None:
        out = self.root / "figures" / "absorption_test"
        result = run_analysis(runs_root=self.runs, output_dir=out, n_perm=40, perm_seed=1)
        self.assertEqual(result["model"], "qwen3.5-122b-a10b")
        analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))["result"]
        self.assertEqual(analysis["conditions"]["below_good"]["p_convert_given_start_opposed"]["k"], 1)
        self.assertEqual(analysis["visible"]["above_good"]["n_parsed"], 1)
        self.assertTrue((out / "convert_vs_leak.png").is_file())


if __name__ == "__main__":
    unittest.main()
