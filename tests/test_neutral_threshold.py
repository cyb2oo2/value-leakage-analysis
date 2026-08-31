from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.E01_neutral_threshold.pilot import (
    DEFAULT_PROMPT_SPEC,
    MockBackend,
    SamplingBackend,
    load_prompt_spec,
    resolve_prompts,
    run_pilot,
)


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "E01_neutral_threshold" / "configs" / "mock_pilot.json"
)


class PromptDesignTests(unittest.TestCase):
    def test_variants_preserve_question_and_same_threshold(self) -> None:
        spec = load_prompt_spec(DEFAULT_PROMPT_SPEC)
        prompts = resolve_prompts(
            spec,
            123456789,
            ("above_good", "below_good", "neutral_equal_good"),
        )
        question = spec.task["question"]
        for prompt in prompts.values():
            self.assertIn(question, prompt)
            self.assertEqual(prompt.count("123,456,789"), 1)
            self.assertIn("The threshold is 123,456,789.", prompt)
            self.assertNotIn("{threshold}", prompt)
        self.assertIn("good cause", prompts["above_good"])
        self.assertIn("bad cause", prompts["above_good"])
        self.assertIn("good cause", prompts["below_good"])
        self.assertIn("bad cause", prompts["below_good"])

    def test_neutral_variant_equalizes_amount_and_stated_valence(self) -> None:
        spec = load_prompt_spec(DEFAULT_PROMPT_SPEC)
        prompt = resolve_prompts(spec, 100, ("neutral_equal_good",))[
            "neutral_equal_good"
        ]
        self.assertIn("donate $100 to preselected Cause A", prompt)
        self.assertIn("donate the same $100 to preselected Cause B", prompt)
        self.assertIn("selected in advance", prompt)
        self.assertIn("equally beneficial", prompt)
        neutral_design = spec.design["neutral_control"]
        self.assertEqual(neutral_design["donation_amount_usd_each_outcome"], 100)
        self.assertIsNone(neutral_design["directional_preference"])
        self.assertIsNone(spec.variants["neutral_equal_good"]["preferred_side"])

    def test_neutral_prompt_remains_close_in_length_and_structure(self) -> None:
        spec = load_prompt_spec(DEFAULT_PROMPT_SPEC)
        prompts = resolve_prompts(spec, 100)
        neutral_words = len(prompts["neutral_equal_good"].split())
        original_words = [
            len(prompts[condition].split())
            for condition in ("above_good", "below_good")
        ]
        self.assertLessEqual(neutral_words / (sum(original_words) / 2), 1.35)
        for prompt in prompts.values():
            self.assertEqual(len(prompt.split("\n\n")), 4)


class NeutralPilotTests(unittest.TestCase):
    def _write_config(
        self,
        root: Path,
        *,
        output_directory: str,
        backend: str = "mock",
        provider: str | None = "local-deterministic-mock",
    ) -> Path:
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["prompt_spec"] = str(DEFAULT_PROMPT_SPEC.resolve())
        config["output_directory"] = output_directory
        config["backend"] = backend
        config["provider"] = provider
        config["n"] = 3
        config["seed"] = 71
        path = root / f"config-{backend}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_mock_backend_satisfies_protocol(self) -> None:
        self.assertIsInstance(MockBackend(), SamplingBackend)

    def test_mock_smoke_is_deterministic_and_writes_complete_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_config(root, output_directory="unused")
            first = run_pilot(config, output_directory=root / "out-one", repo_root=root)
            second = run_pilot(config, output_directory=root / "out-two", repo_root=root)
            self.assertEqual(len(first.raw_response_files), 3)
            self.assertEqual(len(second.raw_response_files), 3)
            for first_file, second_file in zip(
                first.raw_response_files, second.raw_response_files, strict=True
            ):
                self.assertEqual(
                    first_file.read_text(encoding="utf-8"),
                    second_file.read_text(encoding="utf-8"),
                )
                raw = json.loads(first_file.read_text(encoding="utf-8"))
                self.assertEqual(raw["schema_version"], "value-leakage.raw-sampling/v1")
                self.assertEqual(raw["n_requested"], 3)
                self.assertEqual(len(raw["rows"]), 3)
                self.assertEqual(raw["backend"], "mock")
                self.assertIn("prompt", raw)
                self.assertIn("sampling_settings", raw)
            effective = json.loads(first.effective_config.read_text(encoding="utf-8"))
            for field in (
                "model_id",
                "backend",
                "provider",
                "n",
                "temperature",
                "top_p",
                "reasoning",
                "max_tokens",
                "seed",
                "judge",
                "output_directory",
                "prompt_version",
                "raw_response_directory",
            ):
                self.assertIn(field, effective)
            provenance = json.loads(first.provenance.read_text(encoding="utf-8"))
            self.assertEqual(provenance["n_per_condition"], 3)
            self.assertEqual(provenance["backend"], "mock")
            self.assertEqual(len(provenance["prompt_sha256"]), 3)
            self.assertTrue(first.manifest.is_file())

    def test_output_must_be_new_and_outside_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "runs").mkdir()
            config = self._write_config(root, output_directory="unused")
            output = root / "derived" / "pilot"
            run_pilot(config, output_directory=output, repo_root=root)
            with self.assertRaises(FileExistsError):
                run_pilot(config, output_directory=output, repo_root=root)
            with self.assertRaisesRegex(ValueError, "immutable raw root"):
                run_pilot(
                    config,
                    output_directory=root / "runs" / "forbidden",
                    repo_root=root,
                )
            self.assertFalse((root / "runs" / "forbidden").exists())

    def test_real_backends_are_disabled_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_config(
                root,
                output_directory="unused",
                backend="openrouter",
                provider="some-provider",
            )
            output = root / "should-not-exist"
            with self.assertRaisesRegex(NotImplementedError, "cannot make real API calls"):
                run_pilot(config, output_directory=output, repo_root=root)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
