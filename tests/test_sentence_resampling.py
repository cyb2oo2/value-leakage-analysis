import json
import tempfile
import unittest
from pathlib import Path

from experiments.E03_sentence_resampling.pipeline import (
    DeterministicMockBackend,
    SourceTrajectory,
    UnsupportedContinuationCapability,
    run_pipeline,
)
from research.experiment_utils import sha256_file


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


class RecordingMockBackend(DeterministicMockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def generate_replacement(self, request):
        self.events.append(("replacement", request.request_id))
        return super().generate_replacement(request)

    def generate_continuation(self, request):
        self.events.append(("continuation", request.request_id))
        return super().generate_continuation(request)


class SentenceResamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.source_value = {
            "source_id": "source-7",
            "condition": "above_good",
            "threshold": 100,
            "task_prompt": "Exact source task prompt.",
            "sentences": [
                {"sentence_id": "S1", "text": "Population is about one hundred."},
                {"sentence_id": "S2", "text": "There may be two hundred spots each."},
                {"sentence_id": "S3", "text": "The product is twenty thousand."},
            ],
            "visible_final_answer": "20,000",
            "parsed_final_estimate": 20000,
            "metadata": {"segmentation": "manual"},
        }
        self.config_value = {
            "experiment_id": "E03-test",
            "prompt_version": "v-test",
            "target_type": "numerical_pivot",
            "semantic_arm_id": "mock_alternative",
            "continuation_mode": "visible_text_prefix_replay",
            "target_sentence_index": 2,
            "replacement_instruction": "Replace only the selected sentence.",
            "n_replacements": 2,
            "continuations_per_replacement": 3,
            "original_replay_continuations": 3,
            "require_verbatim_prefix": False,
            "require_exact_task_messages": False,
            "random_seed": 12345,
            "backend": {
                "backend": "deterministic_mock",
                "model_id": "mock/sentence-resampler-v1",
                "provider": "local",
                "settings": {"temperature": 0.0},
            },
        }
        self.source = _write(self.root / "inputs" / "source.json", self.source_value)
        self.config = _write(self.root / "inputs" / "config.json", self.config_value)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, name: str, backend=None):
        return run_pipeline(
            config_path=self.config,
            source_path=self.source,
            output_dir=self.root / "derived" / name,
            runs_root=self.runs,
            repo_root=self.root,
            backend=backend,
        )

    def test_mock_pipeline_is_deterministic_and_records_exact_intervention(self) -> None:
        source_before = sha256_file(self.source)
        config_before = sha256_file(self.config)
        first = self._run("first")
        second = self._run("second")

        self.assertEqual(first["requests"].read_bytes(), second["requests"].read_bytes())
        self.assertEqual(first["results"].read_bytes(), second["results"].read_bytes())
        requests = json.loads(first["requests"].read_text(encoding="utf-8"))
        results = json.loads(first["results"].read_text(encoding="utf-8"))
        self.assertTrue(requests["manual_segmentation_authoritative"])
        self.assertEqual(len(requests["replacement_requests"]), 2)
        self.assertEqual(len(requests["continuation_requests"]), 6)
        self.assertEqual(len(requests["original_replay_continuation_requests"]), 3)
        self.assertTrue(requests["candidate_bank"]["frozen_before_continuations"])
        self.assertFalse(requests["candidate_bank"]["continuation_outcomes_observed"])
        self.assertEqual(len(requests["candidate_bank"]["sha256"]), 64)
        schedule = requests["continuation_schedule"]
        self.assertTrue(schedule["randomized_before_continuations"])
        self.assertEqual(schedule["candidate_bank_sha256"], requests["candidate_bank"]["sha256"])
        self.assertEqual(len(schedule["request_order"]), 9)

        replacement_request = requests["replacement_requests"][0]
        self.assertEqual(replacement_request["target_sentence_index"], 2)
        self.assertEqual(replacement_request["target_sentence_id"], "S2")
        self.assertEqual(
            replacement_request["preserved_prefix"],
            [{"sentence_id": "S1", "text": "Population is about one hundred."}],
        )
        continuation_request = requests["continuation_requests"][0]
        self.assertEqual(continuation_request["task_prompt"], "Exact source task prompt.")
        self.assertEqual(continuation_request["condition"], "above_good")
        self.assertEqual(continuation_request["threshold"], 100.0)
        self.assertEqual(continuation_request["intervention_arm"], "replacement")
        self.assertIn("Population is about one hundred.", continuation_request["visible_prefix_replay"])
        self.assertNotIn("The product is twenty thousand.", continuation_request["visible_prefix_replay"])
        self.assertEqual(continuation_request["model_id"], "mock/sentence-resampler-v1")
        self.assertEqual(continuation_request["backend"], "deterministic_mock")
        self.assertEqual(continuation_request["provider"], "local")
        self.assertEqual(continuation_request["settings"], {"temperature": 0.0})
        self.assertIsInstance(continuation_request["seed"], int)
        self.assertEqual(len(continuation_request["visible_prefix_sha256"]), 64)
        self.assertEqual(
            continuation_request["prefix_construction"],
            "manual_sentence_join_with_newline_nonverbatim",
        )

        original_request = requests["original_replay_continuation_requests"][0]
        self.assertIsNone(original_request["replacement_request_id"])
        self.assertEqual(original_request["intervention_arm"], "original_replay")
        self.assertIn("There may be two hundred spots each.", original_request["visible_prefix_replay"])

        response = results["replacement_groups"][0]["continuations"][0]["response"]
        self.assertIsInstance(response["visible_final_answer"], str)
        self.assertIsInstance(response["parsed_final_estimate"], float)
        self.assertIn("replayed visible text", results["comparison"]["interpretation"])
        self.assertEqual(results["comparison"]["n_resampled_visible_final_estimates"], 6)
        self.assertEqual(results["comparison"]["n_original_replay_visible_final_estimates"], 3)
        self.assertTrue(
            results["protocol_integrity"]["candidate_bank_frozen_before_continuations"]
        )
        self.assertEqual(
            results["protocol_integrity"]["candidate_bank_sha256"],
            requests["candidate_bank"]["sha256"],
        )
        self.assertEqual(
            results["comparison"]["source_observed_answer_role"],
            "descriptive_only_not_a_stochastic_control",
        )
        self.assertEqual(sha256_file(self.source), source_before)
        self.assertEqual(sha256_file(self.config), config_before)

        manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
        self.assertFalse(manifest["manifest_self_hash_included"])
        for relative, expected_hash in manifest["files_sha256"].items():
            self.assertEqual(sha256_file(first["manifest"].parent / relative), expected_hash)

    def test_candidate_bank_precedes_randomized_continuations(self) -> None:
        backend = RecordingMockBackend()
        outputs = self._run("recorded-order", backend=backend)
        requests = json.loads(outputs["requests"].read_text(encoding="utf-8"))
        self.assertEqual(
            backend.events[:2],
            [
                ("replacement", item["request_id"])
                for item in requests["replacement_requests"]
            ],
        )
        self.assertTrue(all(event[0] == "continuation" for event in backend.events[2:]))
        self.assertEqual(
            [event[1] for event in backend.events[2:]],
            requests["continuation_schedule"]["request_order"],
        )

    def test_seed_identity_includes_source_target_and_semantic_arm(self) -> None:
        first = self._run("seed-source-a")
        first_requests = json.loads(first["requests"].read_text(encoding="utf-8"))
        changed = dict(self.source_value)
        changed["source_id"] = "source-8"
        _write(self.source, changed)
        second = self._run("seed-source-b")
        second_requests = json.loads(second["requests"].read_text(encoding="utf-8"))
        first_seeds = {
            item["seed"]
            for item in (
                first_requests["replacement_requests"]
                + first_requests["original_replay_continuation_requests"]
                + first_requests["continuation_requests"]
            )
        }
        second_seeds = {
            item["seed"]
            for item in (
                second_requests["replacement_requests"]
                + second_requests["original_replay_continuation_requests"]
                + second_requests["continuation_requests"]
            )
        }
        self.assertTrue(first_seeds.isdisjoint(second_seeds))

        _write(self.source, self.source_value)
        different_arm = dict(self.config_value)
        different_arm["semantic_arm_id"] = "counterpolicy"
        _write(self.config, different_arm)
        third = self._run("seed-arm-b")
        third_requests = json.loads(third["requests"].read_text(encoding="utf-8"))
        third_seeds = {
            item["seed"]
            for item in (
                third_requests["replacement_requests"]
                + third_requests["original_replay_continuation_requests"]
                + third_requests["continuation_requests"]
            )
        }
        self.assertTrue(first_seeds.isdisjoint(third_seeds))
        self.assertNotEqual(
            first_requests["experimental_unit_id"], third_requests["experimental_unit_id"]
        )

    def test_hidden_state_request_fails_before_sampling_or_output(self) -> None:
        hidden = dict(self.config_value)
        hidden["continuation_mode"] = "hidden_cot_internal_state_continuation"
        _write(self.config, hidden)
        backend = DeterministicMockBackend()
        output = self.root / "derived" / "must-not-exist"
        with self.assertRaisesRegex(
            UnsupportedContinuationCapability,
            "hidden-CoT/internal-state continuation is unavailable",
        ):
            run_pipeline(
                config_path=self.config,
                source_path=self.source,
                output_dir=output,
                runs_root=self.runs,
                repo_root=self.root,
                backend=backend,
            )
        self.assertEqual(backend.replacement_calls, 0)
        self.assertEqual(backend.continuation_calls, 0)
        self.assertFalse(output.exists())

    def test_refuses_overwrite_and_any_output_under_runs(self) -> None:
        self._run("once")
        with self.assertRaises(FileExistsError):
            self._run("once")
        backend = DeterministicMockBackend()
        with self.assertRaises(ValueError):
            run_pipeline(
                config_path=self.config,
                source_path=self.source,
                output_dir=self.runs / "derived",
                runs_root=self.runs,
                repo_root=self.root,
                backend=backend,
            )
        self.assertEqual(backend.replacement_calls, 0)

    def test_manual_segmentation_ids_are_authoritative(self) -> None:
        malformed = dict(self.source_value)
        malformed["sentences"] = [
            {"sentence_id": "S1", "text": "first"},
            {"sentence_id": "S3", "text": "skipped S2"},
        ]
        source = SourceTrajectory.from_dict(malformed)
        with self.assertRaisesRegex(ValueError, "expected S2"):
            source.validate()

    def test_verbatim_prefix_requirement_fails_closed_and_validates_spans(self) -> None:
        strict_config = dict(self.config_value)
        strict_config["require_verbatim_prefix"] = True
        strict_config["require_exact_task_messages"] = True
        _write(self.config, strict_config)
        with self.assertRaisesRegex(ValueError, "no visible_reasoning_text/spans"):
            self._run("missing-verbatim")

        verbatim = dict(self.source_value)
        verbatim["task_messages"] = [
            {"role": "user", "content": "Exact source task prompt."}
        ]
        verbatim["visible_reasoning_text"] = (
            "Population is about one hundred.\n"
            "There may be two hundred spots each.\n"
            "The product is twenty thousand."
        )
        verbatim["sentences"] = [
            {
                "sentence_id": "S1",
                "text": "Population is about one hundred.",
                "start_char": 0,
                "end_char": 32,
            },
            {
                "sentence_id": "S2",
                "text": "There may be two hundred spots each.",
                "start_char": 33,
                "end_char": 69,
            },
            {
                "sentence_id": "S3",
                "text": "The product is twenty thousand.",
                "start_char": 70,
                "end_char": 101,
            },
        ]
        _write(self.source, verbatim)
        outputs = self._run("verbatim")
        requests = json.loads(outputs["requests"].read_text(encoding="utf-8"))
        request = requests["original_replay_continuation_requests"][0]
        self.assertEqual(
            request["prefix_construction"],
            "verbatim_source_prefix_plus_assigned_sentence",
        )
        self.assertEqual(
            request["visible_prefix_replay"],
            "Population is about one hundred.\nThere may be two hundred spots each.",
        )


if __name__ == "__main__":
    unittest.main()
