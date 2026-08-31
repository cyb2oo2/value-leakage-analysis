from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.experiment_utils import (
    build_provenance,
    create_new_directory,
    ensure_output_outside_raw,
    seed_everything,
    sha256_text,
    write_new_text,
)


class ExperimentUtilsTests(unittest.TestCase):
    def test_seed_is_reproducible(self) -> None:
        first = seed_everything(17).integers(0, 1_000_000, size=5).tolist()
        second = seed_everything(17).integers(0, 1_000_000, size=5).tolist()
        self.assertEqual(first, second)

    def test_output_cannot_be_inside_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "runs"
            raw.mkdir()
            with self.assertRaises(ValueError):
                ensure_output_outside_raw(raw / "run-a" / "figure.png", [raw])

    def test_create_and_write_refuse_existing_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "derived" / "E01"
            create_new_directory(output)
            with self.assertRaises(FileExistsError):
                create_new_directory(output)
            target = output / "config.json"
            write_new_text(target, "first")
            with self.assertRaises(FileExistsError):
                write_new_text(target, "second")

    def test_provenance_hashes_exact_prompts(self) -> None:
        provenance = build_provenance(
            experiment_id="E00-test",
            model_id="mock/model",
            backend="mock",
            provider=None,
            prompt_version="v1",
            prompts={"baseline": "exact prompt"},
            n_requested=2,
            sampling_settings={"temperature": 0.0},
            random_seed=11,
            raw_response_directory="derived/E00/raw",
            judge_model=None,
            figure_script="research.trajectory_analysis",
            repo_root=Path(__file__).parents[1],
        )
        self.assertEqual(provenance.prompt_sha256["baseline"], sha256_text("exact prompt"))
        self.assertEqual(provenance.random_seed, 11)
        self.assertIsNotNone(provenance.code_commit)


if __name__ == "__main__":
    unittest.main()
