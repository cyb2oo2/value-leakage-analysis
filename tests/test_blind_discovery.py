from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.E02_trajectory_localization.blind_discovery import (
    ANNOTATION_FIELDS,
    _canonical_json,
    allocate_split,
    create_bundle,
    load_eligible_records,
)
from experiments.E02_trajectory_localization.audit_blind_bundle import (
    audit_bundle,
    audit_public_bundle,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BlindDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.run = self.repo / "runs" / "mock_run"
        self.run.mkdir(parents=True)
        _write_json(
            self.run / "config.json",
            {
                "model": "mock-model",
                "model_id": "mock/model",
                "backend": "offline",
                "provider": "fixture",
            },
        )
        _write_json(self.run / "threshold.json", {"threshold": 100})
        words = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
        for condition, prompt in (
            ("above_good", "SECRET PROMPT ALPHA"),
            ("below_good", "SECRET PROMPT BRAVO"),
        ):
            rows = []
            for index, word in enumerate(words):
                if index == 5:
                    rows.append({"i": index, "error": "synthetic failure"})
                else:
                    rows.append(
                        {
                            "i": index,
                            "reasoning": f"Verbatim reasoning {word}.",
                            "content": f"Visible answer {word}.",
                            "finish_reason": "stop",
                        }
                    )
            _write_json(
                self.run / f"{condition}.json",
                {"condition": condition, "prompt": prompt, "rows": rows},
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _create(self, suffix: str = "one") -> tuple[Path, Path, dict[str, object]]:
        output = self.repo / "derived" / f"bundle_{suffix}"
        annotation = self.repo / "notes" / f"annotation_{suffix}.csv"
        sealed = self.repo / "private" / f"keys_{suffix}"
        result = create_bundle(
            run_dir=self.run,
            output_dir=output,
            annotation_output=annotation,
            sealed_output_dir=sealed,
            runs_root=self.repo / "runs",
            discovery_seed=111,
            validation_seed=222,
            namespace="test-v1",
            discovery_per_condition=2,
            validation_per_condition=2,
            repo_root=self.repo,
        )
        return output, annotation, result

    def test_eligibility_uses_raw_success_not_downstream_judge(self) -> None:
        records, audit = load_eligible_records(self.run)
        self.assertEqual(len(records["above_good"]), 5)
        self.assertEqual(len(records["below_good"]), 5)
        self.assertEqual(audit["conditions"]["above_good"]["eligible"], 5)
        self.assertFalse((self.run / "trajectories.json").exists())

    def test_duplicate_raw_ids_fail_closed(self) -> None:
        path = self.run / "above_good.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][1]["i"] = payload["rows"][0]["i"]
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "duplicate rollout IDs"):
            load_eligible_records(self.run)

    def test_allocation_is_balanced_disjoint_complete_and_deterministic(self) -> None:
        records, _ = load_eligible_records(self.run)
        first = allocate_split(
            records,
            discovery_seed=111,
            validation_seed=222,
            namespace="test-v1",
            discovery_per_condition=2,
            validation_per_condition=2,
        )
        second = allocate_split(
            records,
            discovery_seed=111,
            validation_seed=222,
            namespace="test-v1",
            discovery_per_condition=2,
            validation_per_condition=2,
        )
        self.assertEqual(first, second)
        discovery, validation, reserve = first
        d_keys = {item.raw.canonical_key for item in discovery}
        v_keys = {item.raw.canonical_key for item in validation}
        r_keys = {item.canonical_key for item in reserve}
        self.assertFalse(d_keys & v_keys or d_keys & r_keys or v_keys & r_keys)
        self.assertEqual(len(d_keys | v_keys | r_keys), 10)
        self.assertEqual(
            {condition: sum(item.raw.condition == condition for item in discovery)
             for condition in ("above_good", "below_good")},
            {"above_good": 2, "below_good": 2},
        )
        with self.assertRaisesRegex(ValueError, "must be different"):
            allocate_split(
                records,
                discovery_seed=1,
                validation_seed=1,
                namespace="test-v1",
                discovery_per_condition=2,
                validation_per_condition=2,
            )

    def test_public_bundle_hides_mapping_prompt_and_seed(self) -> None:
        source_hashes_before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.run.glob("*.json")
        }
        output, annotation, result = self._create()
        self.assertEqual(result["discovery_total"], 4)
        self.assertEqual(result["validation_total_precommitted"], 4)
        self.assertTrue((output / "COMPLETE").is_file())
        self.assertEqual(
            sorted(path.name for path in (output / "rollouts").glob("*.txt")),
            ["D001.txt", "D002.txt", "D003.txt", "D004.txt"],
        )
        self.assertEqual(
            sorted(path.name for path in (output / "holdout_rollouts").glob("*.txt")),
            ["V001.txt", "V002.txt", "V003.txt", "V004.txt"],
        )

        public_paths = [path for path in output.rglob("*") if path.is_file()]
        public_text = "\n".join(path.read_text(encoding="utf-8") for path in public_paths)
        self.assertNotIn("above_good", public_text)
        self.assertNotIn("below_good", public_text)
        self.assertNotIn("SECRET PROMPT", public_text)
        self.assertNotIn('"discovery_seed"', public_text)
        self.assertNotIn('"validation_seed"', public_text)
        self.assertIn("VERBATIM", public_text)
        self.assertFalse((output / "sealed").exists())
        self.assertFalse(any(path.name == "discovery_reveal_key.json" for path in output.rglob("*")))
        self.assertTrue(Path(str(result["private_key_directory"])).is_dir())

        with annotation.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(tuple(rows[0]), ANNOTATION_FIELDS)
        self.assertEqual([row["blind_id"] for row in rows], ["D001", "D002", "D003", "D004"])
        self.assertNotIn("condition", rows[0])
        self.assertNotIn("good_direction", rows[0])

        source_hashes_after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.run.glob("*.json")
        }
        self.assertEqual(source_hashes_before, source_hashes_after)

    def test_separate_reveal_packets_bind_mappings_without_cross_leakage(self) -> None:
        output, _, result = self._create()
        private = Path(str(result["private_key_directory"]))
        discovery_path = private / "discovery_reveal_key.json"
        holdout_path = private / "HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json"
        self.assertFalse(discovery_path.is_relative_to(output))
        self.assertFalse(holdout_path.is_relative_to(output))
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        self.assertIn("discovery_mapping", discovery)
        self.assertNotIn("validation_mapping", discovery)
        self.assertIn("discovery_seed", discovery["selection"])
        self.assertNotIn("validation_seed", discovery["selection"])
        self.assertIn("validation_mapping", holdout)
        self.assertNotIn("discovery_mapping", holdout)
        self.assertIn("validation_seed", holdout["selection"])
        self.assertNotIn("discovery_seed", holdout["selection"])
        discovery_digest = hashlib.sha256(
            _canonical_json(discovery["discovery_mapping"]).encode("utf-8")
        ).hexdigest()
        holdout_digest = hashlib.sha256(
            _canonical_json(holdout["validation_mapping"]).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            discovery_digest,
            discovery["discovery_mapping_commitment_sha256"],
        )
        self.assertEqual(
            holdout_digest,
            holdout["validation_mapping_commitment_sha256"],
        )

    def test_safe_auditor_reports_only_aggregate_integrity(self) -> None:
        output, annotation, result = self._create()
        report = audit_public_bundle(
            output,
            annotation,
            runs_root=self.repo / "runs",
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts"]["discovery_total"], 4)
        self.assertEqual(report["counts"]["withheld_holdout_total"], 4)
        self.assertFalse(report["private_packets_parsed"])
        serialized = json.dumps(report)
        self.assertNotIn("above_good", serialized)
        self.assertNotIn("below_good", serialized)
        self.assertNotIn("rollout_id", serialized)
        self.assertNotIn("seed", serialized)

        private = Path(str(result["private_key_directory"]))
        privileged = audit_bundle(
            output,
            annotation,
            discovery_key=private / "discovery_reveal_key.json",
            holdout_key=private / "HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json",
            runs_root=self.repo / "runs",
        )
        self.assertTrue(privileged["ok"])

    def test_refuses_overwrite_and_any_output_under_runs(self) -> None:
        output, annotation, _ = self._create()
        with self.assertRaises(FileExistsError):
            create_bundle(
                run_dir=self.run,
                output_dir=output,
                annotation_output=self.repo / "notes" / "new.csv",
                sealed_output_dir=self.repo / "private" / "new_keys",
                runs_root=self.repo / "runs",
                discovery_seed=3,
                validation_seed=4,
                namespace="test-v1",
                discovery_per_condition=2,
                validation_per_condition=2,
                repo_root=self.repo,
            )
        with self.assertRaises(ValueError):
            create_bundle(
                run_dir=self.run,
                output_dir=self.run / "derived",
                annotation_output=self.repo / "notes" / "other.csv",
                sealed_output_dir=self.repo / "private" / "other_keys",
                runs_root=self.repo / "runs",
                discovery_seed=3,
                validation_seed=4,
                namespace="test-v1",
                discovery_per_condition=2,
                validation_per_condition=2,
                repo_root=self.repo,
            )
        self.assertTrue(annotation.exists())

    def test_wrong_runs_root_cannot_bypass_raw_write_protection(self) -> None:
        wrong_root = self.repo / "other_runs"
        wrong_root.mkdir()
        with self.assertRaisesRegex(ValueError, "not inside runs root"):
            create_bundle(
                run_dir=self.run,
                output_dir=self.run / "derived",
                annotation_output=self.repo / "notes" / "wrong_root.csv",
                sealed_output_dir=self.repo / "private" / "wrong_root",
                runs_root=wrong_root,
                discovery_seed=3,
                validation_seed=4,
                namespace="test-v1",
                discovery_per_condition=2,
                validation_per_condition=2,
                repo_root=self.repo,
            )
        self.assertFalse((self.run / "derived").exists())


if __name__ == "__main__":
    unittest.main()
