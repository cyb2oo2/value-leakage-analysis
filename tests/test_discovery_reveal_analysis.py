from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from experiments.E02_trajectory_localization.analyze_discovery_reveal import (
    ANALYSIS_SCHEMA_VERSION,
    REVEAL_FIELDS,
    analyze_discovery_reveal,
)
from experiments.E02_trajectory_localization.blind_discovery import (
    ANNOTATION_FIELDS,
    ANNOTATION_SCHEMA_VERSION,
)
from experiments.E02_trajectory_localization.lock_discovery_annotation import (
    LOCK_SCHEMA_VERSION,
    validate_completed_annotation,
)
from experiments.E02_trajectory_localization.reveal_discovery import (
    REVEAL_SCHEMA_VERSION,
)
from research.experiment_utils import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class DiscoveryRevealAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.runs_root = self.repo / "runs"
        self.runs_root.mkdir(parents=True)
        self.annotation = self.repo / "notes" / "discovery_annotation.csv"
        self.lock = self.repo / "notes" / "discovery_annotation_lock.json"
        self.reveal = self.repo / "derived" / "discovery_reveal"
        self.output = self.repo / "derived" / "discovery_analysis"
        self._build_fixture()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _annotation_row(self, index: int) -> dict[str, str]:
        local_index = (index - 1) % 18
        start_side = "below" if local_index < 9 else "above"
        if local_index < 6:
            revision = "up"
        elif local_index < 12:
            revision = "down"
        elif local_index < 15:
            revision = "none"
        else:
            revision = "unclear"
        row = {field: "" for field in ANNOTATION_FIELDS}
        row.update(
            {
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "blind_id": f"D{index:03d}",
                "annotation_status": "complete",
                "first_target_estimate": str(80 + index),
                "first_target_estimate_quote": f"First estimate quote {index}",
                "first_target_estimate_threshold_side": start_side,
                "population_assumption_notes": "population evidence",
                "species_mix_notes": "species evidence",
                "spots_per_giraffe_notes": "spots evidence",
                "aggregation_notes": "aggregation evidence",
                "sanity_check_revision_notes": "revision evidence",
                "numerical_pivot_present": "true",
                "numerical_pivot_component": "spots_per_giraffe",
                "numerical_pivot_quote": f"Exact pivot quote {index}",
                "numerical_pivot_before_estimate": "80",
                "numerical_pivot_after_estimate": "120",
                "revision_direction": revision,
                "target_estimate_revision_count": "1",
                "threshold_comparison_present": "true",
                "threshold_comparison_notes": "threshold evidence",
                "explicit_value_statement_present": "false",
                "explicit_value_statement_quote": "none",
                "explicit_impartiality_statement_present": "true",
                "explicit_impartiality_statement_quote": (
                    f"Exact impartiality quote {index}"
                ),
                "evaluation_awareness_present": "false",
                "evaluation_awareness_quote": "none",
                "continued_search_after_first_estimate": "true",
                "continued_search_after_value_statement": "not_applicable",
                "stopping_search_notes": "stopping evidence",
                "candidate_control_sentence_quote": f"Ordinary sentence {index}",
                "sequence_notes": "estimate -> pivot -> stop",
                "prompt_direction_disclosure_in_text": "none",
                "disclosed_favored_side": "unclear",
                "prompt_direction_disclosure_quote": "none",
                "annotator_confidence": "high",
                "notes": "",
            }
        )
        # One necessary criterion fails in each condition.
        if index == 1:
            row["candidate_control_sentence_quote"] = "none"
        if index == 19:
            row["numerical_pivot_component"] = "other"
        return row

    def _mapping_row(self, index: int) -> dict[str, Any]:
        return {
            "blind_id": f"D{index:03d}",
            "condition": "above_good" if index <= 18 else "below_good",
            "rollout_id": index,
            "row_position": index - 1,
            "reasoning_sha256": _digest(f"reasoning-{index}"),
            "visible_final_answer_sha256": _digest(f"answer-{index}"),
        }

    def _build_fixture(
        self,
        *,
        mapping_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation_rows = [self._annotation_row(index) for index in range(1, 37)]
        with self.annotation.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=ANNOTATION_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(annotation_rows)

        mapping = [self._mapping_row(index) for index in range(1, 37)]
        if mapping_mutator is not None:
            mapping_mutator(mapping)
        mapping_commitment = _canonical_sha256(mapping)
        annotation_hash = sha256_file(self.annotation)
        expected_ids = [f"D{index:03d}" for index in range(1, 37)]
        annotation_validation = validate_completed_annotation(
            self.annotation, expected_ids
        )
        lock_payload = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "annotation": str(self.annotation.resolve()),
            "annotation_sha256": annotation_hash,
            "annotation_validation": annotation_validation,
            "private_packets_parsed_by_lock": False,
            "bundle_manifest_sha256": _digest("bundle-manifest"),
            "codebook_sha256": _digest("codebook"),
            "hash_anchor_sha256": _digest("anchor"),
            "discovery_reveal_key_sha256": _digest("discovery-packet"),
            "discovery_packet_commitment_sha256": _digest("packet-commitment"),
            "discovery_mapping_commitment_sha256": mapping_commitment,
        }
        _write_json(self.lock, lock_payload)

        self.reveal.mkdir(parents=True, exist_ok=True)
        reveal_csv = self.reveal / "discovery_reveal.csv"
        with reveal_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=REVEAL_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(mapping)
        provenance = {
            "schema_version": REVEAL_SCHEMA_VERSION,
            "bundle_manifest_sha256": lock_payload["bundle_manifest_sha256"],
            "annotation_lock": str(self.lock.resolve()),
            "annotation_lock_sha256": sha256_file(self.lock),
            "annotation_sha256": annotation_hash,
            "codebook_sha256": lock_payload["codebook_sha256"],
            "hash_anchor_sha256": lock_payload["hash_anchor_sha256"],
            "discovery_packet_sha256": lock_payload[
                "discovery_reveal_key_sha256"
            ],
            "discovery_packet_commitment_sha256": lock_payload[
                "discovery_packet_commitment_sha256"
            ],
            "discovery_mapping_commitment_sha256": mapping_commitment,
            "holdout_packet_parsed": False,
            "scope": "discovery mapping only",
            "row_count": 36,
        }
        provenance_path = self.reveal / "provenance.json"
        _write_json(provenance_path, provenance)
        manifest = {
            "schema_version": REVEAL_SCHEMA_VERSION,
            "row_count": 36,
            "artifact_sha256": {
                "discovery_reveal.csv": sha256_file(reveal_csv),
                "provenance.json": sha256_file(provenance_path),
            },
            "contains_discovery_mapping": True,
            "contains_non_discovery_payload": False,
            "holdout_packet_parsed": False,
        }
        manifest_path = self.reveal / "manifest.json"
        _write_json(manifest_path, manifest)
        (self.reveal / "COMPLETE").write_text(
            f"manifest_sha256={sha256_file(manifest_path)}\n", encoding="utf-8"
        )

    def _analyze(self, output: Path | None = None) -> dict[str, Any]:
        return analyze_discovery_reveal(
            annotation=self.annotation,
            annotation_lock=self.lock,
            reveal_dir=self.reveal,
            output_dir=output or self.output,
            runs_root=self.runs_root,
            repo_root=self.repo,
        )

    def test_writes_bounded_descriptive_report_and_nonfinal_screen(self) -> None:
        result = self._analyze()
        self.assertEqual(
            result["candidate_counts_by_condition"],
            {"above_good": 17, "below_good": 17},
        )
        self.assertTrue(result["meets_minimum_six_per_condition"])
        self.assertFalse(result["candidate_screen_is_final_eligibility"])
        self.assertEqual(result["causal_claim"], "none")

        report = json.loads(
            (self.output / "discovery_analysis.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["schema_version"], ANALYSIS_SCHEMA_VERSION)
        self.assertEqual(
            report["sample"]["condition_counts"],
            {"above_good": 18, "below_good": 18},
        )
        self.assertEqual(
            set(report["frozen_annotation_field_counts"]),
            set(ANNOTATION_FIELDS),
        )
        screen = report["necessary_only_impartiality_candidate_screen"]
        self.assertEqual(screen["status"], "not_final_eligibility")
        self.assertNotIn("D001", screen["candidate_ids_by_condition"]["above_good"])
        self.assertNotIn("D019", screen["candidate_ids_by_condition"]["below_good"])

        tables = report["descriptive_tables"]
        self.assertEqual(
            tables["first_target_estimate_threshold_side"]["by_condition"]
            ["above_good"],
            {"below": 9, "equal": 0, "above": 9, "unavailable": 0},
        )
        self.assertEqual(
            tables["condition_favored_revision"]["by_condition"]["above_good"],
            {
                "condition_favored": 6,
                "condition_opposed": 6,
                "no_revision": 3,
                "indeterminate": 3,
            },
        )
        self.assertEqual(
            tables["toward_threshold_revision"]["by_condition"]["above_good"],
            {
                "toward_threshold": 9,
                "away_from_threshold": 3,
                "no_revision": 3,
                "indeterminate": 3,
            },
        )
        serialized_report = json.dumps(report)
        self.assertNotIn("Exact impartiality quote", serialized_report)
        self.assertFalse(report["integrity"]["sealed_packets_read"])
        self.assertFalse(report["integrity"]["holdout_artifacts_read"])
        self.assertFalse(report["integrity"]["raw_runs_read"])

        markdown = (self.output / "discovery_analysis.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("NOT final eligibility", markdown)
        self.assertIn("NO causal claim", markdown)
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected_hash in manifest["artifact_sha256"].items():
            self.assertEqual(sha256_file(self.output / name), expected_hash)
        self.assertEqual(
            (self.output / "COMPLETE").read_text(encoding="utf-8").strip(),
            f"manifest_sha256={sha256_file(manifest_path)}",
        )
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            self._analyze()

    def test_annotation_hash_tamper_fails_without_output(self) -> None:
        self.annotation.write_text(
            self.annotation.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "annotation hash"):
            self._analyze()
        self.assertFalse(self.output.exists())

    def test_reveal_csv_hash_tamper_fails_without_output(self) -> None:
        reveal_csv = self.reveal / "discovery_reveal.csv"
        reveal_csv.write_text(
            reveal_csv.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "reveal artifact hash"):
            self._analyze()
        self.assertFalse(self.output.exists())

    def test_exact_join_and_balance_are_fail_closed_after_valid_hashes(self) -> None:
        for case, mutate, expected_error in (
            (
                "duplicate_id",
                lambda mapping: mapping[-1].__setitem__("blind_id", "D035"),
                "1:1 ordered D001-D036",
            ),
            (
                "unbalanced",
                lambda mapping: mapping[18].__setitem__("condition", "above_good"),
                "exactly 18 rows per condition",
            ),
        ):
            with self.subTest(case=case):
                self.reveal = self.repo / "derived" / f"reveal_{case}"
                self.lock = self.repo / "notes" / f"lock_{case}.json"
                self.output = self.repo / "derived" / f"analysis_{case}"
                self._build_fixture(mapping_mutator=mutate)
                with self.assertRaisesRegex(ValueError, expected_error):
                    self._analyze()
                self.assertFalse(self.output.exists())

    def test_forbidden_raw_or_holdout_scoped_source_is_rejected_before_read(self) -> None:
        raw_reveal = self.runs_root / "mock_run" / "discovery_reveal"
        with self.assertRaisesRegex(ValueError, "must not be read from immutable raw runs"):
            analyze_discovery_reveal(
                annotation=self.annotation,
                annotation_lock=self.lock,
                reveal_dir=raw_reveal,
                output_dir=self.repo / "derived" / "raw_rejected",
                runs_root=self.runs_root,
                repo_root=self.repo,
            )
        holdout_named = self.repo / "derived" / "holdout_reveal"
        with self.assertRaisesRegex(ValueError, "sealed/holdout-scoped"):
            analyze_discovery_reveal(
                annotation=self.annotation,
                annotation_lock=self.lock,
                reveal_dir=holdout_named,
                output_dir=self.repo / "derived" / "holdout_rejected",
                runs_root=self.runs_root,
                repo_root=self.repo,
            )


if __name__ == "__main__":
    unittest.main()
