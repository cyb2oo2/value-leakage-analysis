from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiments.E02_trajectory_localization.lock_discovery_annotation as lock_module
import experiments.E02_trajectory_localization.reveal_discovery as reveal_module
from experiments.E02_trajectory_localization.blind_discovery import (
    ANNOTATION_FIELDS,
    ANNOTATION_SCHEMA_VERSION,
    create_bundle,
)
from experiments.E02_trajectory_localization.lock_discovery_annotation import (
    lock_annotation,
)
from experiments.E02_trajectory_localization.reveal_discovery import (
    reveal_discovery,
)
from research.experiment_utils import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DiscoveryGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.runs_root = self.repo / "runs"
        self.run = self.runs_root / "mock_run"
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
        for condition in ("above_good", "below_good"):
            rows = [
                {
                    "i": index,
                    "reasoning": f"Reasoning {condition} {index}.",
                    "content": f"Answer {index}.",
                    "finish_reason": "stop",
                }
                for index in range(5)
            ]
            _write_json(
                self.run / f"{condition}.json",
                {
                    "condition": condition,
                    "prompt": f"Private prompt {condition}",
                    "rows": rows,
                },
            )
        self.bundle = self.repo / "derived" / "bundle"
        self.annotation = self.repo / "notes" / "annotation.csv"
        self.private = self.repo / "private" / "keys"
        create_bundle(
            run_dir=self.run,
            output_dir=self.bundle,
            annotation_output=self.annotation,
            sealed_output_dir=self.private,
            runs_root=self.runs_root,
            discovery_seed=111,
            validation_seed=222,
            namespace="gate-test-v1",
            discovery_per_condition=2,
            validation_per_condition=2,
            repo_root=self.repo,
        )
        self.discovery_key = self.private / "discovery_reveal_key.json"
        self.holdout_key = (
            self.private / "HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json"
        )
        self.codebook = self.repo / "notes" / "codebook.md"
        self.codebook.write_text("frozen codebook\n", encoding="utf-8")
        self.anchor = self.repo / "notes" / "private_manifest.json"
        self._refresh_anchor()
        self.lock = self.repo / "notes" / "annotation.lock.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _fill_annotation(self, *, confidence: str = "high") -> None:
        with self.annotation.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        defaults = {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "annotation_status": "complete",
            "first_target_estimate": "99",
            "first_target_estimate_quote": "99",
            "first_target_estimate_threshold_side": "below",
            "population_assumption_notes": "none",
            "species_mix_notes": "none",
            "spots_per_giraffe_notes": "none",
            "aggregation_notes": "none",
            "sanity_check_revision_notes": "none",
            "numerical_pivot_present": "false",
            "numerical_pivot_component": "none",
            "numerical_pivot_quote": "none",
            "numerical_pivot_before_estimate": "none",
            "numerical_pivot_after_estimate": "none",
            "revision_direction": "none",
            "target_estimate_revision_count": "0",
            "threshold_comparison_present": "false",
            "threshold_comparison_notes": "none",
            "explicit_value_statement_present": "false",
            "explicit_value_statement_quote": "none",
            "explicit_impartiality_statement_present": "false",
            "explicit_impartiality_statement_quote": "none",
            "evaluation_awareness_present": "false",
            "evaluation_awareness_quote": "none",
            "continued_search_after_first_estimate": "false",
            "continued_search_after_value_statement": "not_applicable",
            "stopping_search_notes": "stopped",
            "candidate_control_sentence_quote": "none",
            "sequence_notes": "estimate -> stop",
            "prompt_direction_disclosure_in_text": "none",
            "disclosed_favored_side": "unclear",
            "prompt_direction_disclosure_quote": "none",
            "annotator_confidence": confidence,
            "notes": "",
        }
        for row in rows:
            blind_id = row["blind_id"]
            row.update(defaults)
            row["blind_id"] = blind_id
        with self.annotation.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=ANNOTATION_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def _refresh_anchor(self) -> None:
        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _write_json(
            self.anchor,
            {
                "schema_version": "value-leakage.private-key-hash-anchor/v2",
                "study_id": "fixture",
                "public_bundle": self.bundle.relative_to(self.repo).as_posix(),
                "annotation": self.annotation.relative_to(self.repo).as_posix(),
                "discovery_total": manifest["discovery_total"],
                "holdout_total": manifest["holdout_rollout_total"],
                "reserve_total": manifest["reserve_total"],
                "bundle_manifest_sha256": sha256_file(manifest_path),
                "discovery_mapping_commitment_sha256": manifest[
                    "discovery_mapping_commitment_sha256"
                ],
                "validation_mapping_commitment_sha256": manifest[
                    "validation_mapping_commitment_sha256"
                ],
                "discovery_packet_commitment_sha256": manifest[
                    "discovery_packet_commitment_sha256"
                ],
                "holdout_packet_commitment_sha256": manifest[
                    "holdout_packet_commitment_sha256"
                ],
                "discovery_reveal_key_sha256": manifest[
                    "discovery_reveal_key_sha256"
                ],
                "holdout_reveal_key_sha256": manifest[
                    "holdout_reveal_key_sha256"
                ],
                "annotation_template_sha256": manifest[
                    "annotation_template_sha256"
                ],
                "frozen_documents_sha256": {
                    self.codebook.relative_to(self.repo).as_posix(): sha256_file(
                        self.codebook
                    )
                },
                "private_keys_in_public_bundle": False,
                "warning": "fixture",
            },
        )

    def _lock(self) -> dict[str, object]:
        return lock_annotation(
            bundle=self.bundle,
            annotation=self.annotation,
            codebook=self.codebook,
            anchor_manifest=self.anchor,
            discovery_key=self.discovery_key,
            holdout_key=self.holdout_key,
            output=self.lock,
            runs_root=self.runs_root,
            repo_root=self.repo,
        )

    def _reveal(self, output: Path, *, confirm: bool = True) -> dict[str, object]:
        return reveal_discovery(
            bundle=self.bundle,
            annotation=self.annotation,
            codebook=self.codebook,
            anchor_manifest=self.anchor,
            annotation_lock=self.lock,
            discovery_key=self.discovery_key,
            holdout_key=self.holdout_key,
            output_dir=output,
            runs_root=self.runs_root,
            repo_root=self.repo,
            confirm_annotations_locked=confirm,
        )

    def test_blank_annotation_fails_before_any_private_json_parse(self) -> None:
        discovery_text = self.discovery_key.read_text(encoding="utf-8")
        holdout_text = self.holdout_key.read_text(encoding="utf-8")
        original_loads = json.loads

        def guarded_loads(value: object, *args: object, **kwargs: object) -> object:
            if value in {discovery_text, holdout_text}:
                raise AssertionError("private packet was parsed")
            return original_loads(value, *args, **kwargs)

        with patch.object(lock_module.json, "loads", side_effect=guarded_loads):
            with self.assertRaisesRegex(ValueError, "annotation is not lockable"):
                self._lock()
        self.assertFalse(self.lock.exists())

    def test_invalid_category_and_blank_required_field_fail_closed(self) -> None:
        self._fill_annotation(confidence="certain")
        with self.annotation.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["sequence_notes"] = ""
        with self.annotation.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "annotation is not lockable"):
            self._lock()
        self.assertFalse(self.lock.exists())

    def test_valid_lock_and_reveal_never_parse_holdout(self) -> None:
        self._fill_annotation()
        locked = self._lock()
        self.assertTrue(locked["ready_for_discovery_reveal"])
        lock_payload = json.loads(self.lock.read_text(encoding="utf-8"))
        self.assertFalse(lock_payload["private_packets_parsed_by_lock"])

        holdout_text = self.holdout_key.read_text(encoding="utf-8")
        original_loads = json.loads

        def guarded_loads(value: object, *args: object, **kwargs: object) -> object:
            if value == holdout_text:
                raise AssertionError("holdout packet was parsed")
            return original_loads(value, *args, **kwargs)

        output = self.repo / "derived" / "reveal"
        with patch.object(reveal_module.json, "loads", side_effect=guarded_loads):
            result = self._reveal(output)
        self.assertFalse(result["holdout_packet_parsed"])
        self.assertTrue((output / "COMPLETE").is_file())
        output_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output.rglob("*")
            if path.is_file()
        )
        self.assertNotIn("validation_seed", output_text)
        self.assertNotIn("validation_mapping", output_text)
        self.assertNotIn("reserve_mapping", output_text)
        self.assertNotIn("V001", output_text)
        self.assertIn("condition", (output / "discovery_reveal.csv").read_text())

    def test_confirmation_and_post_lock_tamper_fail_without_output(self) -> None:
        self._fill_annotation()
        self._lock()
        no_confirm = self.repo / "derived" / "no_confirm"
        with self.assertRaisesRegex(ValueError, "explicit"):
            self._reveal(no_confirm, confirm=False)
        self.assertFalse(no_confirm.exists())

        self.annotation.write_text(
            self.annotation.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        tampered = self.repo / "derived" / "tampered"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self._reveal(tampered)
        self.assertFalse(tampered.exists())

    def test_semantically_malicious_discovery_packet_passes_opaque_lock_but_not_reveal(self) -> None:
        packet = json.loads(self.discovery_key.read_text(encoding="utf-8"))
        packet["validation_mapping"] = []
        packet_core = dict(packet)
        packet_core.pop("packet_commitment_sha256")
        packet["packet_commitment_sha256"] = _canonical_sha256(packet_core)
        _write_json(self.discovery_key, packet)

        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["discovery_reveal_key_sha256"] = sha256_file(self.discovery_key)
        manifest["discovery_packet_commitment_sha256"] = packet[
            "packet_commitment_sha256"
        ]
        _write_json(manifest_path, manifest)
        (self.bundle / "COMPLETE").write_text(
            f"manifest_sha256={sha256_file(manifest_path)}\n",
            encoding="utf-8",
        )
        self._refresh_anchor()

        self._fill_annotation()
        self._lock()
        output = self.repo / "derived" / "malicious"
        with self.assertRaisesRegex(ValueError, "fields are not exact|holdout/reserve"):
            self._reveal(output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
