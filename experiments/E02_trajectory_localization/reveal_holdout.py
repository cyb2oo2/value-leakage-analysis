"""Reveal the locked V-ID holdout mapping exactly once.

Every public, annotation-lock, code, and output-path check runs before the
private packet is read.  The tool then creates an exclusive permanent start
receipt, reads the private bytes once, validates the precommitted split, and
emits only the V mapping and a deterministic eligibility gate.  Reserve rows
inside the legacy v2 packet are checked for split integrity but never emitted
or used to rescue the confirmatory sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.E02_trajectory_localization.blind_discovery import (
    BlindRecord,
    CONDITIONS,
    SCHEMA_VERSION,
    _mapping,
    _reserve_mapping,
    _score,
    load_eligible_records,
    render_rollout,
)
from experiments.E02_trajectory_localization.lock_holdout_annotations import (
    LOCK_SCHEMA_VERSION,
)
from experiments.E02_trajectory_localization.prepare_holdout_templates import (
    public_holdout_ids_from_manifest_and_filenames,
)
from research.experiment_utils import (
    ensure_output_outside_raw,
    sha256_file,
    sha256_text,
    utc_now,
    write_new_json,
    write_new_text,
)


REVEAL_SCHEMA_VERSION = "value-leakage.holdout-reveal/v1"
ELIGIBILITY_GATE_SCHEMA_VERSION = "value-leakage.holdout-eligibility-gate/v1"
RECEIPT_SCHEMA_VERSION = "value-leakage.holdout-reveal-once-receipt/v1"
HOLDOUT_REVEAL_CSV_FIELDS = (
    "blind_id",
    "condition",
    "rollout_id",
    "row_position",
    "reasoning_sha256",
    "visible_final_answer_sha256",
)
_MAPPING_FIELDS = frozenset(HOLDOUT_REVEAL_CSV_FIELDS)
_RESERVE_FIELDS = frozenset(
    {
        "condition",
        "rollout_id",
        "row_position",
        "reasoning_sha256",
        "visible_final_answer_sha256",
    }
)
_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "warning",
        "created_at_utc",
        "packet_nonce",
        "source_run",
        "source_files_sha256",
        "selection",
        "discovery_mapping_commitment_sha256",
        "validation_mapping",
        "validation_mapping_commitment_sha256",
        "reserve_mapping",
        "reserve_order",
        "packet_commitment_sha256",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "algorithm",
        "namespace",
        "validation_seed",
        "validation_per_condition",
    }
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _render_mapping_csv(mapping: Sequence[Mapping[str, Any]]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=HOLDOUT_REVEAL_CSV_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in mapping:
        writer.writerow({field: row[field] for field in HOLDOUT_REVEAL_CSV_FIELDS})
    return stream.getvalue()


def _validate_public_lock_state(
    *,
    bundle: Path,
    annotation_lock: Path,
    holdout_key: Path,
    runs_root: Path,
    expected_total: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    root = bundle.resolve()
    lock_path = annotation_lock.resolve()
    key_path = holdout_key.resolve()
    raw_root = runs_root.resolve()
    blind_ids, manifest = public_holdout_ids_from_manifest_and_filenames(
        root,
        expected_total=expected_total,
    )
    lock = _read_json(lock_path)
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError("holdout annotation lock has the wrong schema")
    if lock.get("private_packet_parsed_by_lock") is not False:
        raise ValueError("holdout lock does not attest an opaque private packet")
    if Path(str(lock.get("bundle", ""))).resolve() != root:
        raise ValueError("holdout lock is bound to a different public bundle")
    if Path(str(lock.get("holdout_key", ""))).resolve() != key_path:
        raise ValueError("holdout lock is bound to a different private key path")
    if lock.get("holdout_key_sha256") != manifest.get("holdout_reveal_key_sha256"):
        raise ValueError("holdout lock and public key commitment disagree")

    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE"
    expected_bindings = {
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_complete_marker_sha256": sha256_file(complete_path),
        "validation_mapping_commitment_sha256": manifest[
            "validation_mapping_commitment_sha256"
        ],
        "holdout_packet_commitment_sha256": manifest[
            "holdout_packet_commitment_sha256"
        ],
        "operation_annotation_sha256": sha256_file(
            Path(str(lock.get("operation_annotation", ""))).resolve()
        ),
        "target_annotation_sha256": sha256_file(
            Path(str(lock.get("target_annotation", ""))).resolve()
        ),
        "template_manifest_sha256": sha256_file(
            Path(str(lock.get("template_manifest", ""))).resolve()
        ),
        "operation_codebook_sha256": sha256_file(
            Path(str(lock.get("operation_codebook", ""))).resolve()
        ),
        "target_codebook_sha256": sha256_file(
            Path(str(lock.get("target_codebook", ""))).resolve()
        ),
    }
    mismatched = sorted(
        field for field, current in expected_bindings.items() if lock.get(field) != current
    )
    frozen_documents = lock.get("frozen_documents_sha256")
    if not isinstance(frozen_documents, dict) or not frozen_documents:
        mismatched.append("frozen_documents_sha256")
    else:
        for recorded, expected_hash in frozen_documents.items():
            path = Path(str(recorded)).resolve()
            if not path.is_file() or sha256_file(path) != expected_hash:
                mismatched.append(f"frozen_document:{recorded}")

    lock_tool = Path(str(lock.get("lock_tool", ""))).resolve()
    reveal_tool = Path(str(lock.get("reveal_tool", ""))).resolve()
    if not lock_tool.is_file() or sha256_file(lock_tool) != lock.get("lock_tool_sha256"):
        mismatched.append("lock_tool_sha256")
    if reveal_tool != Path(__file__).resolve() or sha256_file(reveal_tool) != lock.get(
        "reveal_tool_sha256"
    ):
        mismatched.append("reveal_tool_sha256")

    locked_v_hashes = lock.get("holdout_files_sha256")
    if not isinstance(locked_v_hashes, dict) or set(locked_v_hashes) != set(blind_ids):
        mismatched.append("holdout_files_sha256")
    else:
        for blind_id in blind_ids:
            current = sha256_file(root / "holdout_rollouts" / f"{blind_id}.txt")
            if current != locked_v_hashes.get(blind_id):
                mismatched.append(f"holdout_file:{blind_id}")

    eligible_ids = lock.get("impartiality_eligible_blind_ids")
    if (
        not isinstance(eligible_ids, list)
        or any(blind_id not in blind_ids for blind_id in eligible_ids)
        or len(set(eligible_ids)) != len(eligible_ids)
    ):
        mismatched.append("impartiality_eligible_blind_ids")
    normalized = lock.get("normalized_blind_targets")
    if (
        not isinstance(normalized, list)
        or [source.get("blind_id") for source in normalized if isinstance(source, dict)]
        != blind_ids
        or [
            source.get("blind_id")
            for source in normalized
            if isinstance(source, dict) and source.get("impartiality_eligible") is True
        ]
        != eligible_ids
    ):
        mismatched.append("normalized_blind_targets")
    if mismatched:
        raise ValueError(f"holdout lock binding mismatch: {', '.join(sorted(set(mismatched)))}")
    return lock, manifest, blind_ids


def _validate_mapping_row(row: Mapping[str, Any], *, reserve: bool) -> None:
    expected_fields = _RESERVE_FIELDS if reserve else _MAPPING_FIELDS
    if set(row) != expected_fields:
        raise ValueError("private holdout mapping row has the wrong fields")
    if row.get("condition") not in CONDITIONS:
        raise ValueError("private holdout mapping contains an invalid condition")
    for field in ("rollout_id", "row_position"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"private holdout mapping contains invalid {field}")


def _validate_and_reconstruct_packet(
    packet: Any,
    *,
    packet_hash: str,
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    blind_ids: Sequence[str],
    bundle: Path,
    runs_root: Path,
) -> list[dict[str, Any]]:
    if not isinstance(packet, dict) or set(packet) != _PACKET_FIELDS:
        raise ValueError("private holdout packet fields are not exact")
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("private holdout packet has the wrong schema")
    core = dict(packet)
    recorded_packet_commitment = core.pop("packet_commitment_sha256", None)
    computed_packet_commitment = _canonical_sha256(core)
    if not (
        recorded_packet_commitment
        == computed_packet_commitment
        == manifest["holdout_packet_commitment_sha256"]
        == lock["holdout_packet_commitment_sha256"]
    ):
        raise ValueError("private holdout packet commitment mismatch")
    if packet_hash != lock["holdout_key_sha256"]:
        raise ValueError("private holdout packet byte hash changed after lock")

    selection = packet.get("selection")
    if not isinstance(selection, dict) or set(selection) != _SELECTION_FIELDS:
        raise ValueError("private holdout selection fields are not exact")
    if selection.get("algorithm") != (
        "remaining-pool per-condition SHA-256 ranking and independent display ranking"
    ):
        raise ValueError("private holdout selection algorithm is unknown")
    seed = selection.get("validation_seed")
    count_per_condition = selection.get("validation_per_condition")
    namespace = selection.get("namespace")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(count_per_condition, bool)
        or not isinstance(count_per_condition, int)
        or count_per_condition < 1
        or not isinstance(namespace, str)
        or not namespace.strip()
    ):
        raise ValueError("private holdout selection values are invalid")

    mapping = packet.get("validation_mapping")
    reserve_mapping = packet.get("reserve_mapping")
    if not isinstance(mapping, list) or not isinstance(reserve_mapping, list):
        raise ValueError("private holdout mappings must be lists")
    for row in mapping:
        if not isinstance(row, dict):
            raise ValueError("private holdout mapping row must be an object")
        _validate_mapping_row(row, reserve=False)
    for row in reserve_mapping:
        if not isinstance(row, dict):
            raise ValueError("private reserve mapping row must be an object")
        _validate_mapping_row(row, reserve=True)
    if packet.get("reserve_order") != (
        "precommitted validation-ranking remainder within each arm"
    ):
        raise ValueError("private reserve ordering declaration is invalid")
    if [row["blind_id"] for row in mapping] != list(blind_ids):
        raise ValueError("private holdout V IDs are not exact and ordered")
    if len({row["blind_id"] for row in mapping}) != len(mapping):
        raise ValueError("private holdout V IDs are duplicated")
    expected_counts = Counter({condition: count_per_condition for condition in CONDITIONS})
    if Counter(row["condition"] for row in mapping) != expected_counts:
        raise ValueError("private holdout mapping is not exactly balanced")
    mapping_commitment = _canonical_sha256(mapping)
    if not (
        mapping_commitment
        == packet.get("validation_mapping_commitment_sha256")
        == manifest["validation_mapping_commitment_sha256"]
        == lock["validation_mapping_commitment_sha256"]
    ):
        raise ValueError("private holdout mapping commitment mismatch")
    if packet.get("discovery_mapping_commitment_sha256") != manifest.get(
        "discovery_mapping_commitment_sha256"
    ):
        raise ValueError("discovery commitment changed inside holdout packet")

    raw_root = runs_root.resolve()
    source_run = Path(str(packet.get("source_run", ""))).resolve()
    if source_run == raw_root or not _is_within(source_run, raw_root):
        raise ValueError("private holdout source run is outside the pinned raw root")
    source_hashes = {
        name: sha256_file(source_run / name)
        for name in ("config.json", "threshold.json", "above_good.json", "below_good.json")
    }
    if packet.get("source_files_sha256") != source_hashes:
        raise ValueError("private holdout source snapshot no longer matches")
    by_condition, source_audit = load_eligible_records(source_run)
    raw_by_key = {
        (record.condition, record.rollout_id): record
        for records in by_condition.values()
        for record in records
    }
    validation_keys = {(row["condition"], row["rollout_id"]) for row in mapping}
    reserve_keys = {
        (row["condition"], row["rollout_id"]) for row in reserve_mapping
    }
    if validation_keys & reserve_keys:
        raise ValueError("holdout and reserve mappings overlap")
    if any(key not in raw_by_key for key in validation_keys | reserve_keys):
        raise ValueError("private holdout packet references an ineligible source")

    expected_selected_raw = []
    expected_reserve_raw = []
    for condition in CONDITIONS:
        remaining = [
            raw_by_key[key]
            for key in validation_keys | reserve_keys
            if key[0] == condition
        ]
        missing_count = len(by_condition[condition]) - len(remaining)
        if missing_count != manifest.get("discovery_per_hidden_condition"):
            raise ValueError("holdout packet remaining pool has the wrong size")
        ranked = sorted(
            remaining,
            key=lambda record: (
                _score(namespace, seed, "validation-sample", record),
                record.rollout_id,
            ),
        )
        expected_selected_raw.extend(ranked[:count_per_condition])
        expected_reserve_raw.extend(ranked[count_per_condition:])
    expected_display = sorted(
        expected_selected_raw,
        key=lambda record: (
            _score(namespace, seed, "validation-display", record),
            record.canonical_key,
        ),
    )
    expected_records = [
        BlindRecord(f"V{index:03d}", record)
        for index, record in enumerate(expected_display, start=1)
    ]
    if mapping != _mapping(expected_records):
        raise ValueError("private seed does not reconstruct the exact V mapping")
    if reserve_mapping != _reserve_mapping(expected_reserve_raw):
        raise ValueError("private seed does not reconstruct the reserve integrity rows")
    for item in expected_records:
        expected_text = render_rollout(
            item,
            source_audit["threshold"],
            phase="HOLDOUT",
        ).encode("utf-8")
        public_bytes = (
            bundle.resolve() / "holdout_rollouts" / f"{item.blind_id}.txt"
        ).read_bytes()
        if public_bytes != expected_text:
            raise ValueError("public V text is not bound to the private mapping")
    return mapping


def _eligibility_gate(
    mapping: Sequence[Mapping[str, Any]],
    eligible_blind_ids: Sequence[str],
) -> dict[str, Any]:
    condition_by_id = {str(row["blind_id"]): str(row["condition"]) for row in mapping}
    ids_by_condition = {
        condition: [
            blind_id
            for blind_id in eligible_blind_ids
            if condition_by_id[blind_id] == condition
        ]
        for condition in CONDITIONS
    }
    counts = {condition: len(ids) for condition, ids in ids_by_condition.items()}
    minimum = min(counts.values())
    if minimum >= 6:
        decision = "confirmatory_ready"
    elif minimum >= 4:
        decision = "causal_case_series_only"
    else:
        decision = "stop_insufficient_sources"
    return {
        "schema_version": ELIGIBILITY_GATE_SCHEMA_VERSION,
        "eligibility_definition": "lock-derived impartiality_eligible before condition reveal",
        "eligible_counts_by_condition": counts,
        "eligible_blind_ids_by_condition": ids_by_condition,
        "minimum_condition_count": minimum,
        "confirmatory_minimum_per_condition": 6,
        "case_series_minimum_per_condition": 4,
        "decision": decision,
        "reserve_rescue_forbidden": True,
    }


def reveal_holdout(
    *,
    bundle: Path,
    annotation_lock: Path,
    holdout_key: Path,
    output_dir: Path,
    reveal_receipt: Path,
    runs_root: Path,
    confirm_annotations_and_targets_locked: bool,
    expected_total: int = 60,
) -> dict[str, Any]:
    if not confirm_annotations_and_targets_locked:
        raise ValueError(
            "explicit --confirm-holdout-annotations-and-targets-locked is required"
        )
    root = bundle.resolve()
    raw_root = runs_root.resolve()
    lock_path = annotation_lock.resolve()
    key_path = holdout_key.resolve()
    output = ensure_output_outside_raw(output_dir, [raw_root])
    receipt = ensure_output_outside_raw(reveal_receipt, [raw_root])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite holdout reveal output: {output}")
    if receipt.exists():
        raise FileExistsError(f"holdout reveal was already started: {receipt}")
    if output == root or _is_within(output, root):
        raise ValueError("holdout reveal output must be outside the public bundle")
    if receipt == root or _is_within(receipt, root):
        raise ValueError("holdout reveal receipt must be outside the public bundle")
    if output == key_path.parent or _is_within(output, key_path.parent):
        raise ValueError("holdout reveal output must be outside the sealed key directory")
    if receipt == key_path.parent or _is_within(receipt, key_path.parent):
        raise ValueError("holdout reveal receipt must be outside the sealed key directory")
    if not lock_path.is_file() or not key_path.is_file():
        raise FileNotFoundError("holdout lock and private key must both exist")

    # All non-secret validation finishes before the one-shot receipt is created.
    lock, manifest, blind_ids = _validate_public_lock_state(
        bundle=root,
        annotation_lock=lock_path,
        holdout_key=key_path,
        runs_root=raw_root,
        expected_total=expected_total,
    )
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "started_irreversible_single_reveal",
        "annotation_lock": str(lock_path),
        "annotation_lock_sha256": sha256_file(lock_path),
        "bundle_manifest_sha256": sha256_file(root / "manifest.json"),
        "planned_output": str(output),
        "statement": (
            "The private holdout packet may be read only in this invocation. "
            "This receipt is never overwritten or removed by the tool."
        ),
    }
    write_new_json(receipt, receipt_payload)

    # First and only private-byte read in this function; parsing happens after hash check.
    key_bytes = key_path.read_bytes()
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    if key_hash != lock["holdout_key_sha256"]:
        raise ValueError("private holdout key hash changed after annotation lock")
    try:
        packet = json.loads(key_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private holdout packet is not valid UTF-8 JSON") from exc
    mapping = _validate_and_reconstruct_packet(
        packet,
        packet_hash=key_hash,
        manifest=manifest,
        lock=lock,
        blind_ids=blind_ids,
        bundle=root,
        runs_root=raw_root,
    )
    gate = _eligibility_gate(mapping, lock["impartiality_eligible_blind_ids"])

    if sha256_file(lock_path) != receipt_payload["annotation_lock_sha256"]:
        raise ValueError("annotation lock changed during private reveal validation")
    if hashlib.sha256(key_bytes).hexdigest() != key_hash:
        raise ValueError("in-memory private packet changed unexpectedly")
    output.mkdir(parents=True, exist_ok=False)
    mapping_path = write_new_text(
        output / "holdout_reveal.csv",
        _render_mapping_csv(mapping),
    )
    gate_path = write_new_json(output / "eligibility_gate.json", gate)
    provenance_path = write_new_json(
        output / "provenance.json",
        {
            "schema_version": REVEAL_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "annotation_lock": str(lock_path),
            "annotation_lock_sha256": sha256_file(lock_path),
            "single_reveal_receipt": str(receipt),
            "single_reveal_receipt_sha256": sha256_file(receipt),
            "bundle_manifest_sha256": sha256_file(root / "manifest.json"),
            "holdout_packet_sha256": key_hash,
            "holdout_packet_commitment_sha256": manifest[
                "holdout_packet_commitment_sha256"
            ],
            "validation_mapping_commitment_sha256": manifest[
                "validation_mapping_commitment_sha256"
            ],
            "private_packet_parsed": True,
            "reserve_rows_emitted": False,
            "reserve_rows_used_for_sample_rescue": False,
            "reserve_rows_checked_only_for_precommitted_split_integrity": True,
            "scope": "precommitted V mapping and lock-derived eligibility gate only",
        },
    )
    artifact_paths = (mapping_path, gate_path, provenance_path)
    reveal_manifest_path = write_new_json(
        output / "manifest.json",
        {
            "schema_version": REVEAL_SCHEMA_VERSION,
            "row_count": len(mapping),
            "csv_fields": list(HOLDOUT_REVEAL_CSV_FIELDS),
            "eligibility_gate_schema_version": ELIGIBILITY_GATE_SCHEMA_VERSION,
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifact_paths
            },
            "contains_holdout_mapping": True,
            "contains_reserve_rows": False,
            "single_reveal_receipt_sha256": sha256_file(receipt),
        },
    )
    complete_path = write_new_text(
        output / "COMPLETE",
        f"manifest_sha256={sha256_file(reveal_manifest_path)}\n",
    )
    return {
        "output": str(output),
        "row_count": len(mapping),
        "eligibility_gate": gate,
        "manifest": str(reveal_manifest_path),
        "complete_marker": str(complete_path),
        "single_reveal_receipt": str(receipt),
        "reserve_rows_emitted": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--annotation-lock", type=Path, required=True)
    parser.add_argument("--holdout-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reveal-receipt", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--expected-total", type=int, default=60)
    parser.add_argument(
        "--confirm-holdout-annotations-and-targets-locked",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = reveal_holdout(
            bundle=args.bundle,
            annotation_lock=args.annotation_lock,
            holdout_key=args.holdout_key,
            output_dir=args.output_dir,
            reveal_receipt=args.reveal_receipt,
            runs_root=args.runs_root,
            confirm_annotations_and_targets_locked=(
                args.confirm_holdout_annotations_and_targets_locked
            ),
            expected_total=args.expected_total,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ValueError,
        csv.Error,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        parser.error(str(exc))
    summary = {
        "output": result["output"],
        "row_count": result["row_count"],
        "eligibility_gate": result["eligibility_gate"],
        "manifest": result["manifest"],
        "complete_marker": result["complete_marker"],
        "single_reveal_receipt": result["single_reveal_receipt"],
        "reserve_rows_emitted": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

