"""Reveal only the locked discovery mapping after all fail-closed checks pass.

The holdout packet is read as opaque bytes solely to recheck its hash. It is
never decoded or parsed. No validation/reserve mapping is copied to output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any, Iterator, Sequence

from experiments.E02_trajectory_localization.audit_blind_bundle import (
    audit_public_bundle,
)
from experiments.E02_trajectory_localization.blind_discovery import (
    BlindRecord,
    CONDITIONS,
    SCHEMA_VERSION,
    _mapping,
    _score,
    load_eligible_records,
    render_rollout,
)
from experiments.E02_trajectory_localization.lock_discovery_annotation import (
    LOCK_SCHEMA_VERSION,
    public_discovery_ids,
    validate_completed_annotation,
    validate_hash_anchor,
)
from research.experiment_utils import (
    ensure_output_outside_raw,
    sha256_file,
    sha256_text,
    utc_now,
    write_new_json,
    write_new_text,
)


REVEAL_SCHEMA_VERSION = "value-leakage.discovery-reveal/v1"
_DISCOVERY_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "warning",
        "created_at_utc",
        "packet_nonce",
        "source_run",
        "source_files_sha256",
        "selection",
        "discovery_mapping",
        "discovery_mapping_commitment_sha256",
        "validation_mapping_commitment_sha256",
        "packet_commitment_sha256",
    }
)
_DISCOVERY_SELECTION_FIELDS = frozenset(
    {
        "algorithm",
        "namespace",
        "discovery_seed",
        "discovery_per_condition",
    }
)
_MAPPING_FIELDS = frozenset(
    {
        "blind_id",
        "condition",
        "rollout_id",
        "row_position",
        "reasoning_sha256",
        "visible_final_answer_sha256",
    }
)
_FORBIDDEN_DISCOVERY_KEYS = frozenset(
    {"validation_seed", "validation_mapping", "reserve_mapping"}
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


def _walk_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _packet_commitment(packet: dict[str, Any]) -> str:
    core = dict(packet)
    core.pop("packet_commitment_sha256", None)
    return _canonical_sha256(core)


def _expected_discovery(
    by_condition: dict[str, list[Any]],
    *,
    seed: int,
    namespace: str,
    count_per_condition: int,
) -> list[BlindRecord]:
    selected = []
    for condition in CONDITIONS:
        ranked = sorted(
            by_condition[condition],
            key=lambda record: (
                _score(namespace, seed, "discovery-sample", record),
                record.rollout_id,
            ),
        )
        selected.extend(ranked[:count_per_condition])
    ordered = sorted(
        selected,
        key=lambda record: (
            _score(namespace, seed, "discovery-display", record),
            record.canonical_key,
        ),
    )
    return [
        BlindRecord(f"D{index:03d}", record)
        for index, record in enumerate(ordered, start=1)
    ]


def _render_reveal_csv(mapping: Sequence[dict[str, Any]]) -> str:
    fields = (
        "blind_id",
        "condition",
        "rollout_id",
        "row_position",
        "reasoning_sha256",
        "visible_final_answer_sha256",
    )
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in mapping:
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue()


def reveal_discovery(
    *,
    bundle: Path,
    annotation: Path,
    codebook: Path,
    anchor_manifest: Path,
    annotation_lock: Path,
    discovery_key: Path,
    holdout_key: Path,
    output_dir: Path,
    runs_root: Path,
    repo_root: Path,
    confirm_annotations_locked: bool,
) -> dict[str, Any]:
    if not confirm_annotations_locked:
        raise ValueError(
            "explicit --confirm-discovery-annotations-locked is required"
        )

    root = bundle.resolve()
    annotation_path = annotation.resolve()
    codebook_path = codebook.resolve()
    anchor_path = anchor_manifest.resolve()
    lock_path = annotation_lock.resolve()
    discovery_key_path = discovery_key.resolve()
    holdout_key_path = holdout_key.resolve()
    raw_root = runs_root.resolve()
    output = ensure_output_outside_raw(output_dir, [raw_root])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reveal output: {output}")
    if output == root or _is_within(output, root):
        raise ValueError("reveal output must be outside the immutable public bundle")
    for label, path in (
        ("annotation", annotation_path),
        ("codebook", codebook_path),
        ("private hash anchor", anchor_path),
        ("annotation lock", lock_path),
        ("private discovery key", discovery_key_path),
        ("private holdout key", holdout_key_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if _is_within(discovery_key_path, root) or _is_within(holdout_key_path, root):
        raise ValueError("private keys must be physically outside the public bundle")
    if _is_within(discovery_key_path, raw_root) or _is_within(
        holdout_key_path, raw_root
    ):
        raise ValueError("private keys must be outside immutable raw runs")

    # Public and annotation checks run before either private packet is decoded.
    expected_ids = public_discovery_ids(root)
    annotation_validation = validate_completed_annotation(
        annotation_path, expected_ids
    )
    public_audit = audit_public_bundle(
        root,
        annotation_path,
        runs_root=raw_root,
        require_annotation_template_unchanged=False,
    )
    if not public_audit["ok"]:
        failed = sorted(
            name for name, passed in public_audit["checks"].items() if not passed
        )
        raise ValueError(f"public bundle audit failed: {', '.join(failed)}")

    lock = _read_json(lock_path)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    anchor_validation = validate_hash_anchor(
        anchor_path,
        bundle=root,
        annotation=annotation_path,
        codebook=codebook_path,
        manifest=manifest,
        repo_root=repo_root,
    )
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError("annotation lock has the wrong schema")
    if lock.get("private_packets_parsed_by_lock") is not False:
        raise ValueError("annotation lock does not attest opaque private packets")

    # Read private bytes once. Hash both; only discovery_bytes is decoded later.
    discovery_bytes = discovery_key_path.read_bytes()
    holdout_bytes = holdout_key_path.read_bytes()
    discovery_hash = hashlib.sha256(discovery_bytes).hexdigest()
    holdout_hash = hashlib.sha256(holdout_bytes).hexdigest()
    current_bindings = {
        "bundle": str(root),
        "bundle_manifest": str(manifest_path.resolve()),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_complete_marker_sha256": sha256_file(root / "COMPLETE"),
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
        "discovery_reveal_key": str(discovery_key_path),
        "discovery_reveal_key_sha256": discovery_hash,
        "holdout_reveal_key": str(holdout_key_path),
        "holdout_reveal_key_sha256": holdout_hash,
        "annotation": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "annotation_original_template_sha256": manifest[
            "annotation_template_sha256"
        ],
        "codebook": str(codebook_path),
        "codebook_sha256": sha256_file(codebook_path),
        "hash_anchor": str(anchor_path),
        "hash_anchor_sha256": sha256_file(anchor_path),
    }
    mismatched = sorted(
        field
        for field, observed in current_bindings.items()
        if lock.get(field) != observed
    )
    if mismatched:
        raise ValueError(f"annotation lock binding mismatch: {', '.join(mismatched)}")
    if lock.get("annotation_validation") != annotation_validation:
        raise ValueError("annotation validation summary no longer matches the lock")
    if lock.get("hash_anchor_validation") != anchor_validation:
        raise ValueError("private hash anchor validation no longer matches the lock")
    lock_tool = Path(str(lock.get("lock_tool", ""))).resolve()
    if (
        not lock_tool.is_file()
        or sha256_file(lock_tool) != lock.get("lock_tool_sha256")
    ):
        raise ValueError("the lock tool changed after annotation locking")
    if discovery_hash != manifest["discovery_reveal_key_sha256"]:
        raise ValueError("private discovery key hash does not match manifest")
    if holdout_hash != manifest["holdout_reveal_key_sha256"]:
        raise ValueError("private holdout key hash does not match manifest")

    # This is the first and only point at which the discovery packet is parsed.
    discovery = json.loads(discovery_bytes.decode("utf-8"))
    if not isinstance(discovery, dict) or set(discovery) != _DISCOVERY_PACKET_FIELDS:
        raise ValueError("private discovery packet fields are not exact")
    if any(key in _FORBIDDEN_DISCOVERY_KEYS for key in _walk_keys(discovery)):
        raise ValueError("private discovery packet contains holdout/reserve payload")
    if discovery.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("private discovery packet has the wrong schema")
    selection = discovery.get("selection")
    if not isinstance(selection, dict) or set(selection) != _DISCOVERY_SELECTION_FIELDS:
        raise ValueError("private discovery selection fields are not exact")
    if selection.get("algorithm") != (
        "per-condition SHA-256 ranking and independent SHA-256 display ranking"
    ):
        raise ValueError("private discovery selection algorithm is unknown")
    if (
        isinstance(selection.get("discovery_seed"), bool)
        or not isinstance(selection.get("discovery_seed"), int)
        or selection["discovery_seed"] < 0
        or not isinstance(selection.get("namespace"), str)
        or not selection["namespace"].strip()
        or isinstance(selection.get("discovery_per_condition"), bool)
        or not isinstance(selection.get("discovery_per_condition"), int)
        or selection["discovery_per_condition"] < 1
    ):
        raise ValueError("private discovery selection values are invalid")
    packet_commitment = _packet_commitment(discovery)
    if not (
        packet_commitment
        == discovery.get("packet_commitment_sha256")
        == manifest["discovery_packet_commitment_sha256"]
        == lock["discovery_packet_commitment_sha256"]
    ):
        raise ValueError("private discovery packet commitment mismatch")

    mapping = discovery.get("discovery_mapping")
    if not isinstance(mapping, list) or any(
        not isinstance(row, dict) or set(row) != _MAPPING_FIELDS for row in mapping
    ):
        raise ValueError("private discovery mapping schema is invalid")
    mapping_commitment = _canonical_sha256(mapping)
    if not (
        mapping_commitment
        == discovery.get("discovery_mapping_commitment_sha256")
        == manifest["discovery_mapping_commitment_sha256"]
        == lock["discovery_mapping_commitment_sha256"]
    ):
        raise ValueError("private discovery mapping commitment mismatch")
    if discovery.get("validation_mapping_commitment_sha256") != manifest[
        "validation_mapping_commitment_sha256"
    ]:
        raise ValueError("validation commitment changed inside discovery packet")

    blind_ids = [row.get("blind_id") for row in mapping]
    if blind_ids != expected_ids or len(set(blind_ids)) != len(blind_ids):
        raise ValueError("private discovery blind IDs are not exact and ordered")
    if any(
        row.get("condition") not in CONDITIONS
        or isinstance(row.get("rollout_id"), bool)
        or not isinstance(row.get("rollout_id"), int)
        or row["rollout_id"] < 0
        or isinstance(row.get("row_position"), bool)
        or not isinstance(row.get("row_position"), int)
        or row["row_position"] < 0
        for row in mapping
    ):
        raise ValueError("private discovery mapping contains invalid source fields")
    condition_counts = Counter(row["condition"] for row in mapping)
    expected_per_condition = selection.get("discovery_per_condition")
    if condition_counts != Counter(
        {condition: expected_per_condition for condition in CONDITIONS}
    ):
        raise ValueError("private discovery mapping is not exactly balanced")

    source_run = Path(str(discovery.get("source_run", ""))).resolve()
    if (
        source_run != Path(_read_json(root / "provenance.json")["source_run"]).resolve()
        or source_run == raw_root
        or not _is_within(source_run, raw_root)
    ):
        raise ValueError("private discovery source run is outside the pinned raw root")
    expected_source_hashes = {
        name: sha256_file(source_run / name)
        for name in (
            "config.json",
            "threshold.json",
            "above_good.json",
            "below_good.json",
        )
    }
    if discovery.get("source_files_sha256") != expected_source_hashes:
        raise ValueError("private discovery source snapshot no longer matches")

    by_condition, source_audit = load_eligible_records(source_run)
    expected_records = _expected_discovery(
        by_condition,
        seed=selection["discovery_seed"],
        namespace=selection["namespace"],
        count_per_condition=selection["discovery_per_condition"],
    )
    if mapping != _mapping(expected_records):
        raise ValueError("selection seed does not reconstruct the discovery mapping")
    for item in expected_records:
        public_path = root / "rollouts" / f"{item.blind_id}.txt"
        expected_text = render_rollout(
            item,
            source_audit["threshold"],
            phase="DISCOVERY",
        )
        if public_path.read_text(encoding="utf-8") != expected_text:
            raise ValueError("public discovery text is not bound to the private mapping")

    # Recheck every lock-bound byte immediately before exclusive output creation.
    final_hashes = {
        "manifest": sha256_file(manifest_path),
        "complete": sha256_file(root / "COMPLETE"),
        "annotation": sha256_file(annotation_path),
        "codebook": sha256_file(codebook_path),
        "hash_anchor": sha256_file(anchor_path),
        "discovery_key": hashlib.sha256(discovery_bytes).hexdigest(),
        "holdout_key": hashlib.sha256(holdout_bytes).hexdigest(),
    }
    expected_final_hashes = {
        "manifest": lock["bundle_manifest_sha256"],
        "complete": lock["bundle_complete_marker_sha256"],
        "annotation": lock["annotation_sha256"],
        "codebook": lock["codebook_sha256"],
        "hash_anchor": lock["hash_anchor_sha256"],
        "discovery_key": lock["discovery_reveal_key_sha256"],
        "holdout_key": lock["holdout_reveal_key_sha256"],
    }
    if final_hashes != expected_final_hashes:
        raise ValueError("a lock-bound artifact changed during reveal validation")

    output.mkdir(parents=True, exist_ok=False)
    reveal_csv = write_new_text(
        output / "discovery_reveal.csv", _render_reveal_csv(mapping)
    )
    provenance = write_new_json(
        output / "provenance.json",
        {
            "schema_version": REVEAL_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "bundle_manifest_sha256": final_hashes["manifest"],
            "annotation_lock": str(lock_path),
            "annotation_lock_sha256": sha256_file(lock_path),
            "annotation_sha256": final_hashes["annotation"],
            "codebook_sha256": final_hashes["codebook"],
            "hash_anchor_sha256": final_hashes["hash_anchor"],
            "discovery_packet_sha256": discovery_hash,
            "discovery_packet_commitment_sha256": packet_commitment,
            "discovery_mapping_commitment_sha256": mapping_commitment,
            "holdout_packet_parsed": False,
            "scope": "discovery mapping only",
            "row_count": len(mapping),
        },
    )
    reveal_manifest = write_new_json(
        output / "manifest.json",
        {
            "schema_version": REVEAL_SCHEMA_VERSION,
            "row_count": len(mapping),
            "artifact_sha256": {
                reveal_csv.name: sha256_file(reveal_csv),
                provenance.name: sha256_file(provenance),
            },
            "contains_discovery_mapping": True,
            "contains_non_discovery_payload": False,
            "holdout_packet_parsed": False,
        },
    )
    complete = write_new_text(
        output / "COMPLETE",
        f"manifest_sha256={sha256_file(reveal_manifest)}\n",
    )
    return {
        "output": str(output),
        "row_count": len(mapping),
        "reveal_csv_sha256": sha256_file(reveal_csv),
        "manifest": str(reveal_manifest),
        "complete_marker": str(complete),
        "holdout_packet_parsed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--hash-anchor", type=Path, required=True)
    parser.add_argument("--annotation-lock", type=Path, required=True)
    parser.add_argument("--private-discovery-key", type=Path, required=True)
    parser.add_argument("--private-holdout-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--confirm-discovery-annotations-locked",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = reveal_discovery(
            bundle=args.bundle,
            annotation=args.annotation,
            codebook=args.codebook,
            anchor_manifest=args.hash_anchor,
            annotation_lock=args.annotation_lock,
            discovery_key=args.private_discovery_key,
            holdout_key=args.private_holdout_key,
            output_dir=args.output_dir,
            runs_root=args.runs_root,
            repo_root=args.repo_root,
            confirm_annotations_locked=args.confirm_discovery_annotations_locked,
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
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
