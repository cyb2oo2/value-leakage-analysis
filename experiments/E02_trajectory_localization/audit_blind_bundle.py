"""Audit blinded artifacts without leaking mappings in the report.

The default public audit never reads either private JSON packet. A separate
privileged function exists for post-reveal forensic verification and requires
both packet paths explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.E02_trajectory_localization.blind_discovery import (
    BlindRecord,
    CONDITIONS,
    _mapping,
    _reserve_mapping,
    allocate_split,
    load_eligible_records,
    render_rollout,
)
from research.experiment_utils import sha256_file, sha256_text


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
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
_HOLDOUT_PACKET_FIELDS = frozenset(
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_bytes_once(path: Path) -> tuple[bytes, Any]:
    payload = path.read_bytes()
    return payload, json.loads(payload.decode("utf-8"))


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


def _safe_artifact_targets(
    root: Path,
    artifact_hashes: Mapping[str, str],
) -> tuple[bool, dict[str, Path]]:
    targets: dict[str, Path] = {}
    safe = True
    for relative, expected_hash in artifact_hashes.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or not isinstance(expected_hash, str)
            or not _SHA256_PATTERN.fullmatch(expected_hash)
        ):
            safe = False
            continue
        target = (root / relative).resolve()
        if target == root or not _is_within(target, root):
            safe = False
            continue
        targets[relative] = target
    return safe and len(targets) == len(artifact_hashes), targets


def _expected_public_ids(prefix: str, total: Any) -> list[str]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        return []
    return [f"{prefix}{index:03d}" for index in range(1, total + 1)]


def audit_public_bundle(
    bundle: Path,
    annotation: Path,
    *,
    runs_root: Path,
    require_annotation_template_unchanged: bool = True,
) -> dict[str, Any]:
    """Audit public artifacts while treating private packets as unavailable."""

    root = bundle.resolve()
    annotation_path = annotation.resolve()
    raw_root = runs_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    provenance = _read_json(root / "provenance.json")

    artifact_hashes = manifest.get("artifact_sha256", {})
    artifact_map_valid = isinstance(artifact_hashes, dict)
    safe_paths, artifact_targets = (
        _safe_artifact_targets(root, artifact_hashes)
        if artifact_map_valid
        else (False, {})
    )
    expected_artifacts = set(artifact_targets)
    observed_artifacts = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    }
    artifact_hashes_match = safe_paths and all(
        target.is_file() and sha256_file(target) == artifact_hashes[relative]
        for relative, target in artifact_targets.items()
    )
    complete_text = (root / "COMPLETE").read_text(encoding="utf-8").strip()
    complete_ok = complete_text == f"manifest_sha256={sha256_file(manifest_path)}"

    private_key_names = {
        "discovery_reveal_key.json",
        "HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json",
        "HOLDOUT_DO_NOT_OPEN_UNTIL_HYPOTHESES_FROZEN.json",
    }
    no_private_keys = not any(
        path.name in private_key_names for path in root.rglob("*") if path.is_file()
    )
    private_separation_declared = (
        manifest.get("private_keys_in_public_bundle") is False
        and provenance.get("private_keys_in_public_bundle") is False
    )

    public_metadata = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("README.md", "PRIVATE_KEYS.md", "manifest.json", "provenance.json")
    )
    public_metadata_labels_hidden = all(
        forbidden not in public_metadata
        for forbidden in ("above_good", "below_good")
    )

    expected_discovery_ids = _expected_public_ids(
        "D", manifest.get("discovery_total")
    )
    observed_discovery_ids = sorted(
        path.stem for path in (root / "rollouts").glob("*.txt")
    )
    expected_holdout_ids = _expected_public_ids(
        "V", manifest.get("holdout_rollout_total")
    )
    observed_holdout_ids = sorted(
        path.stem for path in (root / "holdout_rollouts").glob("*.txt")
    )

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as stream:
        annotation_rows = list(csv.DictReader(stream))
    annotation_ids = [row.get("blind_id") for row in annotation_rows]
    annotation_template_unchanged = (
        sha256_file(annotation_path) == manifest.get("annotation_template_sha256")
    )
    annotation_path_matches = (
        Path(str(manifest.get("annotation_template", ""))).resolve()
        == annotation_path
    )

    source_run = Path(str(provenance.get("source_run", ""))).resolve()
    source_contained = (
        raw_root.is_dir()
        and source_run != raw_root
        and _is_within(source_run, raw_root)
    )
    source_hashes_ok = False
    if source_contained:
        source_hashes = provenance.get("source_files_sha256", {})
        try:
            source_hashes_ok = (
                sha256_file(source_run / "config.json") == source_hashes["config.json"]
                and sha256_file(source_run / "threshold.json")
                == source_hashes["threshold.json"]
                and sorted(
                    sha256_file(source_run / name)
                    for name in ("above_good.json", "below_good.json")
                )
                == source_hashes["hidden_intervention_files_sorted"]
            )
        except (FileNotFoundError, KeyError, TypeError):
            source_hashes_ok = False

    hash_fields = (
        "discovery_mapping_commitment_sha256",
        "validation_mapping_commitment_sha256",
        "discovery_packet_commitment_sha256",
        "holdout_packet_commitment_sha256",
        "discovery_reveal_key_sha256",
        "holdout_reveal_key_sha256",
    )
    commitments_well_formed = all(
        isinstance(manifest.get(name), str)
        and bool(_SHA256_PATTERN.fullmatch(manifest[name]))
        for name in hash_fields
    )
    checks = {
        "artifact_paths_are_contained": safe_paths,
        "all_manifest_artifact_hashes_match": artifact_hashes_match,
        "artifact_inventory_matches_manifest": observed_artifacts == expected_artifacts,
        "complete_marker_matches_manifest": complete_ok,
        "private_packets_are_absent_from_public_bundle": no_private_keys,
        "private_packet_separation_is_declared": private_separation_declared,
        "commitments_and_private_hashes_are_well_formed": commitments_well_formed,
        "public_metadata_omits_condition_labels": public_metadata_labels_hidden,
        "public_discovery_ids_match_manifest": observed_discovery_ids
        == expected_discovery_ids,
        "withheld_holdout_ids_match_manifest": observed_holdout_ids
        == expected_holdout_ids,
        "annotation_path_matches_manifest": annotation_path_matches,
        "annotation_hash_policy_satisfied": annotation_template_unchanged
        or not require_annotation_template_unchanged,
        "annotation_ids_match_public_discovery_order": annotation_ids
        == expected_discovery_ids,
        "source_run_is_inside_declared_runs_root": source_contained,
        "immutable_source_hashes_match": source_hashes_ok,
    }
    return {
        "audit_version": "public-opaque-private/v2",
        "ok": all(checks.values()),
        "checks": checks,
        "counts": {
            "discovery_total": len(expected_discovery_ids),
            "withheld_holdout_total": len(expected_holdout_ids),
            "annotation_row_total": len(annotation_rows),
        },
        "annotation_template_unchanged": annotation_template_unchanged,
        "private_packets_parsed": False,
    }


def _mapping_keys(mapping: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(row["condition"], row["rollout_id"]) for row in mapping}


def _packet_commitment_matches(packet: dict[str, Any], expected: str) -> bool:
    core = dict(packet)
    observed = core.pop("packet_commitment_sha256", None)
    return observed == expected == _canonical_sha256(core)


def audit_bundle(
    bundle: Path,
    annotation: Path,
    *,
    discovery_key: Path,
    holdout_key: Path,
    runs_root: Path,
    require_annotation_template_unchanged: bool = True,
) -> dict[str, Any]:
    """Privileged post-reveal audit; this intentionally parses both packets."""

    root = bundle.resolve()
    manifest = _read_json(root / "manifest.json")
    provenance = _read_json(root / "provenance.json")
    public_report = audit_public_bundle(
        root,
        annotation,
        runs_root=runs_root,
        require_annotation_template_unchanged=require_annotation_template_unchanged,
    )

    discovery_bytes, discovery = _read_json_bytes_once(discovery_key.resolve())
    holdout_bytes, holdout = _read_json_bytes_once(holdout_key.resolve())
    key_hashes_match = (
        hashlib.sha256(discovery_bytes).hexdigest()
        == manifest["discovery_reveal_key_sha256"]
        and hashlib.sha256(holdout_bytes).hexdigest()
        == manifest["holdout_reveal_key_sha256"]
    )
    packet_fields_exact = (
        isinstance(discovery, dict)
        and isinstance(holdout, dict)
        and set(discovery) == _DISCOVERY_PACKET_FIELDS
        and set(holdout) == _HOLDOUT_PACKET_FIELDS
    )
    discovery_packet_ok = packet_fields_exact and _packet_commitment_matches(
        discovery, manifest["discovery_packet_commitment_sha256"]
    )
    holdout_packet_ok = packet_fields_exact and _packet_commitment_matches(
        holdout, manifest["holdout_packet_commitment_sha256"]
    )

    discovery_mapping = discovery.get("discovery_mapping", [])
    validation_mapping = holdout.get("validation_mapping", [])
    reserve_mapping = holdout.get("reserve_mapping", [])
    mappings_are_lists = all(
        isinstance(value, list)
        for value in (discovery_mapping, validation_mapping, reserve_mapping)
    )
    discovery_commitment_ok = mappings_are_lists and (
        _canonical_sha256(discovery_mapping)
        == manifest["discovery_mapping_commitment_sha256"]
        == discovery.get("discovery_mapping_commitment_sha256")
        == holdout.get("discovery_mapping_commitment_sha256")
    )
    validation_commitment_ok = mappings_are_lists and (
        _canonical_sha256(validation_mapping)
        == manifest["validation_mapping_commitment_sha256"]
        == discovery.get("validation_mapping_commitment_sha256")
        == holdout.get("validation_mapping_commitment_sha256")
    )

    source_run = Path(provenance["source_run"]).resolve()
    by_condition, source_audit = load_eligible_records(source_run)
    expected_discovery, expected_validation, expected_reserve = allocate_split(
        by_condition,
        discovery_seed=discovery["selection"]["discovery_seed"],
        validation_seed=holdout["selection"]["validation_seed"],
        namespace=discovery["selection"]["namespace"],
        discovery_per_condition=discovery["selection"]["discovery_per_condition"],
        validation_per_condition=holdout["selection"]["validation_per_condition"],
    )
    selection_reconstructs_exactly = (
        discovery["selection"]["namespace"] == holdout["selection"]["namespace"]
        and discovery_mapping == _mapping(expected_discovery)
        and validation_mapping == _mapping(expected_validation)
        and reserve_mapping == _reserve_mapping(expected_reserve)
    )

    d_keys = _mapping_keys(discovery_mapping) if mappings_are_lists else set()
    v_keys = _mapping_keys(validation_mapping) if mappings_are_lists else set()
    r_keys = _mapping_keys(reserve_mapping) if mappings_are_lists else set()
    exact_universe = {
        (condition, record.rollout_id)
        for condition, records in by_condition.items()
        for record in records
    }
    memberships_exact = (
        not (d_keys & v_keys or d_keys & r_keys or v_keys & r_keys)
        and d_keys | v_keys | r_keys == exact_universe
    )
    discovery_ids = [row.get("blind_id") for row in discovery_mapping]
    validation_ids = [row.get("blind_id") for row in validation_mapping]
    ids_exact = (
        discovery_ids == _expected_public_ids("D", len(discovery_mapping))
        and validation_ids == _expected_public_ids("V", len(validation_mapping))
    )
    discovery_counts = Counter(row.get("condition") for row in discovery_mapping)
    validation_counts = Counter(row.get("condition") for row in validation_mapping)
    arms_exact_and_balanced = (
        set(discovery_counts) == set(CONDITIONS)
        and set(validation_counts) == set(CONDITIONS)
        and len(set(discovery_counts.values())) == 1
        and len(set(validation_counts.values())) == 1
    )

    public_text_binding = True
    for item in (*expected_discovery, *expected_validation):
        phase = "DISCOVERY" if item.blind_id.startswith("D") else "HOLDOUT"
        directory = "rollouts" if phase == "DISCOVERY" else "holdout_rollouts"
        expected_text = render_rollout(
            BlindRecord(item.blind_id, item.raw),
            source_audit["threshold"],
            phase=phase,
        )
        path = root / directory / f"{item.blind_id}.txt"
        if not path.is_file() or path.read_text(encoding="utf-8") != expected_text:
            public_text_binding = False
            break

    private_checks = {
        "private_key_byte_hashes_match": key_hashes_match,
        "packet_fields_are_exact": packet_fields_exact,
        "discovery_packet_commitment_matches": discovery_packet_ok,
        "holdout_packet_commitment_matches": holdout_packet_ok,
        "discovery_mapping_commitment_matches": discovery_commitment_ok,
        "validation_mapping_commitment_matches": validation_commitment_ok,
        "selection_reconstructs_exact_mappings": selection_reconstructs_exactly,
        "memberships_equal_exact_eligible_universe": memberships_exact,
        "blind_ids_are_unique_contiguous_and_ordered": ids_exact,
        "both_hidden_arms_are_exact_and_balanced": arms_exact_and_balanced,
        "public_rollout_text_is_bound_to_source_rows": public_text_binding,
        "discovery_packet_omits_holdout_payload": not any(
            key in discovery
            for key in ("validation_seed", "validation_mapping", "reserve_mapping")
        ),
        "holdout_packet_omits_discovery_mapping": "discovery_mapping" not in holdout,
    }
    checks = {**public_report["checks"], **private_checks}
    return {
        "audit_version": "privileged-both-packets/v2",
        "ok": all(checks.values()),
        "checks": checks,
        "counts": {
            "discovery_total": len(discovery_mapping),
            "validation_total": len(validation_mapping),
            "reserve_total": len(reserve_mapping),
            "eligible_total": len(exact_universe),
            "hidden_arm_discovery_counts_sorted": sorted(discovery_counts.values()),
            "hidden_arm_validation_counts_sorted": sorted(validation_counts.values()),
        },
        "annotation_template_unchanged": public_report[
            "annotation_template_unchanged"
        ],
        "private_packets_parsed": True,
        "commitments": {
            "discovery_mapping_sha256": manifest[
                "discovery_mapping_commitment_sha256"
            ],
            "validation_mapping_sha256": manifest[
                "validation_mapping_commitment_sha256"
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--private-discovery-key", type=Path)
    parser.add_argument("--private-holdout-key", type=Path)
    parser.add_argument("--confirm-open-both-private-packets", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    private_requested = any(
        (
            args.private_discovery_key,
            args.private_holdout_key,
            args.confirm_open_both_private_packets,
        )
    )
    if private_requested:
        if not all(
            (
                args.private_discovery_key,
                args.private_holdout_key,
                args.confirm_open_both_private_packets,
            )
        ):
            parser.error(
                "privileged audit requires both key paths and explicit confirmation"
            )
        report = audit_bundle(
            args.bundle,
            args.annotation,
            discovery_key=args.private_discovery_key,
            holdout_key=args.private_holdout_key,
            runs_root=args.runs_root,
        )
    else:
        report = audit_public_bundle(
            args.bundle,
            args.annotation,
            runs_root=args.runs_root,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
