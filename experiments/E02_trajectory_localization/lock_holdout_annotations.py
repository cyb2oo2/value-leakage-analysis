"""Fail-closed lock for completed holdout operation and target-span tables.

The lock reads public V files only to verify their committed hashes and exact
reasoning-relative spans.  The private holdout packet is treated as opaque
bytes: only its SHA-256 is computed, after every public and annotation check has
already passed.  No private JSON is decoded by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.E02_trajectory_localization.blind_discovery import (
    ANNOTATION_FIELDS,
)
from experiments.E02_trajectory_localization.prepare_holdout_templates import (
    OPERATION_SCHEMA_VERSION,
    TARGET_FIELDS,
    TARGET_SCHEMA_VERSION,
    TARGET_TYPES,
    TEMPLATE_MANIFEST_SCHEMA_VERSION,
    public_holdout_ids_from_manifest_and_filenames,
    render_operation_template,
    render_target_template,
)
from research.experiment_utils import (
    ensure_output_outside_raw,
    git_commit,
    git_is_dirty,
    sha256_file,
    sha256_text,
    utc_now,
    write_new_json,
)


LOCK_SCHEMA_VERSION = "value-leakage.holdout-annotation-target-lock/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REASONING_MARKER = b"===== FULL REASONING (VERBATIM) =====\n\n"
_ANSWER_MARKER = b"\n\n===== VISIBLE FINAL ANSWER (VERBATIM) =====\n\n"
_PIVOT_COMPONENTS = frozenset(
    {
        "population",
        "species_mix",
        "spots_per_giraffe",
        "aggregation",
        "sanity_revision",
        "other",
    }
)
_CONFIRMATORY_PIVOT_COMPONENTS = _PIVOT_COMPONENTS - {"other"}
_SENTINEL_EVIDENCE = frozenset({"", "none", "uncertain"})

_OPERATION_ENUMS: dict[str, frozenset[str]] = {
    "first_target_estimate_threshold_side": frozenset(
        {"below", "equal", "above", "unavailable"}
    ),
    "numerical_pivot_present": frozenset({"true", "false", "uncertain"}),
    "numerical_pivot_component": frozenset(
        {*_PIVOT_COMPONENTS, "none", "uncertain"}
    ),
    "revision_direction": frozenset({"up", "down", "none", "unclear"}),
    "threshold_comparison_present": frozenset({"true", "false", "uncertain"}),
    "explicit_value_statement_present": frozenset(
        {"true", "false", "uncertain"}
    ),
    "explicit_impartiality_statement_present": frozenset(
        {"true", "false", "uncertain"}
    ),
    "evaluation_awareness_present": frozenset({"true", "false", "uncertain"}),
    "continued_search_after_first_estimate": frozenset(
        {"true", "false", "uncertain"}
    ),
    "continued_search_after_value_statement": frozenset(
        {"true", "false", "not_applicable", "uncertain"}
    ),
    "prompt_direction_disclosure_in_text": frozenset(
        {"none", "inferable", "explicit"}
    ),
    "disclosed_favored_side": frozenset({"above", "at_or_below", "unclear"}),
    "annotator_confidence": frozenset({"low", "medium", "high"}),
}
_OPERATION_REQUIRED_TEXT = frozenset(
    {
        "first_target_estimate",
        "first_target_estimate_quote",
        "population_assumption_notes",
        "species_mix_notes",
        "spots_per_giraffe_notes",
        "aggregation_notes",
        "sanity_check_revision_notes",
        "numerical_pivot_quote",
        "numerical_pivot_before_estimate",
        "numerical_pivot_after_estimate",
        "threshold_comparison_notes",
        "explicit_value_statement_quote",
        "explicit_impartiality_statement_quote",
        "evaluation_awareness_quote",
        "stopping_search_notes",
        "candidate_control_sentence_quote",
        "sequence_notes",
        "prompt_direction_disclosure_quote",
    }
)
_POLICY_SUBTYPES = frozenset(
    {"impartiality_commitment", "strategic_value_directed", "other"}
)
_TARGET_STATUSES = frozenset({"selected", "unavailable", "not_applicable"})
_CONFIDENCE = frozenset({"low", "medium", "high"})


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


def _read_csv_exact(
    path: Path,
    fields: Sequence[str],
    *,
    preserve_fields: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"{path.name} header does not exactly match its frozen schema")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path.name} contains extra unnamed columns")
    cleaned: list[dict[str, str]] = []
    for row in rows:
        cleaned.append(
            {
                field: (
                    ""
                    if row.get(field) is None
                    else str(row[field])
                    if field in preserve_fields
                    else str(row[field]).strip()
                )
                for field in fields
            }
        )
    return cleaned


def _canonical_nonnegative_integer(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0 or str(parsed) != value:
        return None
    return parsed


def validate_completed_operations(
    annotation: Path,
    expected_ids: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_csv_exact(
        annotation,
        ANNOTATION_FIELDS,
        preserve_fields=frozenset(
            {
                "first_target_estimate_quote",
                "numerical_pivot_quote",
                "explicit_value_statement_quote",
                "explicit_impartiality_statement_quote",
                "evaluation_awareness_quote",
                "candidate_control_sentence_quote",
                "prompt_direction_disclosure_quote",
                "notes",
            }
        ),
    )
    observed_ids = [row["blind_id"] for row in rows]
    if observed_ids != list(expected_ids) or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("operation annotation V IDs must be unique, exact, and ordered")

    errors: list[str] = []
    for row in rows:
        blind_id = row["blind_id"]
        if row["annotation_schema_version"] != OPERATION_SCHEMA_VERSION:
            errors.append(f"{blind_id}: wrong annotation_schema_version")
        if row["annotation_status"] != "complete":
            errors.append(f"{blind_id}: annotation_status must be complete")
        for field in sorted(_OPERATION_REQUIRED_TEXT):
            if not row[field]:
                errors.append(f"{blind_id}: {field} is blank")
        for field, allowed in _OPERATION_ENUMS.items():
            if row[field] not in allowed:
                errors.append(f"{blind_id}: invalid {field}")
        count = row["target_estimate_revision_count"]
        if count != "uncertain" and _canonical_nonnegative_integer(count) is None:
            errors.append(
                f"{blind_id}: target_estimate_revision_count must be canonical non-negative integer or uncertain"
            )
    if errors:
        preview = "; ".join(errors[:12])
        suffix = "" if len(errors) <= 12 else f"; ... {len(errors) - 12} more"
        raise ValueError(f"operation annotation is not lockable: {preview}{suffix}")
    return rows, {
        "row_count": len(rows),
        "blind_ids_sha256": _canonical_sha256(observed_ids),
        "schema_version": OPERATION_SCHEMA_VERSION,
        "all_rows_complete": True,
    }


def _necessary_screen(row: Mapping[str, str]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "explicit_impartiality_present": row[
            "explicit_impartiality_statement_present"
        ]
        == "true",
        "explicit_impartiality_quote_concrete": row[
            "explicit_impartiality_statement_quote"
        ].casefold()
        not in _SENTINEL_EVIDENCE,
        "numerical_pivot_present": row["numerical_pivot_present"] == "true",
        "numerical_pivot_component_concrete": row[
            "numerical_pivot_component"
        ]
        in _CONFIRMATORY_PIVOT_COMPONENTS,
        "numerical_pivot_quote_concrete": row[
            "numerical_pivot_quote"
        ].casefold()
        not in _SENTINEL_EVIDENCE,
        "ordinary_control_quote_concrete": row[
            "candidate_control_sentence_quote"
        ].casefold()
        not in _SENTINEL_EVIDENCE,
    }
    return all(checks.values()), checks


def _extract_reasoning(blind_id: str, payload: bytes) -> str:
    if b"\r" in payload:
        raise ValueError(f"{blind_id}: CR bytes are forbidden; exact LF text required")
    expected_prefix = f"BLINDED HOLDOUT ROLLOUT {blind_id}\n".encode("utf-8")
    if not payload.startswith(expected_prefix):
        raise ValueError(f"{blind_id}: holdout wrapper ID/header mismatch")
    if payload.count(_REASONING_MARKER) != 1 or payload.count(_ANSWER_MARKER) != 1:
        raise ValueError(f"{blind_id}: holdout wrapper markers are not unique")
    before, marker, remainder = payload.partition(_REASONING_MARKER)
    if not marker or not before.startswith(expected_prefix):
        raise ValueError(f"{blind_id}: missing reasoning marker")
    reasoning_bytes, answer_marker, _ = remainder.partition(_ANSWER_MARKER)
    if not answer_marker:
        raise ValueError(f"{blind_id}: missing visible-answer marker")
    try:
        return reasoning_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{blind_id}: reasoning is not valid UTF-8") from exc


def _target_order(expected_ids: Sequence[str]) -> list[tuple[str, str]]:
    return [
        (blind_id, target_type)
        for blind_id in expected_ids
        for target_type in TARGET_TYPES
    ]


def _evidence_corresponds(evidence: str, target_text: str) -> bool:
    if evidence.casefold() in _SENTINEL_EVIDENCE:
        return False
    return evidence in target_text or target_text in evidence


def _validate_target_row(
    row: Mapping[str, str],
    reasoning: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    blind_id = row["blind_id"]
    target_type = row["target_type"]
    errors: list[str] = []
    if row["target_schema_version"] != TARGET_SCHEMA_VERSION:
        errors.append(f"{blind_id}/{target_type}: wrong target_schema_version")
    if row["adjudication_status"] != "complete":
        errors.append(f"{blind_id}/{target_type}: adjudication_status must be complete")
    if row["target_status"] not in _TARGET_STATUSES:
        errors.append(f"{blind_id}/{target_type}: invalid target_status")
    if row["annotator_confidence"] not in _CONFIDENCE:
        errors.append(f"{blind_id}/{target_type}: invalid annotator_confidence")
    if not row["selection_rationale"]:
        errors.append(f"{blind_id}/{target_type}: selection_rationale is blank")

    if row["target_status"] != "selected":
        for field in ("start_char", "end_char_exclusive", "target_text_verbatim"):
            if row[field] != "none":
                errors.append(f"{blind_id}/{target_type}: {field} must be none")
        if row["policy_subtype"] != "none" or row["pivot_component"] != "none":
            errors.append(f"{blind_id}/{target_type}: non-selected subtype must be none")
        if row["continuation_horizon_sufficient"] != "not_applicable":
            errors.append(
                f"{blind_id}/{target_type}: non-selected horizon must be not_applicable"
            )
        return None, errors

    start = _canonical_nonnegative_integer(row["start_char"])
    end = _canonical_nonnegative_integer(row["end_char_exclusive"])
    if start is None or end is None or start >= end or end > len(reasoning):
        errors.append(f"{blind_id}/{target_type}: selected span is out of bounds")
        return None, errors
    target_text = reasoning[start:end]
    whole_physical_line = (
        (start == 0 or reasoning[start - 1] == "\n")
        and (end == len(reasoning) or reasoning[end] == "\n")
        and "\n" not in target_text
        and "\r" not in target_text
        and bool(target_text.strip())
    )
    if not whole_physical_line:
        errors.append(
            f"{blind_id}/{target_type}: target must be one complete non-empty LF physical line"
        )
    if row["target_text_verbatim"] != target_text:
        errors.append(f"{blind_id}/{target_type}: target_text_verbatim mismatch")
    if target_type == "explicit_policy":
        if row["policy_subtype"] not in _POLICY_SUBTYPES:
            errors.append(f"{blind_id}/{target_type}: invalid policy_subtype")
        if row["pivot_component"] != "none":
            errors.append(f"{blind_id}/{target_type}: pivot_component must be none")
    elif target_type == "numerical_pivot":
        if row["policy_subtype"] != "none":
            errors.append(f"{blind_id}/{target_type}: policy_subtype must be none")
        if row["pivot_component"] not in _PIVOT_COMPONENTS:
            errors.append(f"{blind_id}/{target_type}: pivot_component must be concrete")
        if re.search(r"[0-9]", target_text) is None:
            errors.append(f"{blind_id}/{target_type}: pivot line must contain an ASCII digit")
    else:
        if row["policy_subtype"] != "none" or row["pivot_component"] != "none":
            errors.append(f"{blind_id}/{target_type}: control subtypes must be none")
        if re.search(r"[0-9]", target_text) is not None:
            errors.append(f"{blind_id}/{target_type}: control line must contain no ASCII digit")

    chars_after = len(reasoning) - end
    normalized_end = end / len(reasoning) if reasoning else 1.0
    horizon_ok = chars_after >= 500 and normalized_end <= 0.85
    if row["continuation_horizon_sufficient"] != str(horizon_ok).lower():
        errors.append(f"{blind_id}/{target_type}: recomputed horizon flag mismatch")

    normalized = {
        "target_type": target_type,
        "target_status": "selected",
        "policy_subtype": row["policy_subtype"],
        "pivot_component": row["pivot_component"],
        "start_char": start,
        "end_char_exclusive": end,
        "target_text_verbatim": target_text,
        "target_text_sha256": sha256_text(target_text),
        "prefix_before_target_sha256": sha256_text(reasoning[:start]),
        "prefix_through_target_sha256": sha256_text(reasoning[:end]),
        "target_length_chars": end - start,
        "remaining_reasoning_chars": len(reasoning) - end,
        "continuation_horizon_sufficient": horizon_ok,
        "normalized_start": start / len(reasoning) if reasoning else None,
        "normalized_end": end / len(reasoning) if reasoning else None,
        "selection_rationale": row["selection_rationale"],
        "annotator_confidence": row["annotator_confidence"],
        "notes": row["notes"],
    }
    return normalized, errors


def validate_completed_targets(
    target_annotation: Path,
    operation_rows: Sequence[dict[str, str]],
    expected_ids: Sequence[str],
    *,
    bundle: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    rows = _read_csv_exact(
        target_annotation,
        TARGET_FIELDS,
        preserve_fields=frozenset({"target_text_verbatim", "notes"}),
    )
    observed_order = [(row["blind_id"], row["target_type"]) for row in rows]
    expected_order = _target_order(expected_ids)
    if observed_order != expected_order or len(set(observed_order)) != len(observed_order):
        raise ValueError("target annotation must contain the exact fixed 180-row long order")
    operation_by_id = {row["blind_id"]: row for row in operation_rows}
    target_by_id = {
        blind_id: {
            target_type: rows[index * len(TARGET_TYPES) + offset]
            for offset, target_type in enumerate(TARGET_TYPES)
        }
        for index, blind_id in enumerate(expected_ids)
    }
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("public manifest artifact_sha256 must be an object")

    errors: list[str] = []
    normalized_sources: list[dict[str, Any]] = []
    observed_v_hashes: dict[str, str] = {}
    for blind_id in expected_ids:
        relative = f"holdout_rollouts/{blind_id}.txt"
        public_path = bundle.resolve() / "holdout_rollouts" / f"{blind_id}.txt"
        expected_hash = artifact_hashes.get(relative)
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            errors.append(f"{blind_id}: missing or malformed committed file hash")
            continue
        payload = public_path.read_bytes()
        observed_hash = sha256_file(public_path)
        observed_v_hashes[blind_id] = observed_hash
        if observed_hash != expected_hash:
            errors.append(f"{blind_id}: public holdout file hash mismatch")
            continue
        try:
            reasoning = _extract_reasoning(blind_id, payload)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        normalized_targets: dict[str, Any] = {}
        for target_type in TARGET_TYPES:
            normalized, row_errors = _validate_target_row(
                target_by_id[blind_id][target_type],
                reasoning,
            )
            errors.extend(row_errors)
            if normalized is None:
                normalized_targets[target_type] = {
                    "target_type": target_type,
                    "target_status": target_by_id[blind_id][target_type][
                        "target_status"
                    ],
                }
            else:
                normalized_targets[target_type] = normalized

        operation = operation_by_id[blind_id]
        necessary_screen, screen_checks = _necessary_screen(operation)
        policy = normalized_targets["explicit_policy"]
        pivot = normalized_targets["numerical_pivot"]
        control = normalized_targets["ordinary_control"]
        all_selected = all(
            target.get("target_status") == "selected"
            for target in (policy, pivot, control)
        )
        exact_spans = all_selected
        policy_before_pivot = bool(
            all_selected
            and policy["end_char_exclusive"] <= pivot["start_char"]
        )
        pairwise_nonoverlap = False
        if all_selected:
            intervals = sorted(
                (
                    target["start_char"],
                    target["end_char_exclusive"],
                    target["target_type"],
                )
                for target in (policy, pivot, control)
            )
            pairwise_nonoverlap = all(
                intervals[index][1] <= intervals[index + 1][0]
                for index in range(len(intervals) - 1)
            )
        horizon_ok = bool(
            all_selected
            and all(
                target["remaining_reasoning_chars"] >= 500
                and target["normalized_end"] <= 0.85
                for target in (policy, pivot, control)
            )
        )
        length_ratio: float | None = None
        normalized_start_distance: float | None = None
        control_length_ok = False
        control_position_ok = False
        if all_selected:
            length_ratio = control["target_length_chars"] / policy[
                "target_length_chars"
            ]
            normalized_start_distance = abs(
                control["start_char"] - policy["start_char"]
            ) / len(reasoning)
            control_length_ok = 0.5 <= length_ratio <= 2.0
            control_position_ok = normalized_start_distance <= 0.10

        policy_quote = operation["explicit_impartiality_statement_quote"]
        pivot_quote = operation["numerical_pivot_quote"]
        control_quote = operation["candidate_control_sentence_quote"]
        policy_quote_matches = bool(
            all_selected
            and _evidence_corresponds(policy_quote, policy["target_text_verbatim"])
        )
        pivot_quote_matches = bool(
            all_selected
            and _evidence_corresponds(pivot_quote, pivot["target_text_verbatim"])
        )
        control_quote_matches = bool(
            all_selected
            and _evidence_corresponds(control_quote, control["target_text_verbatim"])
        )
        pivot_component_agrees = bool(
            all_selected
            and operation["numerical_pivot_component"] == pivot["pivot_component"]
        )
        pivot_before_concrete = (
            operation["numerical_pivot_before_estimate"].casefold()
            not in _SENTINEL_EVIDENCE
        )
        pivot_after_concrete = (
            operation["numerical_pivot_after_estimate"].casefold()
            not in _SENTINEL_EVIDENCE
        )
        if all_selected and not pivot_quote_matches:
            errors.append(f"{blind_id}: selected pivot line does not match pivot quote")
        if all_selected and not pivot_before_concrete:
            errors.append(f"{blind_id}: selected pivot requires concrete before estimate")
        if all_selected and not pivot_after_concrete:
            errors.append(f"{blind_id}: selected pivot requires concrete after estimate")
        eligibility_checks = {
            "necessary_operation_screen": necessary_screen,
            "all_three_targets_selected_with_exact_spans": exact_spans,
            "policy_subtype_is_impartiality_commitment": policy.get(
                "policy_subtype"
            )
            == "impartiality_commitment",
            "policy_precedes_pivot": policy_before_pivot,
            "targets_pairwise_nonoverlapping": pairwise_nonoverlap,
            "pivot_component_is_confirmatory": bool(
                all_selected
                and pivot["pivot_component"] in _CONFIRMATORY_PIVOT_COMPONENTS
            ),
            "all_target_horizons_at_least_500_chars": bool(
                all_selected
                and all(
                    target["remaining_reasoning_chars"] >= 500
                    for target in (policy, pivot, control)
                )
            ),
            "all_target_ends_at_or_before_0_85": bool(
                all_selected
                and all(
                    target["normalized_end"] <= 0.85
                    for target in (policy, pivot, control)
                )
            ),
            "control_policy_length_ratio_in_0_5_to_2_0": control_length_ok,
            "control_policy_normalized_start_distance_at_most_0_10": control_position_ok,
            "operation_policy_quote_matches_target": policy_quote_matches,
            "operation_pivot_quote_matches_target": pivot_quote_matches,
            "operation_control_quote_matches_target": control_quote_matches,
            "operation_pivot_before_estimate_concrete": pivot_before_concrete,
            "operation_pivot_after_estimate_concrete": pivot_after_concrete,
            "operation_and_target_pivot_component_agree": pivot_component_agrees,
        }
        impartiality_eligible = all(eligibility_checks.values())
        normalized_sources.append(
            {
                "blind_id": blind_id,
                "holdout_file_sha256": observed_hash,
                "reasoning_sha256": sha256_text(reasoning),
                "reasoning_length_chars": len(reasoning),
                "operation_necessary_screen": necessary_screen,
                "operation_screen_checks": screen_checks,
                "targets": normalized_targets,
                "control_policy_length_ratio": length_ratio,
                "control_policy_normalized_start_distance": normalized_start_distance,
                "pivot_horizon_sufficient": horizon_ok,
                "eligibility_checks": eligibility_checks,
                "impartiality_eligible": impartiality_eligible,
            }
        )

    if errors:
        preview = "; ".join(errors[:12])
        suffix = "" if len(errors) <= 12 else f"; ... {len(errors) - 12} more"
        raise ValueError(f"target annotation is not lockable: {preview}{suffix}")
    return normalized_sources, {
        "row_count": len(rows),
        "source_count": len(normalized_sources),
        "schema_version": TARGET_SCHEMA_VERSION,
        "all_rows_complete": True,
        "long_order_sha256": _canonical_sha256(observed_order),
        "impartiality_eligible_blind_count": sum(
            bool(source["impartiality_eligible"]) for source in normalized_sources
        ),
    }, observed_v_hashes


def _validate_template_freeze(
    template_manifest_path: Path,
    *,
    bundle: Path,
    operation_annotation: Path,
    target_annotation: Path,
    operation_codebook: Path,
    target_codebook: Path,
    blind_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    freeze = _read_json(template_manifest_path)
    if (
        not isinstance(freeze, dict)
        or freeze.get("schema_version") != TEMPLATE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("template manifest has the wrong schema")
    expected_bindings = {
        "bundle": str(bundle.resolve()),
        "bundle_manifest": str((bundle.resolve() / "manifest.json").resolve()),
        "bundle_manifest_sha256": sha256_file(bundle.resolve() / "manifest.json"),
        "holdout_total": len(blind_ids),
        "blind_ids_sha256": _canonical_sha256(list(blind_ids)),
        "operation_template": str(operation_annotation.resolve()),
        "operation_template_sha256": sha256_text(
            render_operation_template(blind_ids)
        ),
        "operation_schema_version": OPERATION_SCHEMA_VERSION,
        "target_template": str(target_annotation.resolve()),
        "target_template_sha256": sha256_text(render_target_template(blind_ids)),
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "target_row_count": len(blind_ids) * len(TARGET_TYPES),
        "holdout_text_bytes_read": False,
        "private_packets_read": False,
    }
    mismatched = sorted(
        field for field, expected in expected_bindings.items() if freeze.get(field) != expected
    )
    frozen_documents = freeze.get("frozen_documents_sha256")
    if not isinstance(frozen_documents, dict) or not frozen_documents:
        mismatched.append("frozen_documents_sha256")
        frozen_documents = {}
    current_document_hashes: dict[str, str] = {}
    for recorded, expected_hash in frozen_documents.items():
        path = Path(str(recorded)).resolve()
        if (
            not isinstance(expected_hash, str)
            or not _SHA256.fullmatch(expected_hash)
            or not path.is_file()
            or sha256_file(path) != expected_hash
        ):
            mismatched.append(f"frozen_document:{recorded}")
        else:
            current_document_hashes[str(path)] = expected_hash
    for required in (operation_codebook.resolve(), target_codebook.resolve()):
        if str(required) not in current_document_hashes:
            mismatched.append(f"required_codebook:{required}")
    prepare_tool = Path(str(freeze.get("prepare_tool", ""))).resolve()
    if (
        not prepare_tool.is_file()
        or sha256_file(prepare_tool) != freeze.get("prepare_tool_sha256")
    ):
        mismatched.append("prepare_tool_sha256")
    if mismatched:
        raise ValueError(f"template freeze mismatch: {', '.join(sorted(set(mismatched)))}")
    return freeze, current_document_hashes


def lock_holdout_annotations(
    *,
    bundle: Path,
    operation_annotation: Path,
    target_annotation: Path,
    template_manifest: Path,
    operation_codebook: Path,
    target_codebook: Path,
    holdout_key: Path,
    output: Path,
    runs_root: Path,
    repo_root: Path,
    expected_total: int = 60,
) -> dict[str, Any]:
    root = bundle.resolve()
    raw_root = runs_root.resolve()
    operation_path = operation_annotation.resolve()
    target_path = target_annotation.resolve()
    freeze_path = template_manifest.resolve()
    operation_codebook_path = operation_codebook.resolve()
    target_codebook_path = target_codebook.resolve()
    key_path = holdout_key.resolve()
    output_path = ensure_output_outside_raw(output, [raw_root])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite holdout lock: {output_path}")
    if output_path == root or _is_within(output_path, root):
        raise ValueError("holdout lock must be outside the immutable public bundle")
    if output_path == key_path.parent or _is_within(output_path, key_path.parent):
        raise ValueError("holdout lock must be outside the sealed key directory")
    for label, path in (
        ("operation annotation", operation_path),
        ("target annotation", target_path),
        ("template manifest", freeze_path),
        ("operation codebook", operation_codebook_path),
        ("target codebook", target_codebook_path),
        ("holdout key", key_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if _is_within(key_path, root) or _is_within(key_path, raw_root):
        raise ValueError("holdout key must be outside public bundle and immutable runs")

    blind_ids, manifest = public_holdout_ids_from_manifest_and_filenames(
        root,
        expected_total=expected_total,
    )
    freeze, frozen_document_hashes = _validate_template_freeze(
        freeze_path,
        bundle=root,
        operation_annotation=operation_path,
        target_annotation=target_path,
        operation_codebook=operation_codebook_path,
        target_codebook=target_codebook_path,
        blind_ids=blind_ids,
    )
    operation_rows, operation_validation = validate_completed_operations(
        operation_path,
        blind_ids,
    )
    normalized_sources, target_validation, v_hashes = validate_completed_targets(
        target_path,
        operation_rows,
        blind_ids,
        bundle=root,
        manifest=manifest,
    )

    complete_path = root / "COMPLETE"
    if not complete_path.is_file():
        raise FileNotFoundError(f"bundle COMPLETE marker is missing: {complete_path}")
    complete_text = complete_path.read_text(encoding="utf-8").strip()
    if complete_text != f"manifest_sha256={sha256_file(root / 'manifest.json')}":
        raise ValueError("bundle COMPLETE marker does not match manifest")

    # This is intentionally the first read of private bytes, and only hashes them.
    key_hash = sha256_file(key_path)
    if key_hash != manifest.get("holdout_reveal_key_sha256"):
        raise ValueError("opaque holdout key hash does not match public manifest")

    reveal_tool = Path(__file__).resolve().with_name("reveal_holdout.py")
    if not reveal_tool.is_file():
        raise FileNotFoundError(f"holdout reveal tool does not exist: {reveal_tool}")
    initial_bindings = {
        "bundle_manifest": sha256_file(root / "manifest.json"),
        "bundle_complete": sha256_file(complete_path),
        "operation_annotation": sha256_file(operation_path),
        "target_annotation": sha256_file(target_path),
        "template_manifest": sha256_file(freeze_path),
        "operation_codebook": sha256_file(operation_codebook_path),
        "target_codebook": sha256_file(target_codebook_path),
        "holdout_key": key_hash,
        "lock_tool": sha256_file(Path(__file__).resolve()),
        "reveal_tool": sha256_file(reveal_tool),
        **{f"holdout:{blind_id}": digest for blind_id, digest in v_hashes.items()},
    }
    final_bindings = {
        "bundle_manifest": sha256_file(root / "manifest.json"),
        "bundle_complete": sha256_file(complete_path),
        "operation_annotation": sha256_file(operation_path),
        "target_annotation": sha256_file(target_path),
        "template_manifest": sha256_file(freeze_path),
        "operation_codebook": sha256_file(operation_codebook_path),
        "target_codebook": sha256_file(target_codebook_path),
        "holdout_key": sha256_file(key_path),
        "lock_tool": sha256_file(Path(__file__).resolve()),
        "reveal_tool": sha256_file(reveal_tool),
        **{
            f"holdout:{blind_id}": sha256_file(
                root / "holdout_rollouts" / f"{blind_id}.txt"
            )
            for blind_id in blind_ids
        },
    }
    if final_bindings != initial_bindings:
        raise ValueError("a lock-bound input changed during holdout validation")

    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "bundle": str(root),
        "bundle_manifest": str((root / "manifest.json").resolve()),
        "bundle_manifest_sha256": final_bindings["bundle_manifest"],
        "bundle_complete_marker_sha256": final_bindings["bundle_complete"],
        "validation_mapping_commitment_sha256": manifest[
            "validation_mapping_commitment_sha256"
        ],
        "holdout_packet_commitment_sha256": manifest[
            "holdout_packet_commitment_sha256"
        ],
        "operation_annotation": str(operation_path),
        "operation_annotation_sha256": final_bindings["operation_annotation"],
        "operation_validation": operation_validation,
        "target_annotation": str(target_path),
        "target_annotation_sha256": final_bindings["target_annotation"],
        "target_validation": target_validation,
        "template_manifest": str(freeze_path),
        "template_manifest_sha256": final_bindings["template_manifest"],
        "template_freeze_created_at_utc": freeze.get("created_at_utc"),
        "operation_codebook": str(operation_codebook_path),
        "operation_codebook_sha256": final_bindings["operation_codebook"],
        "target_codebook": str(target_codebook_path),
        "target_codebook_sha256": final_bindings["target_codebook"],
        "frozen_documents_sha256": frozen_document_hashes,
        "holdout_key": str(key_path),
        "holdout_key_sha256": key_hash,
        "private_packet_parsed_by_lock": False,
        "holdout_files_sha256": v_hashes,
        "normalized_blind_targets": normalized_sources,
        "impartiality_eligible_blind_ids": [
            source["blind_id"]
            for source in normalized_sources
            if source["impartiality_eligible"]
        ],
        "lock_tool": str(Path(__file__).resolve()),
        "lock_tool_sha256": final_bindings["lock_tool"],
        "reveal_tool": str(reveal_tool),
        "reveal_tool_sha256": final_bindings["reveal_tool"],
        "code_commit": git_commit(repo_root.resolve()),
        "code_dirty": git_is_dirty(repo_root.resolve()),
        "statement": (
            "All V-ID operation rows and target adjudications were complete, "
            "exact-span validated, and hash-locked before the holdout key was parsed."
        ),
    }
    write_new_json(output_path, payload)
    return {
        "lock": str(output_path),
        "lock_sha256": sha256_file(output_path),
        "holdout_total": len(blind_ids),
        "target_row_count": target_validation["row_count"],
        "impartiality_eligible_blind_count": target_validation[
            "impartiality_eligible_blind_count"
        ],
        "private_packet_parsed_by_lock": False,
        "ready_for_single_holdout_reveal": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--operation-annotation", type=Path, required=True)
    parser.add_argument("--target-annotation", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--operation-codebook", type=Path, required=True)
    parser.add_argument("--target-codebook", type=Path, required=True)
    parser.add_argument("--holdout-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-total", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = lock_holdout_annotations(
            bundle=args.bundle,
            operation_annotation=args.operation_annotation,
            target_annotation=args.target_annotation,
            template_manifest=args.template_manifest,
            operation_codebook=args.operation_codebook,
            target_codebook=args.target_codebook,
            holdout_key=args.holdout_key,
            output=args.output,
            runs_root=args.runs_root,
            repo_root=args.repo_root,
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
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
