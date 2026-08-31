"""Create a reproducible condition-metadata-blinded discovery/holdout split.

The public discovery artifacts omit condition labels, source rollout IDs, raw
prompts, and trajectory-judge outputs.  A sealed JSON key retains the exact
mapping and random seed for later reveal.  No API or model is called.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

from research.experiment_utils import (
    ensure_output_outside_raw,
    git_commit,
    git_is_dirty,
    sha256_file,
    sha256_text,
    utc_now,
    write_new_json,
    write_new_text,
)


SCHEMA_VERSION = "value-leakage.metadata-blinded-split/v2"
ANNOTATION_SCHEMA_VERSION = "value-leakage.discovery-annotation/v0.1"
CONDITIONS = ("above_good", "below_good")
DEFAULT_NAMESPACE = "vl-qwen122b-giraffes-v1"

ANNOTATION_FIELDS = (
    "annotation_schema_version",
    "blind_id",
    "annotation_status",
    "first_target_estimate",
    "first_target_estimate_quote",
    "first_target_estimate_threshold_side",
    "population_assumption_notes",
    "species_mix_notes",
    "spots_per_giraffe_notes",
    "aggregation_notes",
    "sanity_check_revision_notes",
    "numerical_pivot_present",
    "numerical_pivot_component",
    "numerical_pivot_quote",
    "numerical_pivot_before_estimate",
    "numerical_pivot_after_estimate",
    "revision_direction",
    "target_estimate_revision_count",
    "threshold_comparison_present",
    "threshold_comparison_notes",
    "explicit_value_statement_present",
    "explicit_value_statement_quote",
    "explicit_impartiality_statement_present",
    "explicit_impartiality_statement_quote",
    "evaluation_awareness_present",
    "evaluation_awareness_quote",
    "continued_search_after_first_estimate",
    "continued_search_after_value_statement",
    "stopping_search_notes",
    "candidate_control_sentence_quote",
    "sequence_notes",
    "prompt_direction_disclosure_in_text",
    "disclosed_favored_side",
    "prompt_direction_disclosure_quote",
    "annotator_confidence",
    "notes",
)


@dataclass(frozen=True)
class RawRecord:
    condition: str
    rollout_id: int
    row_position: int
    reasoning: str
    content: str

    @property
    def canonical_key(self) -> str:
        return f"{self.condition}:{self.rollout_id:03d}"


@dataclass(frozen=True)
class BlindRecord:
    blind_id: str
    raw: RawRecord


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _score(namespace: str, seed: int, phase: str, record: RawRecord) -> str:
    material = (
        f"{namespace}|{seed}|{phase}|{record.condition}|{record.rollout_id:03d}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_paths(
    run_dir: Path,
    output_dir: Path,
    annotation_output: Path,
    sealed_output_dir: Path,
    runs_root: Path,
) -> tuple[Path, Path, Path]:
    root = runs_root.resolve()
    run = run_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {root}")
    if run == root or not _is_within(run, root):
        raise ValueError(f"source run {run} is not inside runs root {root}")
    output = ensure_output_outside_raw(output_dir, [root, run])
    annotation = ensure_output_outside_raw(annotation_output, [root, run])
    sealed = ensure_output_outside_raw(sealed_output_dir, [root, run])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if annotation.exists():
        raise FileExistsError(f"refusing to overwrite annotation file: {annotation}")
    if sealed.exists():
        raise FileExistsError(f"refusing to overwrite private key directory: {sealed}")
    if _is_within(annotation, output):
        raise ValueError(
            "annotation output must be outside the regenerable blinded bundle"
        )
    if _is_within(sealed, output) or _is_within(output, sealed):
        raise ValueError("private key directory must be separate from the public bundle")
    if annotation == sealed or _is_within(annotation, sealed):
        raise ValueError("annotation and private key outputs must be separate")
    return output, annotation, sealed


def _valid_row(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and not row.get("error")
        and isinstance(row.get("reasoning"), str)
        and bool(row["reasoning"].strip())
        and isinstance(row.get("content"), str)
        and bool(row["content"].strip())
    )


def load_eligible_records(
    run_dir: Path,
) -> tuple[dict[str, list[RawRecord]], dict[str, Any]]:
    run = run_dir.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"source run does not exist: {run}")

    threshold_payload = _read_json(run / "threshold.json")
    threshold = threshold_payload.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold.json must contain a numeric threshold")
    config = _read_json(run / "config.json")
    if not isinstance(config, dict):
        raise ValueError("config.json must be an object")

    by_condition: dict[str, list[RawRecord]] = {}
    audit: dict[str, Any] = {
        "threshold": threshold,
        "model": config.get("model"),
        "model_id": config.get("model_id"),
        "backend": config.get("backend"),
        "provider": config.get("provider"),
        "conditions": {},
    }
    for condition in CONDITIONS:
        path = run / f"{condition}.json"
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("condition") != condition:
            raise ValueError(f"{path.name} has a missing or mismatched condition label")
        if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
            raise ValueError(f"{path.name} must contain a non-empty exact prompt")
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"{path.name} must contain a rows list")

        integer_ids = [
            row.get("i")
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("i"), int)
            and not isinstance(row.get("i"), bool)
            and row["i"] >= 0
        ]
        duplicate_ids = sorted(
            rollout_id
            for rollout_id in set(integer_ids)
            if integer_ids.count(rollout_id) > 1
        )
        if duplicate_ids:
            raise ValueError(
                f"{path.name} contains duplicate rollout IDs: {duplicate_ids[:10]}"
            )
        seen_ids: set[int] = set()
        eligible: list[RawRecord] = []
        invalid_counts = {
            "not_an_object": 0,
            "api_error": 0,
            "empty_reasoning": 0,
            "empty_content": 0,
            "invalid_or_duplicate_rollout_id": 0,
        }
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                invalid_counts["not_an_object"] += 1
                continue
            rollout_id = row.get("i")
            if (
                isinstance(rollout_id, bool)
                or not isinstance(rollout_id, int)
                or rollout_id < 0
            ):
                invalid_counts["invalid_or_duplicate_rollout_id"] += 1
                continue
            seen_ids.add(rollout_id)
            if row.get("error"):
                invalid_counts["api_error"] += 1
            if not isinstance(row.get("reasoning"), str) or not row.get(
                "reasoning", ""
            ).strip():
                invalid_counts["empty_reasoning"] += 1
            if not isinstance(row.get("content"), str) or not row.get(
                "content", ""
            ).strip():
                invalid_counts["empty_content"] += 1
            if not _valid_row(row):
                continue
            eligible.append(
                RawRecord(
                    condition=condition,
                    rollout_id=rollout_id,
                    row_position=position,
                    reasoning=row["reasoning"],
                    content=row["content"],
                )
            )

        by_condition[condition] = eligible
        audit["conditions"][condition] = {
            "raw_rows": len(rows),
            "unique_integer_rollout_ids": len(seen_ids),
            "eligible": len(eligible),
            "ineligibility_observations": invalid_counts,
            "source_sha256": sha256_file(path),
            "finish_reason_counts_among_eligible": _counts(
                row.get("finish_reason")
                for row in rows
                if _valid_row(row)
            ),
        }
    return by_condition, audit


def _counts(values: Sequence[Any] | Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _public_source_audit(source_audit: dict[str, Any]) -> dict[str, Any]:
    arms = list(source_audit["conditions"].values())
    ineligibility_totals: dict[str, int] = {}
    finish_totals: dict[str, int] = {}
    for arm in arms:
        for key, value in arm["ineligibility_observations"].items():
            ineligibility_totals[key] = ineligibility_totals.get(key, 0) + value
        for key, value in arm["finish_reason_counts_among_eligible"].items():
            finish_totals[key] = finish_totals.get(key, 0) + value
    return {
        "raw_rows_total": sum(arm["raw_rows"] for arm in arms),
        "eligible_total": sum(arm["eligible"] for arm in arms),
        "hidden_arm_eligible_counts_sorted": sorted(arm["eligible"] for arm in arms),
        "ineligibility_observations_total": dict(sorted(ineligibility_totals.items())),
        "finish_reason_counts_among_eligible": dict(sorted(finish_totals.items())),
    }


def allocate_split(
    by_condition: dict[str, list[RawRecord]],
    *,
    discovery_seed: int,
    validation_seed: int,
    namespace: str,
    discovery_per_condition: int,
    validation_per_condition: int,
) -> tuple[list[BlindRecord], list[BlindRecord], list[RawRecord]]:
    for label, seed in (
        ("discovery_seed", discovery_seed),
        ("validation_seed", validation_seed),
    ):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if discovery_seed == validation_seed:
        raise ValueError("discovery and validation seeds must be different")
    if not namespace.strip():
        raise ValueError("namespace must be non-empty")
    if discovery_per_condition < 1 or validation_per_condition < 1:
        raise ValueError("discovery and validation counts must be positive")

    discovery_raw: list[RawRecord] = []
    validation_raw: list[RawRecord] = []
    reserve: list[RawRecord] = []
    for condition in CONDITIONS:
        eligible = list(by_condition.get(condition, []))
        required = discovery_per_condition + validation_per_condition
        if len(eligible) < required:
            raise ValueError(
                f"{condition} has {len(eligible)} eligible rows, need {required}"
            )
        discovery_ranked = sorted(
            eligible,
            key=lambda record: (
                _score(namespace, discovery_seed, "discovery-sample", record),
                record.rollout_id,
            ),
        )
        chosen_discovery = discovery_ranked[:discovery_per_condition]
        discovery_keys = {record.canonical_key for record in chosen_discovery}
        remaining = [
            record
            for record in eligible
            if record.canonical_key not in discovery_keys
        ]
        validation_ranked = sorted(
            remaining,
            key=lambda record: (
                _score(namespace, validation_seed, "validation-sample", record),
                record.rollout_id,
            ),
        )
        discovery_raw.extend(chosen_discovery)
        validation_raw.extend(validation_ranked[:validation_per_condition])
        reserve.extend(validation_ranked[validation_per_condition:])

    discovery_order = sorted(
        discovery_raw,
        key=lambda record: (
            _score(namespace, discovery_seed, "discovery-display", record),
            record.canonical_key,
        ),
    )
    validation_order = sorted(
        validation_raw,
        key=lambda record: (
            _score(namespace, validation_seed, "validation-display", record),
            record.canonical_key,
        ),
    )
    discovery = [
        BlindRecord(f"D{index:03d}", record)
        for index, record in enumerate(discovery_order, start=1)
    ]
    validation = [
        BlindRecord(f"V{index:03d}", record)
        for index, record in enumerate(validation_order, start=1)
    ]
    return discovery, validation, reserve


def _mapping(records: Sequence[BlindRecord]) -> list[dict[str, Any]]:
    return [
        {
            "blind_id": item.blind_id,
            "condition": item.raw.condition,
            "rollout_id": item.raw.rollout_id,
            "row_position": item.raw.row_position,
            "reasoning_sha256": sha256_text(item.raw.reasoning),
            "visible_final_answer_sha256": sha256_text(item.raw.content),
        }
        for item in records
    ]


def _reserve_mapping(records: Sequence[RawRecord]) -> list[dict[str, Any]]:
    return [
        {
            "condition": item.condition,
            "rollout_id": item.rollout_id,
            "row_position": item.row_position,
            "reasoning_sha256": sha256_text(item.reasoning),
            "visible_final_answer_sha256": sha256_text(item.content),
        }
        for item in records
    ]


def render_rollout(
    record: BlindRecord,
    threshold: float,
    *,
    phase: str = "DISCOVERY",
) -> str:
    threshold_text = (
        f"{int(threshold):,}" if float(threshold).is_integer() else f"{threshold:,}"
    )
    return "\n".join(
        (
            f"BLINDED {phase.upper()} ROLLOUT {record.blind_id}",
            "",
            f"SHARED THRESHOLD: {threshold_text}",
            "",
            "The source condition, prompt, rollout ID, and trajectory-judge output are withheld.",
            "The model text below is verbatim and may itself disclose the donation direction.",
            "",
            "===== FULL REASONING (VERBATIM) =====",
            "",
            record.raw.reasoning,
            "",
            "===== VISIBLE FINAL ANSWER (VERBATIM) =====",
            "",
            record.raw.content,
            "",
        )
    )


def render_annotation_csv(records: Sequence[BlindRecord]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = {field: "" for field in ANNOTATION_FIELDS}
        row["annotation_schema_version"] = ANNOTATION_SCHEMA_VERSION
        row["blind_id"] = record.blind_id
        writer.writerow(row)
    return stream.getvalue()


def _bundle_readme(
    records: Sequence[BlindRecord],
    annotation_path: Path,
    repo_root: Path,
) -> str:
    try:
        annotation_label = annotation_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        annotation_label = str(annotation_path.resolve())
    links = "\n".join(
        f"- [{record.blind_id}](rollouts/{record.blind_id}.txt)" for record in records
    )
    return f"""# Metadata-blinded Qwen discovery set

Read the {len(records)} rollout files in the fixed order below and fill `{annotation_label}`.
Do not search the raw run for matching text or infer labels from source IDs.
The private reveal keys are physically outside this public bundle. Complete operation
annotations before filling the prompt-direction self-disclosure fields.

This is condition-**metadata** blinding only. Model-authored text is verbatim and
may explicitly reveal the donation direction. Do not redact it and do not label
the rollout as motivated or unbiased during this phase.

Codebook: `experiments/E02_trajectory_localization/annotation_codebook_v0.1.md`

The precommitted holdout files are under `holdout_rollouts/`. Do not open them
until discovery annotations are locked and discovery hypotheses plus the
analysis plan are frozen. Their condition mapping remains private until the
holdout annotations are separately locked.

## Fixed reading order

{links}
"""


def create_bundle(
    *,
    run_dir: Path,
    output_dir: Path,
    annotation_output: Path,
    sealed_output_dir: Path,
    runs_root: Path,
    discovery_seed: int,
    validation_seed: int,
    namespace: str,
    discovery_per_condition: int,
    validation_per_condition: int,
    repo_root: Path,
) -> dict[str, Any]:
    output, annotation, sealed_output = _validate_output_paths(
        run_dir,
        output_dir,
        annotation_output,
        sealed_output_dir,
        runs_root,
    )
    source_files = [
        run_dir.resolve() / "config.json",
        run_dir.resolve() / "threshold.json",
        *(run_dir.resolve() / f"{condition}.json" for condition in CONDITIONS),
    ]
    source_hashes_before = {str(path.resolve()): sha256_file(path) for path in source_files}
    by_condition, source_audit = load_eligible_records(run_dir)
    discovery, validation, reserve = allocate_split(
        by_condition,
        discovery_seed=discovery_seed,
        validation_seed=validation_seed,
        namespace=namespace,
        discovery_per_condition=discovery_per_condition,
        validation_per_condition=validation_per_condition,
    )
    discovery_mapping = _mapping(discovery)
    validation_mapping = _mapping(validation)
    reserve_mapping = _reserve_mapping(reserve)
    discovery_commitment = _payload_sha256(discovery_mapping)
    validation_commitment = _payload_sha256(validation_mapping)
    created_at = utc_now()
    source_snapshot = {
        path.name: source_hashes_before[str(path.resolve())]
        for path in source_files
    }
    discovery_packet_core = {
        "schema_version": SCHEMA_VERSION,
        "warning": "OPEN ONLY AFTER ALL DISCOVERY ANNOTATIONS ARE COMPLETE",
        "created_at_utc": created_at,
        "packet_nonce": secrets.token_hex(32),
        "source_run": str(run_dir.resolve()),
        "source_files_sha256": source_snapshot,
        "selection": {
            "algorithm": "per-condition SHA-256 ranking and independent SHA-256 display ranking",
            "namespace": namespace,
            "discovery_seed": discovery_seed,
            "discovery_per_condition": discovery_per_condition,
        },
        "discovery_mapping": discovery_mapping,
        "discovery_mapping_commitment_sha256": discovery_commitment,
        "validation_mapping_commitment_sha256": validation_commitment,
    }
    discovery_packet_commitment = _payload_sha256(discovery_packet_core)
    discovery_reveal_payload = {
        **discovery_packet_core,
        "packet_commitment_sha256": discovery_packet_commitment,
    }
    holdout_packet_core = {
        "schema_version": SCHEMA_VERSION,
        "warning": "OPEN ONLY AFTER ALL HOLDOUT ANNOTATIONS ARE COMPLETE AND LOCKED",
        "created_at_utc": created_at,
        "packet_nonce": secrets.token_hex(32),
        "source_run": str(run_dir.resolve()),
        "source_files_sha256": source_snapshot,
        "selection": {
            "algorithm": "remaining-pool per-condition SHA-256 ranking and independent display ranking",
            "namespace": namespace,
            "validation_seed": validation_seed,
            "validation_per_condition": validation_per_condition,
        },
        "discovery_mapping_commitment_sha256": discovery_commitment,
        "validation_mapping": validation_mapping,
        "validation_mapping_commitment_sha256": validation_commitment,
        "reserve_mapping": reserve_mapping,
        "reserve_order": "precommitted validation-ranking remainder within each arm",
    }
    holdout_packet_commitment = _payload_sha256(holdout_packet_core)
    holdout_reveal_payload = {
        **holdout_packet_core,
        "packet_commitment_sha256": holdout_packet_commitment,
    }

    output.mkdir(parents=True, exist_ok=False)
    rollouts_dir = output / "rollouts"
    holdout_rollouts_dir = output / "holdout_rollouts"
    rollouts_dir.mkdir()
    holdout_rollouts_dir.mkdir()
    sealed_output.mkdir(parents=True, exist_ok=False)

    written: list[Path] = []
    for item in discovery:
        path = write_new_text(
            rollouts_dir / f"{item.blind_id}.txt",
            render_rollout(item, source_audit["threshold"]),
        )
        written.append(path)
    for item in validation:
        path = write_new_text(
            holdout_rollouts_dir / f"{item.blind_id}.txt",
            render_rollout(item, source_audit["threshold"], phase="HOLDOUT"),
        )
        written.append(path)

    readme_path = write_new_text(
        output / "README.md", _bundle_readme(discovery, annotation, repo_root.resolve())
    )
    discovery_reveal_path = write_new_json(
        sealed_output / "discovery_reveal_key.json", discovery_reveal_payload
    )
    holdout_reveal_path = write_new_json(
        sealed_output / "HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json",
        holdout_reveal_payload,
    )
    private_notice = write_new_text(
        output / "PRIVATE_KEYS.md",
        "# Private reveal keys\n\n"
        "No reveal key is stored in this public bundle. The public manifest binds "
        "both external key files by SHA-256 and full-packet commitment.\n\n"
        f"Lock all {discovery[0].blind_id}–{discovery[-1].blind_id} annotations before "
        "running discovery reveal. Do not reveal the holdout mapping until every "
        f"{validation[0].blind_id}–{validation[-1].blind_id} annotation is separately locked.\n",
    )
    annotation_path = write_new_text(annotation, render_annotation_csv(discovery))

    script_path = Path(__file__).resolve()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_run": str(run_dir.resolve()),
        "source_model": source_audit["model_id"] or source_audit["model"],
        "backend": source_audit["backend"],
        "provider": source_audit["provider"],
        "code_commit": git_commit(repo_root),
        "code_dirty": git_is_dirty(repo_root),
        "generator": str(script_path),
        "generator_sha256": sha256_file(script_path),
        "source_files_sha256": {
            "config.json": source_hashes_before[str((run_dir.resolve() / "config.json"))],
            "threshold.json": source_hashes_before[
                str((run_dir.resolve() / "threshold.json"))
            ],
            "hidden_intervention_files_sorted": sorted(
                source_hashes_before[str((run_dir.resolve() / f"{condition}.json"))]
                for condition in CONDITIONS
            ),
        },
        "eligibility_rule": "row object AND no error AND non-empty reasoning AND non-empty visible content",
        "source_audit": _public_source_audit(source_audit),
        "selection_algorithm": "per-condition SHA-256 ranking",
        "selection_seed": "withheld in external private key until the relevant lock",
        "discovery_total": len(discovery),
        "discovery_per_condition": discovery_per_condition,
        "validation_total_precommitted": len(validation),
        "validation_per_condition": validation_per_condition,
        "reserve_total": len(reserve),
        "private_keys_in_public_bundle": False,
        "holdout_rollouts_precommitted": True,
        "condition_metadata_hidden": True,
        "semantic_condition_blinding_guaranteed": False,
        "limitations": [
            "Model-authored reasoning can explicitly disclose the prompted donation direction.",
            "Visible text is observational evidence, not access to hidden computation.",
            "Discovery counts are exploratory and are not causal estimates.",
        ],
    }
    provenance_path = write_new_json(output / "provenance.json", provenance)
    written.extend(
        (
            readme_path,
            private_notice,
            provenance_path,
        )
    )
    source_hashes_after = {str(path.resolve()): sha256_file(path) for path in source_files}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("immutable source files changed during bundle generation")

    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "discovery_total": len(discovery),
        "discovery_per_hidden_condition": discovery_per_condition,
        "validation_total_precommitted": len(validation),
        "validation_per_hidden_condition": validation_per_condition,
        "holdout_rollout_total": len(validation),
        "reserve_total": len(reserve),
        "eligibility_total": sum(
            source_audit["conditions"][condition]["eligible"]
            for condition in CONDITIONS
        ),
        "hidden_arm_eligible_counts_sorted": sorted(
            source_audit["conditions"][condition]["eligible"]
            for condition in CONDITIONS
        ),
        "source_immutability_verified": True,
        "discovery_mapping_commitment_sha256": discovery_commitment,
        "validation_mapping_commitment_sha256": validation_commitment,
        "discovery_packet_commitment_sha256": discovery_packet_commitment,
        "holdout_packet_commitment_sha256": holdout_packet_commitment,
        "discovery_reveal_key_sha256": sha256_file(discovery_reveal_path),
        "holdout_reveal_key_sha256": sha256_file(holdout_reveal_path),
        "private_keys_in_public_bundle": False,
        "holdout_rollouts_precommitted": True,
        "annotation_template": str(annotation_path.resolve()),
        "annotation_template_sha256": sha256_file(annotation_path),
        "artifact_sha256": {
            path.resolve().relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(written)
        },
        "blinding": {
            "condition_label_hidden": True,
            "raw_prompt_hidden": True,
            "source_rollout_id_hidden": True,
            "trajectory_judge_output_hidden": True,
            "model_text_redacted": False,
            "semantic_condition_blinding_guaranteed": False,
        },
    }
    manifest_path = write_new_json(output / "manifest.json", public_manifest)
    complete_path = write_new_text(
        output / "COMPLETE",
        f"manifest_sha256={sha256_file(manifest_path)}\n",
    )
    return {
        "bundle": str(output),
        "annotation": str(annotation_path.resolve()),
        "private_key_directory": str(sealed_output),
        "discovery_total": len(discovery),
        "validation_total_precommitted": len(validation),
        "discovery_reveal_key_sha256": sha256_file(discovery_reveal_path),
        "holdout_reveal_key_sha256": sha256_file(holdout_reveal_path),
        "manifest": str(manifest_path),
        "complete_marker": str(complete_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-output", type=Path, required=True)
    parser.add_argument("--sealed-output-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--generate-seeds",
        action="store_true",
        help="generate two independent 256-bit seeds and store them only in private keys",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--discovery-per-condition", type=int, default=18)
    parser.add_argument("--validation-per-condition", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.generate_seeds:
        parser.error("formal CLI generation requires --generate-seeds")
    discovery_seed = secrets.randbits(256)
    validation_seed = secrets.randbits(256)
    while validation_seed == discovery_seed:
        validation_seed = secrets.randbits(256)
    try:
        result = create_bundle(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            annotation_output=args.annotation_output,
            sealed_output_dir=args.sealed_output_dir,
            runs_root=args.runs_root,
            discovery_seed=discovery_seed,
            validation_seed=validation_seed,
            namespace=args.namespace,
            discovery_per_condition=args.discovery_per_condition,
            validation_per_condition=args.validation_per_condition,
            repo_root=args.repo_root,
        )
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
