"""Create blank V-ID holdout annotation templates without reading V text bytes.

This pre-holdout gate reads only the public bundle manifest and the names in
``holdout_rollouts/``.  It has no private-key argument and never opens a V file.
All outputs are exclusive-create and are kept outside the immutable bundle and
raw runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

from experiments.E02_trajectory_localization.blind_discovery import (
    ANNOTATION_FIELDS,
)
from research.experiment_utils import (
    ensure_output_outside_raw,
    sha256_file,
    sha256_text,
    utc_now,
    write_new_json,
    write_new_text,
)


OPERATION_SCHEMA_VERSION = "value-leakage.holdout-operation-annotation/v1"
TARGET_SCHEMA_VERSION = "value-leakage.holdout-target-adjudication/v1"
TEMPLATE_MANIFEST_SCHEMA_VERSION = "value-leakage.holdout-template-freeze/v1"
TARGET_TYPES = ("explicit_policy", "numerical_pivot", "ordinary_control")
TARGET_FIELDS = (
    "target_schema_version",
    "blind_id",
    "target_type",
    "adjudication_status",
    "target_status",
    "policy_subtype",
    "pivot_component",
    "start_char",
    "end_char_exclusive",
    "target_text_verbatim",
    "selection_rationale",
    "continuation_horizon_sufficient",
    "annotator_confidence",
    "notes",
)

_V_ID = re.compile(r"V[0-9]{3}")
_HOLDOUT_ARTIFACT = re.compile(r"holdout_rollouts/(V[0-9]{3})[.]txt")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def public_holdout_ids_from_manifest_and_filenames(
    bundle: Path,
    *,
    expected_total: int,
) -> tuple[list[str], dict[str, Any]]:
    """Return fixed V IDs without opening or hashing any V file."""

    root = bundle.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    total = manifest.get("holdout_rollout_total")
    if (
        isinstance(expected_total, bool)
        or not isinstance(expected_total, int)
        or expected_total < 1
    ):
        raise ValueError("expected_total must be a positive integer")
    if total != expected_total:
        raise ValueError(
            f"manifest holdout_rollout_total is {total!r}, expected {expected_total}"
        )
    if manifest.get("validation_total_precommitted") != total:
        raise ValueError("manifest validation_total_precommitted is inconsistent")
    per_condition = manifest.get("validation_per_hidden_condition")
    if (
        isinstance(per_condition, bool)
        or not isinstance(per_condition, int)
        or per_condition * 2 != total
    ):
        raise ValueError("manifest hidden-condition allocation is not exactly balanced")

    expected_ids = [f"V{index:03d}" for index in range(1, total + 1)]
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("manifest artifact_sha256 must be an object")
    committed_ids = sorted(
        match.group(1)
        for relative in artifact_hashes
        if isinstance(relative, str)
        and (match := _HOLDOUT_ARTIFACT.fullmatch(relative)) is not None
    )
    if committed_ids != expected_ids:
        raise ValueError("manifest holdout artifact IDs are not exact and ordered")

    holdout_dir = root / "holdout_rollouts"
    if not holdout_dir.is_dir():
        raise FileNotFoundError(f"holdout rollout directory does not exist: {holdout_dir}")
    entries = sorted(holdout_dir.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() for path in entries):
        raise ValueError("holdout rollout directory must not contain symlinks")
    observed_names = [path.name for path in entries if path.is_file()]
    expected_names = [f"{blind_id}.txt" for blind_id in expected_ids]
    if observed_names != expected_names or len(entries) != len(expected_names):
        raise ValueError("holdout rollout filenames do not match the public manifest")
    if any(not _V_ID.fullmatch(Path(name).stem) for name in observed_names):
        raise ValueError("unexpected holdout rollout filename")
    return expected_ids, manifest


def render_operation_template(blind_ids: Sequence[str]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS, lineterminator="\n")
    writer.writeheader()
    for blind_id in blind_ids:
        row = {field: "" for field in ANNOTATION_FIELDS}
        row["annotation_schema_version"] = OPERATION_SCHEMA_VERSION
        row["blind_id"] = blind_id
        writer.writerow(row)
    return stream.getvalue()


def render_target_template(blind_ids: Sequence[str]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TARGET_FIELDS, lineterminator="\n")
    writer.writeheader()
    for blind_id in blind_ids:
        for target_type in TARGET_TYPES:
            row = {field: "" for field in TARGET_FIELDS}
            row["target_schema_version"] = TARGET_SCHEMA_VERSION
            row["blind_id"] = blind_id
            row["target_type"] = target_type
            writer.writerow(row)
    return stream.getvalue()


def prepare_templates(
    *,
    bundle: Path,
    operation_output: Path,
    target_output: Path,
    template_manifest_output: Path,
    operation_codebook: Path,
    target_codebook: Path,
    frozen_documents: Sequence[Path],
    runs_root: Path,
    expected_total: int = 60,
) -> dict[str, Any]:
    root = bundle.resolve()
    raw_root = runs_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"public bundle does not exist: {root}")
    outputs = [
        ensure_output_outside_raw(operation_output, [raw_root]),
        ensure_output_outside_raw(target_output, [raw_root]),
        ensure_output_outside_raw(template_manifest_output, [raw_root]),
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError("operation, target, and manifest outputs must be distinct")
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite holdout template output: {output}")
        if output == root or _is_within(output, root):
            raise ValueError("holdout template outputs must be outside the public bundle")

    operation_codebook_path = operation_codebook.resolve()
    target_codebook_path = target_codebook.resolve()
    document_paths = [
        operation_codebook_path,
        target_codebook_path,
        *(path.resolve() for path in frozen_documents),
    ]
    if len(set(document_paths)) != len(document_paths):
        raise ValueError("frozen document paths must be unique")
    for path in document_paths:
        if not path.is_file():
            raise FileNotFoundError(f"frozen document does not exist: {path}")

    blind_ids, manifest = public_holdout_ids_from_manifest_and_filenames(
        root,
        expected_total=expected_total,
    )
    operation_path = write_new_text(
        outputs[0],
        render_operation_template(blind_ids),
    )
    target_path = write_new_text(
        outputs[1],
        render_target_template(blind_ids),
    )
    payload = {
        "schema_version": TEMPLATE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "bundle": str(root),
        "bundle_manifest": str((root / "manifest.json").resolve()),
        "bundle_manifest_sha256": sha256_file(root / "manifest.json"),
        "validation_mapping_commitment_sha256": manifest[
            "validation_mapping_commitment_sha256"
        ],
        "holdout_packet_commitment_sha256": manifest[
            "holdout_packet_commitment_sha256"
        ],
        "holdout_reveal_key_sha256": manifest["holdout_reveal_key_sha256"],
        "holdout_total": len(blind_ids),
        "blind_ids_sha256": _canonical_sha256(blind_ids),
        "holdout_filenames": [f"{blind_id}.txt" for blind_id in blind_ids],
        "operation_template": str(operation_path.resolve()),
        "operation_template_sha256": sha256_file(operation_path),
        "operation_schema_version": OPERATION_SCHEMA_VERSION,
        "target_template": str(target_path.resolve()),
        "target_template_sha256": sha256_file(target_path),
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "target_row_count": len(blind_ids) * len(TARGET_TYPES),
        "frozen_documents_sha256": {
            str(path): sha256_file(path) for path in document_paths
        },
        "prepare_tool": str(Path(__file__).resolve()),
        "prepare_tool_sha256": sha256_file(Path(__file__).resolve()),
        "holdout_text_bytes_read": False,
        "private_packets_read": False,
        "statement": (
            "Templates and protocol hashes were frozen from public metadata "
            "before any holdout text or private packet was read by this tool."
        ),
    }
    manifest_output = write_new_json(outputs[2], payload)
    return {
        "operation_template": str(operation_path),
        "target_template": str(target_path),
        "template_manifest": str(manifest_output),
        "holdout_total": len(blind_ids),
        "target_row_count": len(blind_ids) * len(TARGET_TYPES),
        "holdout_text_bytes_read": False,
        "private_packets_read": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--operation-output", type=Path, required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    parser.add_argument("--template-manifest-output", type=Path, required=True)
    parser.add_argument("--operation-codebook", type=Path, required=True)
    parser.add_argument("--target-codebook", type=Path, required=True)
    parser.add_argument(
        "--frozen-document",
        type=Path,
        action="append",
        default=[],
        help="repeat for hypotheses, exclusions, metrics, and design/config files",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--expected-total", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = prepare_templates(
            bundle=args.bundle,
            operation_output=args.operation_output,
            target_output=args.target_output,
            template_manifest_output=args.template_manifest_output,
            operation_codebook=args.operation_codebook,
            target_codebook=args.target_codebook,
            frozen_documents=args.frozen_document,
            runs_root=args.runs_root,
            expected_total=args.expected_total,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

