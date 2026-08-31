"""Fail-closed, single-record helpers for manual holdout annotation.

This module never infers a condition, assigns an annotation label, enumerates
holdout rollout files, or accepts a reveal-key path.  ``show-reasoning`` and
``locate-line`` open exactly one explicitly named ``Vxxx.txt`` file and never
return its visible-final-answer block.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


TARGET_TYPES = ("explicit_policy", "numerical_pivot", "ordinary_control")

_V_ID = re.compile(r"V[0-9]{3}")
_FORBIDDEN_PATH_PARTS = frozenset({"sealed", "runs"})
_GLOB_CHARACTERS = frozenset("*?[]")
_REASONING_MARKER = b"===== FULL REASONING (VERBATIM) =====\n\n"
_ANSWER_MARKER = b"\n\n===== VISIBLE FINAL ANSWER (VERBATIM) =====\n\n"


def _sha256_text(value: str) -> str:
    """Hash the exact, non-normalized Unicode text encoded as UTF-8."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_blind_id(blind_id: str) -> str:
    if _V_ID.fullmatch(blind_id) is None:
        raise ValueError("blind_id must be exactly one ID in Vxxx form")
    return blind_id


def _path_has_forbidden_part(path: Path) -> bool:
    return any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in path.parts)


def _safe_input_path(path: str | Path, *, kind: str) -> Path:
    """Resolve one literal input path and reject raw/private path classes."""

    raw = Path(path)
    if any(character in str(raw) for character in _GLOB_CHARACTERS):
        raise ValueError("wildcards and glob syntax are forbidden in input paths")
    if _path_has_forbidden_part(raw):
        raise ValueError("paths containing a sealed or runs component are forbidden")

    resolved = raw.resolve(strict=True)
    if _path_has_forbidden_part(resolved):
        raise ValueError("resolved paths containing a sealed or runs component are forbidden")
    if raw.is_symlink() or resolved.is_symlink():
        raise ValueError("symlink inputs are forbidden")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"expected a file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"expected a directory: {resolved}")
    return resolved


def _read_csv(path: str | Path, required_fields: Sequence[str]) -> list[dict[str, str]]:
    csv_path = _safe_input_path(path, kind="file")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if len(fields) != len(set(fields)):
            raise ValueError(f"{csv_path.name} contains duplicate header fields")
        missing = [field for field in required_fields if field not in fields]
        if missing:
            raise ValueError(
                f"{csv_path.name} is missing required fields: {', '.join(missing)}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{csv_path.name} contains extra unnamed columns")
    return [
        {
            field: "" if value is None else str(value).strip()
            for field, value in row.items()
        }
        for row in rows
    ]


def _ordered_operation_ids(rows: Sequence[dict[str, str]]) -> list[str]:
    if not rows:
        raise ValueError("operation CSV must contain at least one V row")
    ids = [_validate_blind_id(row["blind_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("operation CSV blind IDs must be unique")
    if ids != sorted(ids, key=lambda item: int(item[1:])):
        raise ValueError("operation CSV blind IDs must be in ascending V order")
    return ids


def next_incomplete(
    operation_csv: str | Path,
    target_csv: str | Path,
) -> dict[str, str | None]:
    """Return the first unfinished V row and its fixed manual-annotation stage."""

    operations = _read_csv(operation_csv, ("blind_id", "annotation_status"))
    targets = _read_csv(
        target_csv,
        ("blind_id", "target_type", "adjudication_status"),
    )
    blind_ids = _ordered_operation_ids(operations)

    expected_target_order = [
        (blind_id, target_type)
        for blind_id in blind_ids
        for target_type in TARGET_TYPES
    ]
    observed_target_order: list[tuple[str, str]] = []
    for row in targets:
        blind_id = _validate_blind_id(row["blind_id"])
        target_type = row["target_type"]
        if target_type not in TARGET_TYPES:
            raise ValueError(f"{blind_id}: unknown target_type {target_type!r}")
        observed_target_order.append((blind_id, target_type))
    if observed_target_order != expected_target_order:
        raise ValueError(
            "target CSV must contain exactly three ordered target rows per operation V ID"
        )

    target_by_id = {
        blind_id: targets[index * len(TARGET_TYPES) : (index + 1) * len(TARGET_TYPES)]
        for index, blind_id in enumerate(blind_ids)
    }
    for operation in operations:
        blind_id = operation["blind_id"]
        target_rows = target_by_id[blind_id]
        operation_complete = operation["annotation_status"] == "complete"
        completed_targets = [
            row["adjudication_status"] == "complete" for row in target_rows
        ]

        if not operation_complete:
            if any(completed_targets):
                raise ValueError(
                    f"{blind_id}: target adjudication cannot be complete before operation annotation"
                )
            return {"blind_id": blind_id, "stage": "operation", "target_type": None}

        first_incomplete: int | None = None
        for index, is_complete in enumerate(completed_targets):
            if not is_complete and first_incomplete is None:
                first_incomplete = index
            elif is_complete and first_incomplete is not None:
                raise ValueError(
                    f"{blind_id}: target adjudications must be completed in fixed order"
                )
        if first_incomplete is not None:
            return {
                "blind_id": blind_id,
                "stage": "target_adjudication",
                "target_type": TARGET_TYPES[first_incomplete],
            }

    return {"blind_id": None, "stage": "complete", "target_type": None}


def _rollout_path(
    *,
    blind_id: str,
    bundle: str | Path | None = None,
    rollouts_dir: str | Path | None = None,
) -> Path:
    valid_id = _validate_blind_id(blind_id)
    if (bundle is None) == (rollouts_dir is None):
        raise ValueError("provide exactly one of bundle or rollouts_dir")

    if bundle is not None:
        root = _safe_input_path(bundle, kind="directory")
        directory = _safe_input_path(root / "holdout_rollouts", kind="directory")
    else:
        directory = _safe_input_path(rollouts_dir, kind="directory")

    candidate = directory / f"{valid_id}.txt"
    path = _safe_input_path(candidate, kind="file")
    if path.parent != directory:
        raise ValueError("holdout file must be a direct child of the selected directory")
    return path


def _extract_reasoning(path: Path, blind_id: str) -> str:
    """Extract only the exact reasoning bytes from one validated V wrapper."""

    payload = path.read_bytes()
    if b"\r" in payload:
        raise ValueError(f"{blind_id}: CR bytes are forbidden; exact LF text required")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{blind_id}: wrapper is not strict UTF-8") from exc

    expected_header = f"BLINDED HOLDOUT ROLLOUT {blind_id}\n".encode("utf-8")
    if not payload.startswith(expected_header):
        raise ValueError(f"{blind_id}: wrapper ID/header mismatch")
    if payload.count(_REASONING_MARKER) != 1:
        raise ValueError(f"{blind_id}: reasoning delimiter must occur exactly once")
    if payload.count(_ANSWER_MARKER) != 1:
        raise ValueError(f"{blind_id}: visible-answer delimiter must occur exactly once")

    reasoning_start = payload.index(_REASONING_MARKER) + len(_REASONING_MARKER)
    answer_start = payload.index(_ANSWER_MARKER)
    if answer_start <= reasoning_start:
        raise ValueError(f"{blind_id}: wrapper delimiters are out of order")
    reasoning_bytes = payload[reasoning_start:answer_start]
    if not reasoning_bytes:
        raise ValueError(f"{blind_id}: reasoning block is empty")
    return reasoning_bytes.decode("utf-8", errors="strict")


def show_reasoning(
    *,
    blind_id: str,
    bundle: str | Path | None = None,
    rollouts_dir: str | Path | None = None,
) -> str:
    """Read exactly one V file and return only its reasoning block."""

    path = _rollout_path(
        blind_id=blind_id,
        bundle=bundle,
        rollouts_dir=rollouts_dir,
    )
    return _extract_reasoning(path, blind_id)


def _physical_lines(reasoning: str) -> list[tuple[int, int, str]]:
    """Return exact LF-delimited lines with Unicode code-point offsets."""

    lines: list[tuple[int, int, str]] = []
    start = 0
    while True:
        newline = reasoning.find("\n", start)
        if newline == -1:
            lines.append((start, len(reasoning), reasoning[start:]))
            return lines
        lines.append((start, newline, reasoning[start:newline]))
        start = newline + 1


def locate_line(
    *,
    blind_id: str,
    text: str,
    bundle: str | Path | None = None,
    rollouts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Locate every exact full-physical-line match in one V reasoning block."""

    if not isinstance(text, str) or not text:
        raise ValueError("text must be one non-empty complete physical line")
    if "\r" in text or "\n" in text:
        raise ValueError("text must not contain a line terminator")

    reasoning = show_reasoning(
        blind_id=blind_id,
        bundle=bundle,
        rollouts_dir=rollouts_dir,
    )
    matches: list[dict[str, Any]] = []
    for start, end, line in _physical_lines(reasoning):
        if line != text:
            continue
        prefix_before = reasoning[:start]
        prefix_through = reasoning[:end]
        matches.append(
            {
                "span": [start, end],
                "start_char": start,
                "end_char_exclusive": end,
                "target_text_verbatim": line,
                "target_sha256": _sha256_text(line),
                "prefix_before_sha256": _sha256_text(prefix_before),
                "prefix_through_sha256": _sha256_text(prefix_through),
            }
        )
    if not matches:
        raise ValueError("text is not an exact complete physical line in the reasoning block")
    return {
        "blind_id": _validate_blind_id(blind_id),
        "hash_encoding": "UTF-8; no Unicode or newline normalization",
        "reasoning_sha256": _sha256_text(reasoning),
        "match_count": len(matches),
        "matches": matches,
    }


def _add_rollout_location(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path)
    group.add_argument(
        "--rollouts-dir",
        "--holdout-rollouts",
        dest="rollouts_dir",
        type=Path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_parser = subparsers.add_parser("next", help="show the next manual stage")
    next_parser.add_argument(
        "--operation-csv",
        "--operation-annotation",
        dest="operation_csv",
        type=Path,
        required=True,
    )
    next_parser.add_argument(
        "--target-csv",
        "--target-annotation",
        dest="target_csv",
        type=Path,
        required=True,
    )

    show_parser = subparsers.add_parser(
        "show-reasoning",
        help="show only one explicitly requested reasoning block",
    )
    _add_rollout_location(show_parser)
    show_parser.add_argument("--blind-id", required=True)

    locate_parser = subparsers.add_parser(
        "locate-line",
        help="locate every exact full-line match in one reasoning block",
    )
    _add_rollout_location(locate_parser)
    locate_parser.add_argument("--blind-id", required=True)
    locate_parser.add_argument("--text", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "next":
            result = next_incomplete(args.operation_csv, args.target_csv)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        elif args.command == "show-reasoning":
            reasoning = show_reasoning(
                blind_id=args.blind_id,
                bundle=args.bundle,
                rollouts_dir=args.rollouts_dir,
            )
            sys.stdout.write(reasoning)
            if not reasoning.endswith("\n"):
                sys.stdout.write("\n")
        else:
            result = locate_line(
                blind_id=args.blind_id,
                text=args.text,
                bundle=args.bundle,
                rollouts_dir=args.rollouts_dir,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
