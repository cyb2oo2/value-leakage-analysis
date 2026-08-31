"""Read and export raw rollout records without interpreting motivation.

The estimate judge and trajectory judge read different source fields.  This
tool therefore never treats the final trajectory point as the visible final
answer.  Missing judge artifacts remain explicitly unavailable.

Examples:
    python -m research.inspect_rollouts --model qwen3.5-122b-a10b --condition baseline --index 0
    python -m research.inspect_rollouts --model qwen3.5-122b-a10b --compare --index 7
    python -m research.inspect_rollouts --model qwen3.5-122b-a10b --compare --random 3 --seed 20260825
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

from research.inspect_runs import CONDITIONS, read_json, resolve_run


@dataclass(frozen=True)
class ArtifactValue:
    status: str
    value: Any = None
    reason: str | None = None


@dataclass(frozen=True)
class RolloutView:
    run_name: str
    run_dir: str
    model: str | None
    model_id: str | None
    backend: str | None
    provider: str | None
    condition: str
    rollout_id: int
    row_position: int
    prompt: str
    reference_threshold: float
    prompt_threshold: float | None
    reasoning: str
    visible_final_answer: str
    parsed_final_estimate: ArtifactValue
    successive_estimates: ArtifactValue
    api_error: str | None


def _artifact_at(mapping: Any, condition: str, position: int, label: str) -> ArtifactValue:
    if not isinstance(mapping, dict):
        return ArtifactValue("unavailable", reason=f"{label} artifact is missing or malformed")
    if condition not in mapping:
        return ArtifactValue(
            "unavailable",
            reason=f"{label} artifact has no {condition!r} condition",
        )
    values = mapping[condition]
    if not isinstance(values, list) or position >= len(values):
        return ArtifactValue("unavailable", reason=f"{label} artifact has no aligned slot")
    value = values[position]
    if value is None:
        return ArtifactValue("invalid", reason=f"{label} judge returned no parseable value")
    return ArtifactValue("available", value=value)


def load_rollout(run_dir: Path | str, condition: str, rollout_id: int) -> RolloutView:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    path = Path(run_dir).resolve()
    raw_path = path / f"{condition}.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"condition file is missing: {raw_path}")
    raw = read_json(raw_path)
    rows = raw.get("rows") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"malformed rows in {raw_path}")
    matches = [(position, row) for position, row in enumerate(rows) if row.get("i") == rollout_id]
    if not matches:
        raise IndexError(f"rollout id {rollout_id} is absent from {raw_path}")
    if len(matches) > 1:
        raise ValueError(f"rollout id {rollout_id} occurs more than once in {raw_path}")
    position, row = matches[0]

    config = read_json(path / "config.json") if (path / "config.json").is_file() else {}
    threshold_record = read_json(path / "threshold.json")
    threshold = threshold_record.get("threshold")
    if not isinstance(threshold, (int, float)) or threshold == 0:
        raise ValueError(f"invalid reference threshold in {path / 'threshold.json'}")
    estimates = read_json(path / "estimates.json") if (path / "estimates.json").is_file() else None
    trajectories = read_json(path / "trajectories.json") if (path / "trajectories.json").is_file() else None

    return RolloutView(
        run_name=path.name,
        run_dir=str(path),
        model=config.get("model") if isinstance(config, dict) else None,
        model_id=config.get("model_id") if isinstance(config, dict) else None,
        backend=config.get("backend") if isinstance(config, dict) else None,
        provider=config.get("provider") if isinstance(config, dict) else None,
        condition=condition,
        rollout_id=rollout_id,
        row_position=position,
        prompt=raw.get("prompt", ""),
        reference_threshold=float(threshold),
        prompt_threshold=raw.get("threshold"),
        reasoning=row.get("reasoning") or "",
        visible_final_answer=row.get("content") or "",
        parsed_final_estimate=_artifact_at(estimates, condition, position, "estimate judge"),
        successive_estimates=_artifact_at(trajectories, condition, position, "trajectory judge"),
        api_error=row.get("error"),
    )


def available_rollout_ids(run_dir: Path | str, conditions: Sequence[str]) -> list[int]:
    id_sets: list[set[int]] = []
    path = Path(run_dir)
    for condition in conditions:
        raw = read_json(path / f"{condition}.json")
        rows = raw.get("rows", [])
        id_sets.append({row["i"] for row in rows if isinstance(row, dict) and isinstance(row.get("i"), int)})
    return sorted(set.intersection(*id_sets)) if id_sets else []


def select_random_ids(
    run_dir: Path | str,
    conditions: Sequence[str],
    count: int,
    seed: int,
) -> list[int]:
    candidates = available_rollout_ids(run_dir, conditions)
    if count < 1:
        raise ValueError("random sample count must be at least 1")
    if count > len(candidates):
        raise ValueError(f"requested {count} rollouts but only {len(candidates)} ids are shared")
    return sorted(random.Random(seed).sample(candidates, count))


def _display_artifact(artifact: ArtifactValue) -> str:
    if artifact.status == "available":
        if isinstance(artifact.value, list):
            return ", ".join(str(value) for value in artifact.value)
        return str(artifact.value)
    return f"[{artifact.status}: {artifact.reason}]"


def _format_number(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,}"


def render_markdown(views: Sequence[RolloutView], seed: int | None = None) -> str:
    lines = ["# Rollout inspection", ""]
    if seed is not None:
        lines.extend((f"Random seed: `{seed}`", ""))
    for view in views:
        lines.extend(
            (
                f"## {view.run_name} / {view.condition} / rollout {view.rollout_id}",
                "",
                f"- Model ID: `{view.model_id or view.model or 'unknown'}`",
                f"- Backend/provider: `{view.backend or 'unknown'}` / `{view.provider or 'default'}`",
                f"- Reference threshold: `{_format_number(view.reference_threshold)}`",
                f"- Threshold embedded in prompt: `{view.prompt_threshold}`",
                f"- Raw API error: `{view.api_error or 'none'}`",
                f"- Parsed final estimate (visible-answer judge): `{_display_artifact(view.parsed_final_estimate)}`",
                f"- Successive estimates (reasoning judge): `{_display_artifact(view.successive_estimates)}`",
                "",
                "### Exact prompt",
                "",
                "```text",
                view.prompt,
                "```",
                "",
                "### Full reasoning",
                "",
                "```text",
                view.reasoning or "[empty reasoning]",
                "```",
                "",
                "### Visible final answer",
                "",
                "```text",
                view.visible_final_answer or "[empty visible answer]",
                "```",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_text(views: Sequence[RolloutView], seed: int | None = None) -> str:
    chunks = []
    if seed is not None:
        chunks.append(f"RANDOM SEED: {seed}")
    for view in views:
        chunks.append(
            "\n".join(
                (
                    f"RUN: {view.run_name}",
                    f"CONDITION: {view.condition}",
                    f"ROLLOUT ID: {view.rollout_id}",
                    f"MODEL ID: {view.model_id or view.model or 'unknown'}",
                    f"BACKEND / PROVIDER: {view.backend or 'unknown'} / {view.provider or 'default'}",
                    f"REFERENCE THRESHOLD: {_format_number(view.reference_threshold)}",
                    f"PROMPT THRESHOLD: {view.prompt_threshold}",
                    f"RAW API ERROR: {view.api_error or 'none'}",
                    f"PARSED FINAL ESTIMATE (VISIBLE-ANSWER JUDGE): {_display_artifact(view.parsed_final_estimate)}",
                    f"SUCCESSIVE ESTIMATES (REASONING JUDGE): {_display_artifact(view.successive_estimates)}",
                    "",
                    "EXACT PROMPT",
                    view.prompt,
                    "",
                    "FULL REASONING",
                    view.reasoning or "[empty reasoning]",
                    "",
                    "VISIBLE FINAL ANSWER",
                    view.visible_final_answer or "[empty visible answer]",
                )
            )
        )
    return ("\n\n" + "=" * 88 + "\n\n").join(chunks).rstrip() + "\n"


def _crossed_threshold(view: RolloutView) -> str:
    if view.successive_estimates.status != "available" or not view.successive_estimates.value:
        return ""
    values = view.successive_estimates.value
    threshold = view.reference_threshold
    sides = {-1 if value < threshold else 1 if value > threshold else 0 for value in values}
    return str(-1 in sides and 1 in sides).lower()


ANNOTATION_FIELDS = (
    "rollout_id",
    "model",
    "condition",
    "threshold",
    "first_estimate",
    "trajectory_last_estimate",
    "parsed_visible_final_estimate",
    "parsed_visible_final_estimate_status",
    "crossed_threshold",
    "explicit_bias_statement",
    "explicit_impartiality_statement",
    "reasoning_component_notes",
    "notes",
)


def annotation_rows(views: Sequence[RolloutView]) -> list[dict[str, Any]]:
    rows = []
    for view in views:
        trajectory = (
            view.successive_estimates.value
            if view.successive_estimates.status == "available"
            else []
        )
        rows.append(
            {
                "rollout_id": view.rollout_id,
                "model": view.model_id or view.model or "",
                "condition": view.condition,
                "threshold": view.reference_threshold,
                "first_estimate": trajectory[0] if trajectory else "",
                "trajectory_last_estimate": trajectory[-1] if trajectory else "",
                "parsed_visible_final_estimate": (
                    view.parsed_final_estimate.value
                    if view.parsed_final_estimate.status == "available"
                    else ""
                ),
                "parsed_visible_final_estimate_status": view.parsed_final_estimate.status,
                "crossed_threshold": _crossed_threshold(view),
                "explicit_bias_statement": "",
                "explicit_impartiality_statement": "",
                "reasoning_component_notes": "",
                "notes": "",
            }
        )
    return rows


def write_new(path: Path, content: str, protected_run_dir: Path) -> None:
    resolved = path.resolve()
    if resolved == protected_run_dir.resolve() or protected_run_dir.resolve() in resolved.parents:
        raise ValueError("refusing to write an export inside the immutable run directory")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_annotation_csv(views: Sequence[RolloutView]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(annotation_rows(views))
    return stream.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--run-dir", type=Path)
    location.add_argument("--model", help="unambiguous run/model/model-id query")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    condition = parser.add_mutually_exclusive_group(required=True)
    condition.add_argument("--condition", choices=CONDITIONS)
    condition.add_argument("--compare", action="store_true", help="show all three conditions")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--random", type=int, metavar="N", help="sample N shared rollout ids")
    parser.add_argument("--seed", type=int, help="required with --random; recorded in output")
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    parser.add_argument("--export", type=Path, help="write rendered inspection; refuses overwrite")
    parser.add_argument("--annotation", type=Path, help="write blank manual annotation .csv or .json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Windows terminals commonly default to GBK.  Raw rollouts can contain
    # arbitrary Unicode, so make stdout lossless instead of failing mid-read.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.random is not None and args.seed is None:
        parser.error("--seed is required with --random")
    if args.random is None and args.seed is not None:
        parser.error("--seed is only meaningful with --random")

    run_dir = args.run_dir.resolve() if args.run_dir else resolve_run(args.runs_root, args.model)
    conditions = CONDITIONS if args.compare else (args.condition,)
    ids = (
        [args.index]
        if args.index is not None
        else select_random_ids(run_dir, conditions, args.random, args.seed)
    )
    views = [load_rollout(run_dir, condition, rollout_id) for rollout_id in ids for condition in conditions]

    if args.format == "json":
        rendered = json.dumps([asdict(view) for view in views], indent=2, ensure_ascii=False) + "\n"
    elif args.format == "markdown":
        rendered = render_markdown(views, args.seed)
    else:
        rendered = render_text(views, args.seed)
    if args.export:
        write_new(args.export, rendered, run_dir)
        print(f"saved {args.export}")
    else:
        print(rendered, end="")

    if args.annotation:
        suffix = args.annotation.suffix.casefold()
        if suffix == ".csv":
            annotation = render_annotation_csv(views)
        elif suffix == ".json":
            annotation = json.dumps(annotation_rows(views), indent=2, ensure_ascii=False) + "\n"
        else:
            raise ValueError("annotation path must end in .csv or .json")
        write_new(args.annotation, annotation, run_dir)
        print(f"saved {args.annotation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
