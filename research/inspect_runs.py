"""Inventory shipped and newly-created Value Leakage run directories.

Examples:
    python -m research.inspect_runs
    python -m research.inspect_runs --model qwen3.5 --format json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CONDITIONS = ("baseline", "below_good", "above_good")
REQUIRED_FILES = (
    "baseline.json",
    "below_good.json",
    "above_good.json",
    "estimates.json",
    "trajectories.json",
    "threshold.json",
    "factor.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    run_dir: str
    model: str | None
    model_id: str | None
    backend: str | None
    provider: str | None
    configured_n: int | None
    threshold: float | None
    files_present: tuple[str, ...]
    files_missing: tuple[str, ...]
    raw_rows: dict[str, int | None]
    trajectory_valid: dict[str, int | None]
    estimate_valid: dict[str, int | None]

    @property
    def complete(self) -> bool:
        return not self.files_missing


def _safe_read(path: Path, default: Any) -> Any:
    try:
        return read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def summarize_run(run_dir: Path) -> RunSummary:
    run_dir = run_dir.resolve()
    config = _safe_read(run_dir / "config.json", {})
    threshold_record = _safe_read(run_dir / "threshold.json", {})
    trajectories = _safe_read(run_dir / "trajectories.json", {})
    estimates = _safe_read(run_dir / "estimates.json", {})

    raw_rows: dict[str, int | None] = {}
    trajectory_valid: dict[str, int | None] = {}
    estimate_valid: dict[str, int | None] = {}
    for condition in CONDITIONS:
        raw = _safe_read(run_dir / f"{condition}.json", None)
        rows = raw.get("rows") if isinstance(raw, dict) else None
        raw_rows[condition] = len(rows) if isinstance(rows, list) else None

        condition_trajectories = trajectories.get(condition)
        trajectory_valid[condition] = (
            sum(isinstance(item, list) and bool(item) for item in condition_trajectories)
            if isinstance(condition_trajectories, list)
            else None
        )
        condition_estimates = estimates.get(condition)
        estimate_valid[condition] = (
            sum(item is not None for item in condition_estimates)
            if isinstance(condition_estimates, list)
            else None
        )

    present = tuple(name for name in REQUIRED_FILES if (run_dir / name).is_file())
    missing = tuple(name for name in REQUIRED_FILES if name not in present)
    threshold = threshold_record.get("threshold") if isinstance(threshold_record, dict) else None
    return RunSummary(
        run_name=run_dir.name,
        run_dir=str(run_dir),
        model=config.get("model") if isinstance(config, dict) else None,
        model_id=config.get("model_id") if isinstance(config, dict) else None,
        backend=config.get("backend") if isinstance(config, dict) else None,
        provider=config.get("provider") if isinstance(config, dict) else None,
        configured_n=config.get("count") if isinstance(config, dict) else None,
        threshold=threshold,
        files_present=present,
        files_missing=missing,
        raw_rows=raw_rows,
        trajectory_valid=trajectory_valid,
        estimate_valid=estimate_valid,
    )


def list_runs(runs_root: Path | str = "runs", model_query: str | None = None) -> list[RunSummary]:
    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {root}")
    summaries = [summarize_run(path) for path in sorted(root.iterdir()) if path.is_dir()]
    if model_query:
        needle = model_query.casefold()
        summaries = [
            item
            for item in summaries
            if any(
                needle in (candidate or "").casefold()
                for candidate in (item.run_name, item.model, item.model_id)
            )
        ]
    return summaries


def resolve_run(runs_root: Path | str, query: str) -> Path:
    """Resolve one run by exact name/model first, then unambiguous substring."""
    matches = list_runs(runs_root, query)
    exact = [
        item
        for item in matches
        if query.casefold() in {
            item.run_name.casefold(),
            (item.model or "").casefold(),
            (item.model_id or "").casefold(),
        }
    ]
    candidates = exact or matches
    if not candidates:
        raise ValueError(f"no run matches {query!r}")
    if len(candidates) > 1:
        names = ", ".join(item.run_name for item in candidates)
        raise ValueError(f"run query {query!r} is ambiguous: {names}")
    return Path(candidates[0].run_dir)


def _format_table(rows: Iterable[RunSummary]) -> str:
    items = list(rows)
    if not items:
        return "No matching runs."
    headers = ("run", "model_id", "backend/provider", "N", "threshold", "files")
    body = []
    for item in items:
        route = "/".join(part for part in (item.backend, item.provider) if part) or "-"
        body.append(
            (
                item.run_name,
                item.model_id or item.model or "-",
                route,
                str(item.configured_n) if item.configured_n is not None else "-",
                f"{item.threshold:,.0f}" if item.threshold is not None else "-",
                "complete" if item.complete else "missing: " + ", ".join(item.files_missing),
            )
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in body)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--model", help="case-insensitive run/model/model-id filter")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = list_runs(args.runs_root, args.model)
    if args.format == "json":
        print(json.dumps([asdict(item) | {"complete": item.complete} for item in summaries], indent=2))
    else:
        print(_format_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

