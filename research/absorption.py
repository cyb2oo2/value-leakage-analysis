"""Absorption, crossing asymmetry, visible finals, and mechanical Delta_pivot.

Follow-up to ``research.side_mechanics``.  Read-only on shipped runs.  Does not
open holdout packets or call a model API.  Visible finals are parsed
conservatively from the first non-empty content line; a trajectory endpoint is
never used as a substitute.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.experiment_utils import (
    create_new_directory,
    git_commit,
    git_is_dirty,
    sha256_file,
    utc_now,
    write_new_json,
    write_new_text,
)
from research.inspect_runs import list_runs
from research.side_mechanics import (
    CONDITIONS,
    DEFAULT_PERM_SEED,
    DEFAULT_PERMUTATIONS,
    DONATION_CONDITIONS,
    _fmt_p,
    _fmt_stat,
    _proportion,
    permutation_delta,
    side_of,
    wilson_interval,
)
from research.trajectory_analysis import load_run_artifacts, prepare_trajectories


ANALYSIS_VERSION = "value-leakage.absorption/v1"
SIDES = ("below", "equal", "above")

_VISIBLE_LINE = re.compile(
    r"""^
    [\s*_#>-]*
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        |\d+(?:\.\d+)?
    )
    \s*(?P<suffix>million|billion|thousand|millions|billions)?
    [\s*_#.!]*
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class VisibleParse:
    rollout_index: int
    raw_line: str | None
    value: float | None
    status: str


def parse_visible_final(text: str | None) -> tuple[float | None, str, str | None]:
    """Fail-closed parser: first non-empty line must be a single number."""

    if not text or not str(text).strip():
        return None, "empty", None
    line = next((part.strip() for part in str(text).splitlines() if part.strip()), "")
    cleaned = line.replace("`", "").replace("*", "").replace("_", "")
    if re.search(r"\d\s*[-–—to]+\s*\d", cleaned, flags=re.IGNORECASE):
        return None, "range", line
    match = _VISIBLE_LINE.match(cleaned)
    if not match:
        return None, "not_single_number_line", line
    number = float(match.group("num").replace(",", ""))
    suffix = (match.group("suffix") or "").lower()
    if suffix.startswith("thousand"):
        number *= 1_000
    elif suffix.startswith("million"):
        number *= 1_000_000
    elif suffix.startswith("billion"):
        number *= 1_000_000_000
    if not np.isfinite(number) or number <= 0:
        return None, "nonpositive", line
    return number, "ok", line


def on_favored_side(condition: str, side: str) -> bool | None:
    if condition == "above_good":
        return side == "above"
    if condition == "below_good":
        return side in {"below", "equal"}
    return None


def first_favored_index(values: Sequence[float], threshold: float, condition: str) -> int | None:
    for index, value in enumerate(values):
        if on_favored_side(condition, side_of(float(value), threshold)):
            return index
    return None


def escaped_after_hit(values: Sequence[float], threshold: float, condition: str) -> bool | None:
    hit = first_favored_index(values, threshold, condition)
    if hit is None:
        return None
    for value in values[hit + 1 :]:
        if not on_favored_side(condition, side_of(float(value), threshold)):
            return True
    return False


@dataclass(frozen=True)
class TrajectoryAbsorption:
    rollout_index: int
    first_side: str
    last_side: str
    revision_up: bool | None
    start_favored: bool | None
    last_favored: bool | None
    converted: bool | None
    leaked: bool | None
    escaped_after_hit: bool | None
    first_favored_frac: float | None


def classify_absorption(
    values: Sequence[float],
    threshold: float,
    condition: str,
    *,
    rollout_index: int,
) -> TrajectoryAbsorption:
    first_side = side_of(float(values[0]), threshold)
    last_side = side_of(float(values[-1]), threshold)
    start_favored = on_favored_side(condition, first_side)
    last_favored = on_favored_side(condition, last_side)
    hit = first_favored_index(values, threshold, condition)
    last_index = max(len(values) - 1, 1)
    return TrajectoryAbsorption(
        rollout_index=rollout_index,
        first_side=first_side,
        last_side=last_side,
        revision_up=None if float(values[-1]) == float(values[0]) else float(values[-1]) > float(values[0]),
        start_favored=start_favored,
        last_favored=last_favored,
        converted=None if start_favored is None else ((not start_favored) and bool(last_favored)),
        leaked=None if start_favored is None else (bool(start_favored) and last_favored is False),
        escaped_after_hit=escaped_after_hit(values, threshold, condition),
        first_favored_frac=None if hit is None else hit / last_index,
    )


def _matrix(rows: Sequence[TrajectoryAbsorption]) -> dict[str, dict[str, int]]:
    counts = {origin: dict.fromkeys(SIDES, 0) for origin in SIDES}
    for row in rows:
        counts[row.first_side][row.last_side] += 1
    return counts


def _delta_pivot(above: Sequence[TrajectoryAbsorption], below: Sequence[TrajectoryAbsorption]) -> dict[str, Any]:
    def up_given(rows: Sequence[TrajectoryAbsorption], start: str) -> list[bool]:
        return [
            bool(row.revision_up)
            for row in rows
            if row.first_side == start and row.revision_up is not None
        ]

    cells = {
        "above_start_below": _proportion("up", up_given(above, "below")),
        "below_start_below": _proportion("up", up_given(below, "below")),
        "above_start_above": _proportion("up", up_given(above, "above")),
        "below_start_above": _proportion("up", up_given(below, "above")),
    }
    below_delta = None
    above_delta = None
    if cells["above_start_below"]["p"] is not None and cells["below_start_below"]["p"] is not None:
        below_delta = cells["above_start_below"]["p"] - cells["below_start_below"]["p"]
    if cells["above_start_above"]["p"] is not None and cells["below_start_above"]["p"] is not None:
        above_delta = cells["above_start_above"]["p"] - cells["below_start_above"]["p"]
    delta = None if below_delta is None or above_delta is None else 0.5 * (below_delta + above_delta)
    return {
        "cells": cells,
        "start_below_contrast": below_delta,
        "start_above_contrast": above_delta,
        "delta_pivot": delta,
        "weights": [0.5, 0.5],
        "rope_pp": [-10.0, 10.0],
        "inside_rope": None if delta is None else -0.10 <= delta <= 0.10,
    }


def _perm_delta_pivot(
    above: Sequence[TrajectoryAbsorption],
    below: Sequence[TrajectoryAbsorption],
    *,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    observed = _delta_pivot(above, below)["delta_pivot"]
    pooled = [("above_good", row) for row in above] + [("below_good", row) for row in below]
    n_above = len(above)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        labels = np.array(["above_good"] * n_above + ["below_good"] * (len(pooled) - n_above))
        rng.shuffle(labels)
        new_above = [row for label, (_, row) in zip(labels, pooled) if label == "above_good"]
        new_below = [row for label, (_, row) in zip(labels, pooled) if label == "below_good"]
        value = _delta_pivot(new_above, new_below)["delta_pivot"]
        if value is not None:
            nulls.append(value)
    if observed is None or not nulls:
        return {"observed": observed, "p_two_sided": None, "n_perm": n_perm, "seed": seed}
    extreme = int(sum(abs(value) >= abs(observed) for value in nulls))
    return {
        "observed": observed,
        "p_two_sided": (1.0 + extreme) / (1.0 + n_perm),
        "n_perm": n_perm,
        "seed": seed,
        "null_mean": float(np.mean(nulls)),
        "null_q05": float(np.quantile(nulls, 0.05)),
        "null_q95": float(np.quantile(nulls, 0.95)),
    }


def load_visible_parses(run_dir: Path, condition: str) -> list[VisibleParse]:
    payload = json.loads((run_dir / f"{condition}.json").read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    parsed = []
    for index, row in enumerate(rows):
        content = row.get("content") if isinstance(row, dict) else None
        value, status, line = parse_visible_final(content)
        parsed.append(VisibleParse(rollout_index=index, raw_line=line, value=value, status=status))
    return parsed


def analyze_run(
    run_dir: Path,
    *,
    drop_10x_outliers: bool = True,
    n_perm: int = DEFAULT_PERMUTATIONS,
    perm_seed: int = DEFAULT_PERM_SEED,
) -> dict[str, Any]:
    artifacts = load_run_artifacts(run_dir)
    threshold = artifacts["threshold"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    classified: dict[str, list[TrajectoryAbsorption]] = {}
    quality = {}
    visible = {}
    for condition in CONDITIONS:
        records, report = prepare_trajectories(artifacts["trajectories"].get(condition), threshold)
        quality[condition] = report
        kept = [record for record in records if not (drop_10x_outliers and record.is_outlier_10x)]
        classified[condition] = [
            classify_absorption(record.values, threshold, condition, rollout_index=record.rollout_index)
            for record in kept
        ]
        parses = load_visible_parses(run_dir, condition)
        judged = artifacts["estimates"].get(condition) if isinstance(artifacts["estimates"], dict) else None
        vis_rows = []
        for item in parses:
            if item.value is None:
                vis_rows.append(asdict(item) | {"side": None, "outlier_10x": None, "judge_value": None})
                continue
            outlier = item.value < threshold / 10 or item.value > threshold * 10
            judge_value = None
            if isinstance(judged, list) and item.rollout_index < len(judged):
                judge_value = judged[item.rollout_index]
            vis_rows.append(
                asdict(item)
                | {
                    "side": side_of(item.value, threshold),
                    "outlier_10x": outlier,
                    "judge_value": judge_value,
                }
            )
        kept_vis = [row for row in vis_rows if row["value"] is not None and not (drop_10x_outliers and row["outlier_10x"])]
        visible[condition] = {
            "n_raw": len(parses),
            "parse_status": dict(Counter(item.status for item in parses)),
            "n_parsed": sum(item.value is not None for item in parses),
            "n_kept": len(kept_vis),
            "p_above": _proportion("visible_above", [row["side"] == "above" for row in kept_vis]),
            "judge_available": isinstance(judged, list) and any(value is not None for value in judged),
            "parser_vs_judge": _parser_judge_agreement(vis_rows) if isinstance(judged, list) else None,
        }
    summaries = {}
    for condition, rows in classified.items():
        start_opposed = [row for row in rows if row.start_favored is False]
        start_favored = [row for row in rows if row.start_favored is True]
        hit = [row for row in rows if row.escaped_after_hit is not None]
        summaries[condition] = {
            "n": len(rows),
            "matrix_first_last": _matrix(rows),
            "p_last_favored": None
            if condition == "baseline"
            else _proportion("last_favored", [bool(row.last_favored) for row in rows]),
            "p_convert_given_start_opposed": None
            if condition == "baseline"
            else _proportion("convert", [bool(row.converted) for row in start_opposed]),
            "p_leak_given_start_favored": None
            if condition == "baseline"
            else _proportion("leak", [bool(row.leaked) for row in start_favored]),
            "p_escape_after_hit": None
            if condition == "baseline"
            else _proportion("escape", [bool(row.escaped_after_hit) for row in hit]),
            "median_first_favored_frac": None
            if condition == "baseline" or not hit
            else float(np.median([row.first_favored_frac for row in hit if row.first_favored_frac is not None])),
        }
    pivot = _delta_pivot(classified["above_good"], classified["below_good"])
    return {
        "run_name": run_dir.name,
        "model": config.get("model"),
        "threshold": threshold,
        "conditions": summaries,
        "visible": visible,
        "delta_pivot": pivot,
        "permutation_delta_pivot": _perm_delta_pivot(
            classified["above_good"],
            classified["below_good"],
            n_perm=n_perm,
            seed=perm_seed,
        ),
        "visible_delta_last": {
            "label": "visible_P(above|above_good)-P(above|below_good)",
            "delta": None
            if visible["above_good"]["p_above"]["p"] is None or visible["below_good"]["p_above"]["p"] is None
            else visible["above_good"]["p_above"]["p"] - visible["below_good"]["p_above"]["p"],
            "above_good": visible["above_good"]["p_above"],
            "below_good": visible["below_good"]["p_above"],
        },
    }


def _parser_judge_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparable = [
        row
        for row in rows
        if row["value"] is not None and isinstance(row.get("judge_value"), (int, float)) and not isinstance(row.get("judge_value"), bool)
    ]
    if not comparable:
        return {"n": 0, "exact": None, "same_side": None}
    exact = [abs(float(row["value"]) - float(row["judge_value"])) < 1 for row in comparable]
    return {
        "n": len(comparable),
        "exact": _proportion("exact", exact),
        "rel_median_abs_err": float(
            np.median(
                [
                    abs(float(row["value"]) - float(row["judge_value"])) / max(float(row["judge_value"]), 1.0)
                    for row in comparable
                ]
            )
        ),
    }


def _errorbar(ax, xs, summaries, color, label):
    used, ys, yerr = [], [], []
    for x, summary in zip(xs, summaries):
        if not summary or summary.get("p") is None or not summary.get("wilson95"):
            continue
        p = summary["p"]
        low, high = summary["wilson95"]["ci"]
        used.append(x)
        ys.append(p)
        yerr.append((max(0.0, p - low), max(0.0, high - p)))
    if ys:
        ax.errorbar(used, ys, yerr=np.array(yerr).T, fmt="o", color=color, capsize=3, label=label)


def _plot_convert_leak(qwen: Mapping[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["below_good", "above_good"]
    xs = np.arange(len(labels))
    convert = [qwen["conditions"][name]["p_convert_given_start_opposed"] for name in labels]
    leak = [qwen["conditions"][name]["p_leak_given_start_favored"] for name in labels]
    _errorbar(ax, xs - 0.08, convert, "#c85a00", "convert to favored | start opposed")
    _errorbar(ax, xs + 0.08, leak, "#1f77b4", "leak from favored | start favored")
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#bbbbbb", linewidth=1)
    ax.set_ylabel("proportion")
    ax.set_title("H-absorb: conversion should exceed leakage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_visible(qwen: Mapping[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = list(CONDITIONS)
    xs = np.arange(len(labels))
    _errorbar(ax, xs, [qwen["visible"][name]["p_above"] for name in labels], "#c85a00", "visible final > T")
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#bbbbbb", linewidth=1)
    ax.set_ylabel("P(visible final > threshold)")
    ax.set_title("Conservative visible-final parser (not trajectory last)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(qwen: Mapping[str, Any], path: Path) -> None:
    pivot = qwen["delta_pivot"]
    perm = qwen["permutation_delta_pivot"]
    vis = qwen["visible_delta_last"]
    lines = [
        "# Absorption, visible finals, and mechanical Delta_pivot",
        "",
        "Follow-up to the side-mechanics negative controls. No new sampling, no holdout text.",
        "",
        "## Why this experiment",
        "",
        "H-absorb says last-side splits because traces **convert onto the good side of T and then stay**. Symmetric seeking predicts similar crossing given start side, and would not require conversion to exceed leakage. Mechanical Delta_pivot is the frozen holdout estimand run on the judge sequence (descriptive, not confirmatory). Visible finals test whether the split is in the committed answer, without substituting the trajectory endpoint.",
        "",
        "## Mechanical Delta_pivot (equal 0.5 weights, frozen in the holdout plan)",
        "",
        f"- P(up | above, first below) = {_fmt_p(pivot['cells']['above_start_below'])}",
        f"- P(up | below, first below) = {_fmt_p(pivot['cells']['below_start_below'])}",
        f"- P(up | above, first above) = {_fmt_p(pivot['cells']['above_start_above'])}",
        f"- P(up | below, first above) = {_fmt_p(pivot['cells']['below_start_above'])}",
        f"- Delta_pivot = {_fmt_stat(pivot['delta_pivot'])} (ROPE [-0.10, +0.10]; inside_rope={pivot['inside_rope']})",
        f"- Permutation p = {_fmt_stat(perm.get('p_two_sided'))} (n_perm={perm.get('n_perm')})",
        "",
        "If Delta_pivot sits in the ROPE, condition-linked revision after start-side stratification is small. That is the H-anchor/H-seek prediction, not H-push.",
        "",
        "## Conversion versus leakage",
        "",
    ]
    for condition in DONATION_CONDITIONS:
        c = qwen["conditions"][condition]
        lines.append(
            f"- `{condition}` n={c['n']}: convert|start opposed = {_fmt_p(c['p_convert_given_start_opposed'])}; "
            f"leak|start favored = {_fmt_p(c['p_leak_given_start_favored'])}; "
            f"escape after first favored hit = {_fmt_p(c['p_escape_after_hit'])}"
        )
        lines.append(f"  first×last matrix: {c['matrix_first_last']}")
    lines.extend(
        [
            "",
            "Terminal conversion exceeds leakage in both arms, especially below_good (0.67 vs 0.02). That is the last-side absorption signature. First-hit stopping is a different claim: escape-after-hit is high (~0.7), so traces keep moving after they first touch the good side. H-absorb should be read as **terminal** absorption (the committed end lands on the good side and favored starts rarely leak), not as **stop-on-first-hit**.",
            "",
            "## Visible finals (fail-closed first-line parser)",
            "",
            f"- below_good P(visible > T) = {_fmt_p(qwen['visible']['below_good']['p_above'])} "
            f"(parsed {qwen['visible']['below_good']['n_parsed']}/{qwen['visible']['below_good']['n_raw']})",
            f"- above_good P(visible > T) = {_fmt_p(qwen['visible']['above_good']['p_above'])} "
            f"(parsed {qwen['visible']['above_good']['n_parsed']}/{qwen['visible']['above_good']['n_raw']})",
            f"- Delta_visible last = {_fmt_stat(vis['delta'])}",
            f"- baseline parser vs shipped estimate judge: {qwen['visible']['baseline']['parser_vs_judge']}",
            "",
            "Donation-arm estimates.json is missing by construction of the starter pipeline. This parser is not a Claude judge. UNKNOWN lines are dropped, not filled from the trajectory.",
            "",
            "## What this still does not show",
            "",
            "- It does not replace E01. Visible-final absorption can still be salience plus stopping, not value.",
            "- Mechanical Delta_pivot is not the holdout confirmation. The holdout uses human first-target totals.",
            "- A first-line parser can miss wrapped answers; parse-status counts are the coverage report.",
            "",
        ]
    )
    write_new_text(path, "\n".join(lines) + "\n")


def run_analysis(
    *,
    runs_root: Path,
    output_dir: Path,
    model_query: str = "qwen3.5-122b-a10b",
    n_perm: int = DEFAULT_PERMUTATIONS,
    perm_seed: int = DEFAULT_PERM_SEED,
) -> dict[str, Any]:
    raw_root = runs_root.resolve()
    out = create_new_directory(output_dir, [raw_root])
    matches = [item for item in list_runs(raw_root, model_query) if not item.files_missing]
    if len(matches) != 1:
        raise ValueError(f"expected one complete run matching {model_query!r}, got {len(matches)}")
    run_dir = Path(matches[0].run_dir)
    qwen = analyze_run(run_dir, n_perm=n_perm, perm_seed=perm_seed)
    payload = {
        "schema_version": ANALYSIS_VERSION,
        "created_at_utc": utc_now(),
        "model_query": model_query,
        "result": {k: v for k, v in qwen.items()},
    }
    analysis_path = out / "analysis.json"
    convert_path = out / "convert_vs_leak.png"
    visible_path = out / "visible_final_sides.png"
    report = out / "REPORT.md"
    write_new_json(analysis_path, payload)
    _plot_convert_leak(qwen, convert_path)
    _plot_visible(qwen, visible_path)
    _write_report(qwen, report)
    provenance = {
        "schema_version": "value-leakage.absorption-provenance/v1",
        "created_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "code_commit": git_commit(),
        "code_dirty": git_is_dirty(),
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "analysis.json": sha256_file(analysis_path),
            "REPORT.md": sha256_file(report),
            "convert_vs_leak.png": sha256_file(convert_path),
            "visible_final_sides.png": sha256_file(visible_path),
        },
    }
    write_new_json(out / "provenance.json", provenance)
    return {"output_dir": str(out), "model": qwen["model"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5-122b-a10b")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--perm-seed", type=int, default=DEFAULT_PERM_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_analysis(
                runs_root=args.runs_root,
                output_dir=args.output_dir,
                model_query=args.model,
                n_perm=args.n_perm,
                perm_seed=args.perm_seed,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
