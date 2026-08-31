"""Transparent, read-only analysis of Value Leakage trajectory artifacts.

This module deliberately keeps two concepts separate:

* ``estimates.json`` contains judge-parsed *visible final answers*.
* ``trajectories.json`` contains estimates extracted from the reasoning trace.

A trajectory endpoint is never substituted for a missing visible final answer.
All generated artifacts are written to a new directory outside ``runs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("baseline", "below_good", "above_good")
STRATA = ("pooled", "start_above", "start_below", "start_equal")
DEFAULT_GRID_POINTS = 1000
DEFAULT_WINDOW_FRACTION = 0.20
DEFAULT_OUTLIER_FACTOR = 10.0
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
ANALYSIS_VERSION = "1"

COLORS = {
    "baseline": "#607D8B",
    "below_good": "#1f77b4",
    "above_good": "#c85a00",
}
LABELS = {
    "baseline": "baseline",
    "below_good": "below favoured",
    "above_good": "above favoured",
}


@dataclass(frozen=True)
class AnalysisSettings:
    """Every numerical choice that can affect the derived artifacts."""

    seed: int
    grid_points: int = DEFAULT_GRID_POINTS
    window_fraction: float = DEFAULT_WINDOW_FRACTION
    outlier_factor: float = DEFAULT_OUTLIER_FACTOR
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    figure_dpi: int = 150

    def validate(self) -> None:
        if self.grid_points < 2:
            raise ValueError("grid_points must be at least 2")
        if not 0 < self.window_fraction <= 0.5:
            raise ValueError("window_fraction must be in (0, 0.5]")
        if self.outlier_factor <= 1:
            raise ValueError("outlier_factor must be greater than 1")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if self.figure_dpi < 1:
            raise ValueError("figure_dpi must be positive")


@dataclass(frozen=True)
class TrajectoryRecord:
    rollout_index: int
    values: tuple[float, ...]
    stratum: str
    is_outlier_10x: bool


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Load analysis inputs without modifying or supplementing the run."""

    run_path = Path(run_dir).resolve()
    threshold_path = run_path / "threshold.json"
    trajectories_path = run_path / "trajectories.json"
    if not run_path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_path}")
    if not threshold_path.is_file():
        raise FileNotFoundError(f"missing required artifact: {threshold_path}")
    if not trajectories_path.is_file():
        raise FileNotFoundError(f"missing required artifact: {trajectories_path}")

    threshold_artifact = _read_json(threshold_path)
    threshold = (threshold_artifact.get("threshold")
                 if isinstance(threshold_artifact, dict)
                 else threshold_artifact)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold.json must contain a numeric threshold")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")

    trajectories = _read_json(trajectories_path)
    if not isinstance(trajectories, dict):
        raise ValueError("trajectories.json must be a condition-keyed object")

    estimates_path = run_path / "estimates.json"
    estimates_file_present = estimates_path.is_file()
    estimates = _read_json(estimates_path) if estimates_file_present else {}
    if not isinstance(estimates, dict):
        raise ValueError("estimates.json must be a condition-keyed object")

    return {
        "run_dir": run_path,
        "threshold": threshold,
        "threshold_artifact": threshold_artifact,
        "trajectories": trajectories,
        "estimates": estimates,
        "estimates_file_present": estimates_file_present,
    }


def normalize_estimate(estimate: float, threshold: float) -> float:
    """Return the shared threshold-normalized quantity used by the starter."""

    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and positive")
    return (float(estimate) - threshold) / threshold


def normalize_trajectory(values: Sequence[float], threshold: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - threshold) / threshold


def resample_trajectory(values: Sequence[float], grid_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
    """Linearly interpolate an estimate-event sequence to a common grid."""

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("a trajectory must contain at least two estimates")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectory estimates must be finite")
    return np.interp(
        np.linspace(0.0, 1.0, grid_points),
        np.linspace(0.0, 1.0, len(arr)),
        arr,
    )


def _numeric_trajectory(raw: Any) -> tuple[tuple[float, ...] | None, str | None]:
    if raw is None:
        return None, "null"
    if not isinstance(raw, list):
        return None, "not_a_list"
    if len(raw) < 2:
        return None, "fewer_than_two_estimates"
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, "non_numeric_estimate"
        number = float(value)
        if not math.isfinite(number):
            return None, "non_finite_estimate"
        values.append(number)
    return tuple(values), None


def _stratum(first: float, threshold: float) -> str:
    if first > threshold:
        return "start_above"
    if first < threshold:
        return "start_below"
    return "start_equal"


def prepare_trajectories(
    raw: Any,
    threshold: float,
    outlier_factor: float = DEFAULT_OUTLIER_FACTOR,
) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
    """Validate trajectories and expose every exclusion count and reason."""

    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("each trajectory condition must contain a list")
    lower, upper = threshold / outlier_factor, threshold * outlier_factor
    invalid_reasons: Counter[str] = Counter()
    records: list[TrajectoryRecord] = []
    for index, candidate in enumerate(raw):
        values, reason = _numeric_trajectory(candidate)
        if values is None:
            invalid_reasons[reason or "unknown"] += 1
            continue
        outlier = any(value < lower or value > upper for value in values)
        records.append(TrajectoryRecord(
            rollout_index=index,
            values=values,
            stratum=_stratum(values[0], threshold),
            is_outlier_10x=outlier,
        ))

    stratum_unfiltered = {
        name: (len(records) if name == "pooled"
               else sum(record.stratum == name for record in records))
        for name in STRATA
    }
    filtered = [record for record in records if not record.is_outlier_10x]
    stratum_filtered = {
        name: (len(filtered) if name == "pooled"
               else sum(record.stratum == name for record in filtered))
        for name in STRATA
    }
    quality = {
        "total_raw": len(raw),
        "valid_unfiltered": len(records),
        "invalid": len(raw) - len(records),
        "invalid_by_reason": dict(sorted(invalid_reasons.items())),
        "outlier_rule": f"drop trajectory if any estimate is outside [threshold/{outlier_factor:g}, threshold*{outlier_factor:g}]",
        "outlier_among_valid": sum(record.is_outlier_10x for record in records),
        "valid_after_10x_filter": len(filtered),
        "stratum_counts_unfiltered": stratum_unfiltered,
        "stratum_counts_after_10x_filter": stratum_filtered,
    }
    return records, quality


def select_stratum(records: Sequence[TrajectoryRecord], stratum: str) -> list[TrajectoryRecord]:
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum: {stratum}")
    if stratum == "pooled":
        return list(records)
    return [record for record in records if record.stratum == stratum]


def rollout_window_metrics(
    record: TrajectoryRecord,
    threshold: float,
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    window_fraction: float = DEFAULT_WINDOW_FRACTION,
) -> dict[str, Any]:
    """Compute early/late mean and median, then paired within-trace drift."""

    grid = resample_trajectory(record.values, grid_points)
    window_points = max(1, int(round(grid_points * window_fraction)))
    early = grid[:window_points]
    late = grid[-window_points:]
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    early_median = float(np.median(early))
    late_median = float(np.median(late))
    return {
        "rollout_index": record.rollout_index,
        "start_stratum": record.stratum,
        "outlier_10x": record.is_outlier_10x,
        "n_raw_estimates": len(record.values),
        "window_fraction": window_fraction,
        "window_grid_points": window_points,
        "first_reasoning_estimate": record.values[0],
        "first_reasoning_estimate_normalized": normalize_estimate(record.values[0], threshold),
        "reasoning_trajectory_endpoint": record.values[-1],
        "reasoning_trajectory_endpoint_normalized": normalize_estimate(record.values[-1], threshold),
        "early_mean_estimate": early_mean,
        "late_mean_estimate": late_mean,
        "early_median_estimate": early_median,
        "late_median_estimate": late_median,
        "early_mean_normalized": normalize_estimate(early_mean, threshold),
        "late_mean_normalized": normalize_estimate(late_mean, threshold),
        "early_median_normalized": normalize_estimate(early_median, threshold),
        "late_median_normalized": normalize_estimate(late_median, threshold),
        "within_trajectory_drift_mean_normalized": (late_mean - early_mean) / threshold,
        "within_trajectory_drift_median_normalized": (late_median - early_median) / threshold,
    }


def trajectory_curve(
    records: Sequence[TrajectoryRecord],
    threshold: float,
    grid_points: int = DEFAULT_GRID_POINTS,
) -> dict[str, Any] | None:
    if not records:
        return None
    stacked = np.vstack([
        normalize_trajectory(resample_trajectory(record.values, grid_points), threshold)
        for record in records
    ])
    q25, q75 = np.quantile(stacked, [0.25, 0.75], axis=0)
    return {
        "n": len(records),
        "position": np.linspace(0.0, 1.0, grid_points).tolist(),
        "median_normalized_estimate": np.median(stacked, axis=0).tolist(),
        "q25_normalized_estimate": q25.tolist(),
        "q75_normalized_estimate": q75.tolist(),
    }


def _descriptive(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"available": False, "n": 0}
    return {
        "available": True,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _derived_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _fallback_bootstrap_median(
    values: Sequence[float], *, confidence: float, resamples: int, seed: int,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(resamples, len(arr)), replace=True)
    stats = np.median(draws, axis=1)
    alpha = 1.0 - confidence
    return {
        "statistic": "median",
        "estimate": float(np.median(arr)),
        "confidence": confidence,
        "ci_low": float(np.quantile(stats, alpha / 2)),
        "ci_high": float(np.quantile(stats, 1 - alpha / 2)),
        "n": len(arr),
        "resamples": resamples,
        "seed": seed,
        "backend": "internal_numpy_fallback",
    }


def _bootstrap_median(
    values: Sequence[float], settings: AnalysisSettings, label: str,
) -> dict[str, Any] | None:
    if not values:
        return None
    seed = _derived_seed(settings.seed, label)
    try:
        from research.statistics import bootstrap_ci

        result = bootstrap_ci(
            values,
            confidence=settings.confidence,
            resamples=settings.bootstrap_resamples,
            seed=seed,
        )
        return result.to_dict()
    except ModuleNotFoundError:
        return _fallback_bootstrap_median(
            values,
            confidence=settings.confidence,
            resamples=settings.bootstrap_resamples,
            seed=seed,
        )


def _fallback_difference(
    above: Sequence[float], below: Sequence[float], *, confidence: float,
    resamples: int, seed: int,
) -> dict[str, Any]:
    a = np.asarray(above, dtype=float)
    b = np.asarray(below, dtype=float)
    rng = np.random.default_rng(seed)
    a_draws = rng.choice(a, size=(resamples, len(a)), replace=True)
    b_draws = rng.choice(b, size=(resamples, len(b)), replace=True)
    effects = np.median(a_draws, axis=1) - np.median(b_draws, axis=1)
    alpha = 1.0 - confidence
    return {
        "statistic": "difference_in_medians",
        "effect_size": float(np.median(a) - np.median(b)),
        "confidence": confidence,
        "ci_low": float(np.quantile(effects, alpha / 2)),
        "ci_high": float(np.quantile(effects, 1 - alpha / 2)),
        "n_a": len(a),
        "n_b": len(b),
        "resamples": resamples,
        "seed": seed,
        "backend": "internal_numpy_fallback",
    }


def _difference_in_medians(
    above: Sequence[float], below: Sequence[float], settings: AnalysisSettings,
    label: str,
) -> dict[str, Any] | None:
    if not above or not below:
        return None
    seed = _derived_seed(settings.seed, label)
    try:
        from research.statistics import difference_in_medians

        result = difference_in_medians(
            above,
            below,
            confidence=settings.confidence,
            resamples=settings.bootstrap_resamples,
            seed=seed,
        )
        return result.to_dict()
    except ModuleNotFoundError:
        return _fallback_difference(
            above,
            below,
            confidence=settings.confidence,
            resamples=settings.bootstrap_resamples,
            seed=seed,
        )


def _condition_stratum_analysis(
    records: Sequence[TrajectoryRecord],
    threshold: float,
    settings: AnalysisSettings,
    label: str,
) -> dict[str, Any]:
    metrics = [
        rollout_window_metrics(
            record,
            threshold,
            grid_points=settings.grid_points,
            window_fraction=settings.window_fraction,
        )
        for record in records
    ]
    drift = [row["within_trajectory_drift_mean_normalized"] for row in metrics]
    return {
        "n": len(records),
        "per_rollout": metrics,
        "summaries": {
            "early_mean_normalized": _descriptive(row["early_mean_normalized"] for row in metrics),
            "late_mean_normalized": _descriptive(row["late_mean_normalized"] for row in metrics),
            "early_median_normalized": _descriptive(row["early_median_normalized"] for row in metrics),
            "late_median_normalized": _descriptive(row["late_median_normalized"] for row in metrics),
            "within_trajectory_drift_mean_normalized": {
                **_descriptive(drift),
                "bootstrap_median_ci": _bootstrap_median(drift, settings, f"{label}:drift_mean"),
            },
            "within_trajectory_drift_median_normalized": _descriptive(
                row["within_trajectory_drift_median_normalized"] for row in metrics
            ),
            "reasoning_trajectory_endpoint_normalized": _descriptive(
                row["reasoning_trajectory_endpoint_normalized"] for row in metrics
            ),
        },
        "curve": trajectory_curve(records, threshold, settings.grid_points),
    }


def _mrf_report(
    by_condition: dict[str, dict[str, Any]],
    settings: AnalysisSettings,
    label: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for stratum in STRATA:
        values: dict[str, list[float]] = {}
        for condition in CONDITIONS:
            per_rollout = by_condition[condition][stratum]["per_rollout"]
            values[condition] = [
                row["within_trajectory_drift_mean_normalized"]
                for row in per_rollout
            ]
        above = values["above_good"]
        below = values["below_good"]
        effect = (float(np.median(above) - np.median(below))
                  if above and below else None)
        output[stratum] = {
            "definition": "median per-rollout ((late-20%-mean - early-20%-mean) / threshold), above_good minus below_good",
            "motivated_reasoning_factor": effect,
            "delta_above": float(np.median(above)) if above else None,
            "delta_below": float(np.median(below)) if below else None,
            "delta_baseline": (float(np.median(values["baseline"]))
                               if values["baseline"] else None),
            "n_above": len(above),
            "n_below": len(below),
            "n_baseline": len(values["baseline"]),
            "bootstrap_difference_in_medians": _difference_in_medians(
                above, below, settings, f"{label}:{stratum}:mrf"
            ),
        }
    return output


def _clean_estimates(raw: list[Any], threshold: float, outlier_factor: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    outliers = 0
    lower, upper = threshold / outlier_factor, threshold * outlier_factor
    for index, value in enumerate(raw):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            invalid += 1
            continue
        number = float(value)
        is_outlier = number < lower or number > upper
        outliers += int(is_outlier)
        rows.append({
            "rollout_index": index,
            "visible_final_estimate": number,
            "visible_final_estimate_normalized": normalize_estimate(number, threshold),
            "outlier_10x": is_outlier,
        })
    return rows, {
        "total": len(raw),
        "valid_unfiltered": len(rows),
        "invalid_or_null": invalid,
        "outlier_among_valid": outliers,
        "valid_after_10x_filter": len(rows) - outliers,
    }


def visible_final_estimate_analysis(
    estimates: dict[str, Any],
    *,
    estimates_file_present: bool,
    threshold: float,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    """Analyze only estimates.json; never infer missing values from reasoning."""

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        if condition not in estimates:
            conditions[condition] = {
                "artifact_status": "missing_condition_artifact",
                "available": False,
                "n": 0,
                "reason": ("estimates.json is absent" if not estimates_file_present
                           else f"estimates.json has no {condition!r} key"),
                "fallback_used": False,
            }
            continue
        raw = estimates[condition]
        if not isinstance(raw, list):
            conditions[condition] = {
                "artifact_status": "invalid_condition_artifact",
                "available": False,
                "n": 0,
                "reason": "condition artifact is not a list",
                "fallback_used": False,
            }
            continue
        rows, counts = _clean_estimates(raw, threshold, settings.outlier_factor)
        unfiltered = [row["visible_final_estimate_normalized"] for row in rows]
        filtered = [row["visible_final_estimate_normalized"] for row in rows
                    if not row["outlier_10x"]]
        conditions[condition] = {
            "artifact_status": "present",
            "available": bool(rows),
            "fallback_used": False,
            "counts": counts,
            "values": rows,
            "unfiltered_distribution": {
                **_descriptive(unfiltered),
                "bootstrap_median_ci": _bootstrap_median(
                    unfiltered, settings, f"visible_final:{condition}:unfiltered"
                ),
            },
            "robustness_10x_filtered_distribution": {
                **_descriptive(filtered),
                "bootstrap_median_ci": _bootstrap_median(
                    filtered, settings, f"visible_final:{condition}:filtered"
                ),
            },
        }

    above = conditions["above_good"]
    below = conditions["below_good"]
    above_values = ([row["visible_final_estimate_normalized"] for row in above.get("values", [])]
                    if above.get("artifact_status") == "present" else [])
    below_values = ([row["visible_final_estimate_normalized"] for row in below.get("values", [])]
                    if below.get("artifact_status") == "present" else [])
    return {
        "source": "estimates.json only",
        "artifact_file_status": "present" if estimates_file_present else "missing",
        "missing_value_policy": "report unavailable; never substitute trajectories.json endpoints",
        "conditions": conditions,
        "above_minus_below_difference_in_medians": _difference_in_medians(
            above_values, below_values, settings, "visible_final:above-minus-below"
        ),
    }


def analyze_run(run_dir: str | Path, settings: AnalysisSettings) -> dict[str, Any]:
    """Return a JSON-serializable analysis without writing any files."""

    settings.validate()
    artifacts = load_run_artifacts(run_dir)
    threshold = artifacts["threshold"]
    raw_trajectories = artifacts["trajectories"]

    records: dict[str, list[TrajectoryRecord]] = {}
    quality: dict[str, Any] = {}
    artifact_status: dict[str, str] = {}
    for condition in CONDITIONS:
        if condition not in raw_trajectories:
            artifact_status[condition] = "missing_condition_artifact"
            condition_raw: Any = []
        else:
            artifact_status[condition] = "present"
            condition_raw = raw_trajectories[condition]
        records[condition], quality[condition] = prepare_trajectories(
            condition_raw, threshold, settings.outlier_factor
        )

    modes: dict[str, Any] = {}
    for mode, apply_filter in (("unfiltered", False), ("robustness_10x_filtered", True)):
        by_condition: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            source = ([record for record in records[condition] if not record.is_outlier_10x]
                      if apply_filter else records[condition])
            by_condition[condition] = {}
            for stratum in STRATA:
                selected = select_stratum(source, stratum)
                by_condition[condition][stratum] = _condition_stratum_analysis(
                    selected,
                    threshold,
                    settings,
                    f"{mode}:{condition}:{stratum}",
                )
        modes[mode] = by_condition

    reasoning_endpoints: dict[str, Any] = {}
    for condition in CONDITIONS:
        values = [normalize_estimate(record.values[-1], threshold)
                  for record in records[condition]]
        reasoning_endpoints[condition] = {
            "artifact_status": artifact_status[condition],
            "n": len(values),
            "unfiltered_distribution": {
                **_descriptive(values),
                "bootstrap_median_ci": _bootstrap_median(
                    values, settings, f"reasoning_endpoint:{condition}"
                ),
            },
        }

    return {
        "schema_version": ANALYSIS_VERSION,
        "source_run": str(artifacts["run_dir"]),
        "threshold": threshold,
        "settings": asdict(settings),
        "trajectory_artifact_status": artifact_status,
        "trajectory_quality": quality,
        "trajectory_analysis": {
            "position_definition": "normalized position in the judge-extracted estimate sequence, not token/sentence time",
            "normalization": "(estimate - threshold) / threshold",
            "filter_modes": modes,
            "mrf": {
                "starter_compatible_unfiltered": {
                    "note": "Matches starter plot.py drift(), which does not apply the 10x filter.",
                    "by_stratum": _mrf_report(
                        modes["unfiltered"], settings, "mrf:starter_unfiltered"
                    ),
                },
                "robustness_10x_filtered": {
                    "note": "Explicit robustness analysis; trajectories with any point outside the 10x band are removed before drift aggregation.",
                    "outlier_factor": settings.outlier_factor,
                    "by_stratum": _mrf_report(
                        modes["robustness_10x_filtered"], settings, "mrf:filtered"
                    ),
                },
            },
            "reasoning_trajectory_endpoint_distribution": {
                "label": "reasoning-trajectory endpoint (not visible final answer)",
                "warning": "Do not use this as a replacement for missing estimates.json values.",
                "conditions": reasoning_endpoints,
            },
        },
        "visible_final_estimate_distribution": visible_final_estimate_analysis(
            artifacts["estimates"],
            estimates_file_present=artifacts["estimates_file_present"],
            threshold=threshold,
            settings=settings,
        ),
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_directory(
    output_dir: str | Path,
    run_dir: str | Path,
    runs_root: str | Path | None = None,
) -> Path:
    output = Path(output_dir).resolve()
    run = Path(run_dir).resolve()
    inferred_runs_root = (run.parent if run.parent.name.casefold() == "runs" else None)
    protected = Path(runs_root).resolve() if runs_root is not None else inferred_runs_root
    if _is_within(output, run):
        raise ValueError("output directory must not be inside the source run")
    if protected is not None and _is_within(output, protected):
        raise ValueError(f"output directory must not be inside runs root: {protected}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    return output


def _plot_trajectory_curves(analysis: dict[str, Any], path: Path, dpi: int) -> None:
    mode = analysis["trajectory_analysis"]["filter_modes"]["robustness_10x_filtered"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, stratum in zip(axes.flat, STRATA):
        for condition in CONDITIONS:
            packed = mode[condition][stratum]["curve"]
            if packed is None:
                continue
            x = np.asarray(packed["position"])
            centre = np.asarray(packed["median_normalized_estimate"])
            low = np.asarray(packed["q25_normalized_estimate"])
            high = np.asarray(packed["q75_normalized_estimate"])
            color = COLORS[condition]
            ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
            ax.plot(x, centre, color=color, linewidth=1.8,
                    label=f"{LABELS[condition]} (n={packed['n']})")
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_title(stratum.replace("_", " "))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized position in extracted estimate sequence")
    for ax in axes[:, 0]:
        ax.set_ylabel("(estimate - threshold) / threshold")
    fig.suptitle("Reasoning trajectories (10x-filtered robustness view)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_final_estimates(analysis: dict[str, Any], path: Path, dpi: int) -> None:
    final = analysis["visible_final_estimate_distribution"]["conditions"]
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for position, condition in enumerate(CONDITIONS, start=1):
        artifact = final[condition]
        values = [row["visible_final_estimate_normalized"]
                  for row in artifact.get("values", [])]
        if values:
            plotted = True
            box = ax.boxplot(values, positions=[position], widths=0.5,
                             patch_artist=True, showfliers=True)
            box["boxes"][0].set_facecolor(COLORS[condition])
            box["boxes"][0].set_alpha(0.35)
        else:
            label = ("artifact unavailable" if artifact["artifact_status"] != "present"
                     else "no parsed values")
            ax.text(position, 0.02, label, rotation=90, ha="center", va="bottom",
                    color="#777777", fontsize=8)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, len(CONDITIONS) + 1), [LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Visible final estimate: (estimate - threshold) / threshold")
    ax.set_title("Visible final-answer distributions (estimates.json only)")
    ax.grid(axis="y", alpha=0.2)
    if not plotted:
        ax.text(0.5, 0.9, "No parsed visible final estimates available",
                transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(start: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=start,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_provenance(
    run_dir: Path,
    settings: AnalysisSettings,
    config_source: Path | None = None,
) -> dict[str, Any]:
    inputs = {}
    for name in ("config.json", "threshold.json", "trajectories.json", "estimates.json"):
        path = run_dir / name
        inputs[name] = ({"present": True, "sha256": _sha256(path)}
                        if path.is_file() else {"present": False})
    script_path = Path(__file__).resolve()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "script": str(script_path),
        "script_sha256": _sha256(script_path),
        "analysis_version": ANALYSIS_VERSION,
        "settings": asdict(settings),
        "config_source": (str(config_source.resolve()) if config_source else None),
        "config_source_sha256": (_sha256(config_source) if config_source else None),
        "git_commit": _git_commit(script_path.parent),
        "input_artifacts": inputs,
        "outputs": [
            "analysis.json",
            "analysis_config.json",
            "provenance.json",
            "trajectory_curves.png",
            "final_estimate_distributions.png",
        ],
    }


def run_analysis(
    run_dir: str | Path,
    output_dir: str | Path,
    settings: AnalysisSettings,
    *,
    runs_root: str | Path | None = None,
    config_source: str | Path | None = None,
) -> dict[str, Path]:
    """Analyze a run and atomically reserve a never-overwritten output dir."""

    settings.validate()
    run_path = Path(run_dir).resolve()
    output = validate_output_directory(output_dir, run_path, runs_root)
    analysis = analyze_run(run_path, settings)
    config_path = Path(config_source).resolve() if config_source else None
    provenance = build_provenance(run_path, settings, config_path)

    output.mkdir(parents=True, exist_ok=False)
    analysis_path = output / "analysis.json"
    config_output_path = output / "analysis_config.json"
    provenance_path = output / "provenance.json"
    trajectory_plot = output / "trajectory_curves.png"
    final_plot = output / "final_estimate_distributions.png"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    config_output_path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_trajectory_curves(analysis, trajectory_plot, settings.figure_dpi)
    _plot_final_estimates(analysis, final_plot, settings.figure_dpi)
    return {
        "analysis": analysis_path,
        "config": config_output_path,
        "provenance": provenance_path,
        "trajectory_plot": trajectory_plot,
        "final_estimate_plot": final_plot,
    }


def _load_analysis_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("analysis config must be a JSON object")
    allowed = {
        "run_dir", "output_dir", "runs_root", "seed", "grid_points",
        "window_fraction", "outlier_factor", "bootstrap_resamples",
        "confidence", "figure_dpi",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown analysis config keys: {sorted(unknown)}")
    return payload


def _choose(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else config.get(key, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config; CLI flags override its values")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--seed", type=int, help="required explicitly or in --config")
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--window-fraction", type=float)
    parser.add_argument("--outlier-factor", type=float)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--figure-dpi", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_analysis_config(args.config) if args.config else {}
        run_dir = _choose(args.run_dir, config, "run_dir")
        output_dir = _choose(args.output_dir, config, "output_dir")
        seed = _choose(args.seed, config, "seed")
        if run_dir is None or output_dir is None or seed is None:
            parser.error("run_dir, output_dir, and seed are required via CLI or --config")
        settings = AnalysisSettings(
            seed=int(seed),
            grid_points=int(_choose(args.grid_points, config, "grid_points", DEFAULT_GRID_POINTS)),
            window_fraction=float(_choose(args.window_fraction, config, "window_fraction", DEFAULT_WINDOW_FRACTION)),
            outlier_factor=float(_choose(args.outlier_factor, config, "outlier_factor", DEFAULT_OUTLIER_FACTOR)),
            bootstrap_resamples=int(_choose(args.bootstrap_resamples, config, "bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
            confidence=float(_choose(args.confidence, config, "confidence", DEFAULT_CONFIDENCE)),
            figure_dpi=int(_choose(args.figure_dpi, config, "figure_dpi", 150)),
        )
        paths = run_analysis(
            run_dir,
            output_dir,
            settings,
            runs_root=_choose(args.runs_root, config, "runs_root"),
            config_source=args.config,
        )
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
