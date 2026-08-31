"""Small, transparent statistical helpers for trajectory analysis.

The functions in this module deliberately expose filtering and bootstrap
choices.  They do not calculate p-values, mutate inputs, or write artifacts.
All structured return values provide ``to_dict()`` for strict JSON output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np


OutlierRule: TypeAlias = Literal["none", "bounds", "iqr"]
BootstrapSize: TypeAlias = int | Literal["observed"]
Statistic: TypeAlias = Callable[[np.ndarray], Real]


@dataclass(frozen=True)
class FilterReport:
    """Audit trail for cleaning one numeric sample."""

    name: str
    input_count: int
    finite_count: int
    invalid_count: int
    outlier_count: int
    kept_count: int
    invalid_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    outlier_rule: str
    lower_bound: float | None
    upper_bound: float | None

    @property
    def valid_count(self) -> int:
        """Number retained after both validity and outlier filtering."""

        return self.kept_count

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_count": self.input_count,
            "finite_count": self.finite_count,
            "invalid_count": self.invalid_count,
            "outlier_count": self.outlier_count,
            "kept_count": self.kept_count,
            "valid_count": self.valid_count,
            "invalid_indices": list(self.invalid_indices),
            "outlier_indices": list(self.outlier_indices),
            "outlier_rule": self.outlier_rule,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class PairedFilterReport:
    """Audit trail when pairs must remain aligned during filtering."""

    input_pairs: int
    invalid_pairs: int
    invalid_values_a: int
    invalid_values_b: int
    outlier_pairs: int
    kept_pairs: int
    invalid_pair_indices: tuple[int, ...]
    outlier_pair_indices: tuple[int, ...]
    outlier_rule_a: str
    outlier_rule_b: str
    lower_bound_a: float | None
    upper_bound_a: float | None
    lower_bound_b: float | None
    upper_bound_b: float | None

    @property
    def valid_pairs(self) -> int:
        return self.kept_pairs

    def to_dict(self) -> dict[str, object]:
        return {
            "input_pairs": self.input_pairs,
            "invalid_pairs": self.invalid_pairs,
            "invalid_values_a": self.invalid_values_a,
            "invalid_values_b": self.invalid_values_b,
            "outlier_pairs": self.outlier_pairs,
            "kept_pairs": self.kept_pairs,
            "valid_pairs": self.valid_pairs,
            "invalid_pair_indices": list(self.invalid_pair_indices),
            "outlier_pair_indices": list(self.outlier_pair_indices),
            "outlier_rule_a": self.outlier_rule_a,
            "outlier_rule_b": self.outlier_rule_b,
            "lower_bound_a": self.lower_bound_a,
            "upper_bound_a": self.upper_bound_a,
            "lower_bound_b": self.lower_bound_b,
            "upper_bound_b": self.upper_bound_b,
        }


@dataclass(frozen=True)
class CleanedSample:
    values: tuple[float, ...]
    report: FilterReport

    def to_dict(self) -> dict[str, object]:
        return {"values": list(self.values), "report": self.report.to_dict()}


@dataclass(frozen=True)
class SummaryResult:
    n: int
    median: float
    quantiles: dict[str, float]
    quantile_method: str
    filter_report: FilterReport

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "median": self.median,
            "quantiles": dict(self.quantiles),
            "quantile_method": self.quantile_method,
            "filter_report": self.filter_report.to_dict(),
        }


@dataclass(frozen=True)
class BootstrapResult:
    statistic: str
    effect_size: float
    ci_low: float
    ci_high: float
    confidence: float
    n: int
    sample_size: int
    resamples: int
    seed: int
    ci_method: str
    filter_report: FilterReport

    def to_dict(self) -> dict[str, object]:
        return {
            "statistic": self.statistic,
            "effect_size": self.effect_size,
            "ci": [self.ci_low, self.ci_high],
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n": self.n,
            "sample_size": self.sample_size,
            "resamples": self.resamples,
            "seed": self.seed,
            "ci_method": self.ci_method,
            "filter_report": self.filter_report.to_dict(),
        }


@dataclass(frozen=True)
class DifferenceResult:
    statistic: str
    effect_size: float
    ci_low: float
    ci_high: float
    confidence: float
    mode: Literal["independent", "paired"]
    direction: str
    n_a: int
    n_b: int
    n_pairs: int | None
    sample_size_a: int
    sample_size_b: int
    resamples: int
    seed: int
    ci_method: str
    filter_report_a: FilterReport | None
    filter_report_b: FilterReport | None
    paired_filter_report: PairedFilterReport | None

    @property
    def n(self) -> dict[str, int]:
        counts = {"a": self.n_a, "b": self.n_b}
        if self.n_pairs is not None:
            counts["pairs"] = self.n_pairs
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "statistic": self.statistic,
            "effect_size": self.effect_size,
            "ci": [self.ci_low, self.ci_high],
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "mode": self.mode,
            "direction": self.direction,
            "n": self.n,
            "sample_size": {"a": self.sample_size_a, "b": self.sample_size_b},
            "resamples": self.resamples,
            "seed": self.seed,
            "ci_method": self.ci_method,
            "filter_report_a": (
                None if self.filter_report_a is None else self.filter_report_a.to_dict()
            ),
            "filter_report_b": (
                None if self.filter_report_b is None else self.filter_report_b.to_dict()
            ),
            "paired_filter_report": (
                None
                if self.paired_filter_report is None
                else self.paired_filter_report.to_dict()
            ),
        }


def _materialize(values: Iterable[object], name: str) -> list[object]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("sample name must be a non-empty string")
    if isinstance(values, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise TypeError(f"{name} must be an ordered one-dimensional iterable")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {values.shape}")
    try:
        materialized = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an ordered one-dimensional iterable") from exc
    if not materialized:
        raise ValueError(f"{name} must not be empty")
    return materialized


def _finite_real(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _finite_bound(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    converted = _finite_real(value)
    if converted is None:
        raise ValueError(f"{name} must be a finite real number or None")
    return converted


def _outlier_bounds(
    values: np.ndarray,
    *,
    outlier_rule: OutlierRule,
    lower_bound: float | None,
    upper_bound: float | None,
    iqr_multiplier: float,
) -> tuple[float | None, float | None, str]:
    if outlier_rule not in ("none", "bounds", "iqr"):
        raise ValueError("outlier_rule must be one of: 'none', 'bounds', 'iqr'")
    lower = _finite_bound(lower_bound, "lower_bound")
    upper = _finite_bound(upper_bound, "upper_bound")
    if outlier_rule == "none":
        if lower is not None or upper is not None:
            raise ValueError("bounds require outlier_rule='bounds'")
        return None, None, "none"
    if outlier_rule == "bounds":
        if lower is None and upper is None:
            raise ValueError("outlier_rule='bounds' requires at least one bound")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower_bound must be <= upper_bound")
        lower_text = "-inf" if lower is None else f"{lower:.12g}"
        upper_text = "inf" if upper is None else f"{upper:.12g}"
        return lower, upper, f"bounds: {lower_text} <= x <= {upper_text}"
    if lower is not None or upper is not None:
        raise ValueError("explicit bounds cannot be combined with outlier_rule='iqr'")
    multiplier = _finite_real(iqr_multiplier)
    if multiplier is None or multiplier <= 0:
        raise ValueError("iqr_multiplier must be a finite real number > 0")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = float(q3 - q1)
    lower = float(q1 - multiplier * iqr)
    upper = float(q3 + multiplier * iqr)
    return (
        lower,
        upper,
        f"iqr: q1 - {multiplier:.12g}*IQR <= x <= q3 + {multiplier:.12g}*IQR",
    )


def clean_numeric(
    values: Iterable[object],
    *,
    name: str = "values",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> CleanedSample:
    """Drop non-numeric/non-finite values, then apply one explicit outlier rule.

    Booleans are invalid numeric observations. Bounds are inclusive. An error is
    raised if the input is empty or filtering leaves no observations.
    """

    raw = _materialize(values, name)
    valid: list[tuple[int, float]] = []
    invalid_indices: list[int] = []
    for index, item in enumerate(raw):
        numeric = _finite_real(item)
        if numeric is None:
            invalid_indices.append(index)
        else:
            valid.append((index, numeric))
    if not valid:
        raise ValueError(
            f"{name} has no finite numeric observations "
            f"({len(invalid_indices)} invalid of {len(raw)})"
        )
    finite = np.asarray([value for _, value in valid], dtype=float)
    lower, upper, rule_text = _outlier_bounds(
        finite,
        outlier_rule=outlier_rule,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iqr_multiplier=iqr_multiplier,
    )
    kept: list[float] = []
    outlier_indices: list[int] = []
    for original_index, value in valid:
        is_outlier = (lower is not None and value < lower) or (
            upper is not None and value > upper
        )
        if is_outlier:
            outlier_indices.append(original_index)
        else:
            kept.append(value)
    if not kept:
        raise ValueError(
            f"{name} has no observations after filtering "
            f"({len(invalid_indices)} invalid, {len(outlier_indices)} outliers; {rule_text})"
        )
    report = FilterReport(
        name=name,
        input_count=len(raw),
        finite_count=len(valid),
        invalid_count=len(invalid_indices),
        outlier_count=len(outlier_indices),
        kept_count=len(kept),
        invalid_indices=tuple(invalid_indices),
        outlier_indices=tuple(outlier_indices),
        outlier_rule=rule_text,
        lower_bound=lower,
        upper_bound=upper,
    )
    return CleanedSample(tuple(kept), report)


def _clean_options(
    name: str,
    outlier_rule: OutlierRule,
    lower_bound: float | None,
    upper_bound: float | None,
    iqr_multiplier: float,
) -> dict[str, object]:
    return {
        "name": name,
        "outlier_rule": outlier_rule,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "iqr_multiplier": iqr_multiplier,
    }


def median(
    values: Iterable[object],
    *,
    name: str = "values",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> float:
    cleaned = clean_numeric(
        values,
        **_clean_options(name, outlier_rule, lower_bound, upper_bound, iqr_multiplier),
    )
    return float(np.median(cleaned.values))


def _probabilities(probabilities: Iterable[Real]) -> tuple[float, ...]:
    raw = _materialize(probabilities, "probabilities")
    parsed: list[float] = []
    for probability in raw:
        numeric = _finite_real(probability)
        if numeric is None or not 0 <= numeric <= 1:
            raise ValueError("each quantile probability must be a finite number in [0, 1]")
        parsed.append(numeric)
    if len(set(parsed)) != len(parsed):
        raise ValueError("quantile probabilities must be unique")
    return tuple(parsed)


def quantiles(
    values: Iterable[object],
    probabilities: Iterable[Real] = (0.25, 0.5, 0.75),
    *,
    method: str = "linear",
    name: str = "values",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> dict[str, float]:
    """Return JSON-friendly quantiles keyed by their requested probability."""

    requested = _probabilities(probabilities)
    if not isinstance(method, str) or not method:
        raise ValueError("quantile method must be a non-empty string")
    cleaned = clean_numeric(
        values,
        **_clean_options(name, outlier_rule, lower_bound, upper_bound, iqr_multiplier),
    )
    try:
        results = np.quantile(cleaned.values, requested, method=method)
    except ValueError as exc:
        raise ValueError(f"unsupported quantile method: {method!r}") from exc
    return {
        format(probability, ".12g"): float(value)
        for probability, value in zip(requested, np.atleast_1d(results), strict=True)
    }


def summarize(
    values: Iterable[object],
    *,
    probabilities: Iterable[Real] = (0.25, 0.5, 0.75),
    quantile_method: str = "linear",
    name: str = "values",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> SummaryResult:
    cleaned = clean_numeric(
        values,
        **_clean_options(name, outlier_rule, lower_bound, upper_bound, iqr_multiplier),
    )
    requested = _probabilities(probabilities)
    try:
        values_at_quantiles = np.quantile(
            cleaned.values, requested, method=quantile_method
        )
    except ValueError as exc:
        raise ValueError(f"unsupported quantile method: {quantile_method!r}") from exc
    return SummaryResult(
        n=len(cleaned.values),
        median=float(np.median(cleaned.values)),
        quantiles={
            format(probability, ".12g"): float(value)
            for probability, value in zip(
                requested, np.atleast_1d(values_at_quantiles), strict=True
            )
        },
        quantile_method=quantile_method,
        filter_report=cleaned.report,
    )


def _bootstrap_settings(
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, int, int]:
    confidence_value = _finite_real(confidence)
    if confidence_value is None or not 0 < confidence_value < 1:
        raise ValueError("confidence must be a finite number strictly between 0 and 1")
    if isinstance(resamples, (bool, np.bool_)) or not isinstance(resamples, Integral):
        raise TypeError("resamples must be an integer")
    if int(resamples) < 1:
        raise ValueError("resamples must be >= 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return confidence_value, int(resamples), int(seed)


def _resolved_size(sample_size: BootstrapSize, observed: int, name: str) -> int:
    if sample_size == "observed":
        return observed
    if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, Integral):
        raise TypeError(f"{name} must be a positive integer or 'observed'")
    if int(sample_size) < 1:
        raise ValueError(f"{name} must be >= 1")
    return int(sample_size)


def _statistic_value(
    statistic: Statistic, values: np.ndarray, statistic_name: str
) -> float:
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    if not isinstance(statistic_name, str) or not statistic_name.strip():
        raise ValueError("statistic_name must be a non-empty string")
    try:
        result = statistic(values)
    except Exception as exc:
        raise ValueError(f"statistic {statistic_name!r} failed") from exc
    numeric = _finite_real(result)
    if numeric is None:
        raise ValueError(f"statistic {statistic_name!r} must return one finite real number")
    return numeric


def _percentile_interval(
    bootstrap_values: np.ndarray, confidence: float
) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_values, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_ci(
    values: Iterable[object],
    *,
    statistic: Statistic = np.median,
    statistic_name: str = "median",
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    sample_size: BootstrapSize = "observed",
    name: str = "values",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> BootstrapResult:
    """Percentile bootstrap CI for a scalar statistic.

    ``sample_size='observed'`` means the post-filter sample size. The resolved
    integer, RNG seed, confidence level, and resample count are all recorded.
    """

    confidence_value, resample_count, seed_value = _bootstrap_settings(
        confidence, resamples, seed
    )
    cleaned = clean_numeric(
        values,
        **_clean_options(name, outlier_rule, lower_bound, upper_bound, iqr_multiplier),
    )
    array = np.asarray(cleaned.values, dtype=float)
    resolved_size = _resolved_size(sample_size, len(array), "sample_size")
    effect = _statistic_value(statistic, array, statistic_name)
    rng = np.random.default_rng(seed_value)
    bootstrapped = np.empty(resample_count, dtype=float)
    for index in range(resample_count):
        sample = array[rng.integers(0, len(array), size=resolved_size)]
        bootstrapped[index] = _statistic_value(statistic, sample, statistic_name)
    low, high = _percentile_interval(bootstrapped, confidence_value)
    return BootstrapResult(
        statistic=statistic_name,
        effect_size=effect,
        ci_low=low,
        ci_high=high,
        confidence=confidence_value,
        n=len(array),
        sample_size=resolved_size,
        resamples=resample_count,
        seed=seed_value,
        ci_method="percentile",
        filter_report=cleaned.report,
    )


def _clean_pairs(
    values_a: Iterable[object],
    values_b: Iterable[object],
    *,
    outlier_rule: OutlierRule,
    lower_bound: float | None,
    upper_bound: float | None,
    iqr_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, PairedFilterReport]:
    raw_a = _materialize(values_a, "a")
    raw_b = _materialize(values_b, "b")
    if len(raw_a) != len(raw_b):
        raise ValueError(
            "paired samples must have equal input lengths before filtering; "
            f"got {len(raw_a)} and {len(raw_b)}"
        )
    valid: list[tuple[int, float, float]] = []
    invalid_indices: list[int] = []
    invalid_a = 0
    invalid_b = 0
    for index, (item_a, item_b) in enumerate(zip(raw_a, raw_b, strict=True)):
        numeric_a = _finite_real(item_a)
        numeric_b = _finite_real(item_b)
        invalid_a += numeric_a is None
        invalid_b += numeric_b is None
        if numeric_a is None or numeric_b is None:
            invalid_indices.append(index)
        else:
            valid.append((index, numeric_a, numeric_b))
    if not valid:
        raise ValueError(
            "paired samples have no complete finite pairs "
            f"({len(invalid_indices)} invalid pairs of {len(raw_a)})"
        )
    finite_a = np.asarray([item[1] for item in valid], dtype=float)
    finite_b = np.asarray([item[2] for item in valid], dtype=float)
    low_a, high_a, rule_a = _outlier_bounds(
        finite_a,
        outlier_rule=outlier_rule,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iqr_multiplier=iqr_multiplier,
    )
    low_b, high_b, rule_b = _outlier_bounds(
        finite_b,
        outlier_rule=outlier_rule,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iqr_multiplier=iqr_multiplier,
    )
    kept_a: list[float] = []
    kept_b: list[float] = []
    outlier_indices: list[int] = []
    for original_index, item_a, item_b in valid:
        outlier_a = (low_a is not None and item_a < low_a) or (
            high_a is not None and item_a > high_a
        )
        outlier_b = (low_b is not None and item_b < low_b) or (
            high_b is not None and item_b > high_b
        )
        if outlier_a or outlier_b:
            outlier_indices.append(original_index)
        else:
            kept_a.append(item_a)
            kept_b.append(item_b)
    if not kept_a:
        raise ValueError(
            "paired samples have no pairs after filtering "
            f"({len(invalid_indices)} invalid, {len(outlier_indices)} outlier pairs)"
        )
    report = PairedFilterReport(
        input_pairs=len(raw_a),
        invalid_pairs=len(invalid_indices),
        invalid_values_a=invalid_a,
        invalid_values_b=invalid_b,
        outlier_pairs=len(outlier_indices),
        kept_pairs=len(kept_a),
        invalid_pair_indices=tuple(invalid_indices),
        outlier_pair_indices=tuple(outlier_indices),
        outlier_rule_a=rule_a,
        outlier_rule_b=rule_b,
        lower_bound_a=low_a,
        upper_bound_a=high_a,
        lower_bound_b=low_b,
        upper_bound_b=high_b,
    )
    return np.asarray(kept_a), np.asarray(kept_b), report


def _independent_sizes(
    sample_size: BootstrapSize | tuple[int, int], n_a: int, n_b: int
) -> tuple[int, int]:
    if sample_size == "observed":
        return n_a, n_b
    if isinstance(sample_size, Integral) and not isinstance(sample_size, (bool, np.bool_)):
        shared_size = _resolved_size(sample_size, n_a, "sample_size")
        return shared_size, shared_size
    if isinstance(sample_size, tuple) and len(sample_size) == 2:
        return (
            _resolved_size(sample_size[0], n_a, "sample_size[0]"),
            _resolved_size(sample_size[1], n_b, "sample_size[1]"),
        )
    raise TypeError(
        "independent sample_size must be 'observed', one positive integer applied "
        "to both groups, or a two-integer tuple (size_a, size_b)"
    )


def difference_in_statistic(
    values_a: Iterable[object],
    values_b: Iterable[object],
    *,
    statistic: Statistic,
    statistic_name: str,
    paired: bool = False,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    sample_size: BootstrapSize | tuple[int, int] = "observed",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> DifferenceResult:
    """Bootstrap ``statistic(a) - statistic(b)`` with an explicit direction.

    Independent samples are resampled separately. Paired samples are filtered
    pairwise and resampled using the same index vector; this preserves pairing.
    """

    if not isinstance(paired, bool):
        raise TypeError("paired must be a bool")
    confidence_value, resample_count, seed_value = _bootstrap_settings(
        confidence, resamples, seed
    )
    rng = np.random.default_rng(seed_value)
    bootstrapped = np.empty(resample_count, dtype=float)

    if paired:
        array_a, array_b, paired_report = _clean_pairs(
            values_a,
            values_b,
            outlier_rule=outlier_rule,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            iqr_multiplier=iqr_multiplier,
        )
        if isinstance(sample_size, tuple):
            raise TypeError("paired sample_size must be one integer or 'observed'")
        resolved_a = resolved_b = _resolved_size(
            sample_size, len(array_a), "sample_size"
        )
        effect = _statistic_value(statistic, array_a, statistic_name) - _statistic_value(
            statistic, array_b, statistic_name
        )
        for index in range(resample_count):
            indices = rng.integers(0, len(array_a), size=resolved_a)
            bootstrapped[index] = _statistic_value(
                statistic, array_a[indices], statistic_name
            ) - _statistic_value(statistic, array_b[indices], statistic_name)
        report_a = report_b = None
        n_pairs: int | None = len(array_a)
        mode: Literal["independent", "paired"] = "paired"
    else:
        cleaned_a = clean_numeric(
            values_a,
            **_clean_options("a", outlier_rule, lower_bound, upper_bound, iqr_multiplier),
        )
        cleaned_b = clean_numeric(
            values_b,
            **_clean_options("b", outlier_rule, lower_bound, upper_bound, iqr_multiplier),
        )
        array_a = np.asarray(cleaned_a.values, dtype=float)
        array_b = np.asarray(cleaned_b.values, dtype=float)
        resolved_a, resolved_b = _independent_sizes(
            sample_size, len(array_a), len(array_b)
        )
        effect = _statistic_value(statistic, array_a, statistic_name) - _statistic_value(
            statistic, array_b, statistic_name
        )
        for index in range(resample_count):
            sample_a = array_a[rng.integers(0, len(array_a), size=resolved_a)]
            sample_b = array_b[rng.integers(0, len(array_b), size=resolved_b)]
            bootstrapped[index] = _statistic_value(
                statistic, sample_a, statistic_name
            ) - _statistic_value(statistic, sample_b, statistic_name)
        report_a = cleaned_a.report
        report_b = cleaned_b.report
        paired_report = None
        n_pairs = None
        mode = "independent"

    low, high = _percentile_interval(bootstrapped, confidence_value)
    return DifferenceResult(
        statistic=statistic_name,
        effect_size=float(effect),
        ci_low=low,
        ci_high=high,
        confidence=confidence_value,
        mode=mode,
        direction="a_minus_b",
        n_a=len(array_a),
        n_b=len(array_b),
        n_pairs=n_pairs,
        sample_size_a=resolved_a,
        sample_size_b=resolved_b,
        resamples=resample_count,
        seed=seed_value,
        ci_method="percentile",
        filter_report_a=report_a,
        filter_report_b=report_b,
        paired_filter_report=paired_report,
    )


def difference_in_medians(
    values_a: Iterable[object],
    values_b: Iterable[object],
    *,
    paired: bool = False,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    sample_size: BootstrapSize | tuple[int, int] = "observed",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> DifferenceResult:
    return difference_in_statistic(
        values_a,
        values_b,
        statistic=np.median,
        statistic_name="median",
        paired=paired,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        sample_size=sample_size,
        outlier_rule=outlier_rule,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iqr_multiplier=iqr_multiplier,
    )


def difference_in_drift(
    drifts_a: Iterable[object],
    drifts_b: Iterable[object],
    *,
    center: Literal["median", "mean"] = "median",
    paired: bool = False,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    sample_size: BootstrapSize | tuple[int, int] = "observed",
    outlier_rule: OutlierRule = "none",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    iqr_multiplier: float = 1.5,
) -> DifferenceResult:
    """Compare already-computed per-trajectory drifts (group a minus b)."""

    if center == "median":
        statistic: Statistic = np.median
    elif center == "mean":
        statistic = np.mean
    else:
        raise ValueError("center must be 'median' or 'mean'")
    return difference_in_statistic(
        drifts_a,
        drifts_b,
        statistic=statistic,
        statistic_name=f"{center}_drift",
        paired=paired,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        sample_size=sample_size,
        outlier_rule=outlier_rule,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iqr_multiplier=iqr_multiplier,
    )


__all__ = [
    "BootstrapResult",
    "CleanedSample",
    "DifferenceResult",
    "FilterReport",
    "PairedFilterReport",
    "SummaryResult",
    "bootstrap_ci",
    "clean_numeric",
    "difference_in_drift",
    "difference_in_medians",
    "difference_in_statistic",
    "median",
    "quantiles",
    "summarize",
]
