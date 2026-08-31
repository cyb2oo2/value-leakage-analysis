from __future__ import annotations

import json
import unittest

import numpy as np

from research.statistics import (
    bootstrap_ci,
    clean_numeric,
    difference_in_drift,
    difference_in_medians,
    difference_in_statistic,
    median,
    quantiles,
    summarize,
)


class CleaningTests(unittest.TestCase):
    def test_finite_cleaning_and_bounds_report_every_drop(self) -> None:
        cleaned = clean_numeric(
            [1, None, np.nan, True, 2, np.inf, 100],
            outlier_rule="bounds",
            lower_bound=0,
            upper_bound=10,
        )
        self.assertEqual(cleaned.values, (1.0, 2.0))
        self.assertEqual(cleaned.report.input_count, 7)
        self.assertEqual(cleaned.report.invalid_indices, (1, 2, 3, 5))
        self.assertEqual(cleaned.report.outlier_indices, (6,))
        self.assertEqual(cleaned.report.kept_count, 2)
        self.assertEqual(cleaned.report.valid_count, 2)
        json.dumps(cleaned.to_dict(), allow_nan=False)

    def test_iqr_filter_is_explicit(self) -> None:
        cleaned = clean_numeric([1, 2, 2, 3, 100], outlier_rule="iqr")
        self.assertEqual(cleaned.values, (1.0, 2.0, 2.0, 3.0))
        self.assertEqual(cleaned.report.outlier_count, 1)
        self.assertIn("iqr", cleaned.report.outlier_rule)

    def test_ambiguous_and_empty_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            clean_numeric([])
        with self.assertRaisesRegex(TypeError, "ordered one-dimensional"):
            clean_numeric({1, 2})
        with self.assertRaisesRegex(ValueError, "no finite numeric"):
            clean_numeric([None, "3", np.nan])
        with self.assertRaisesRegex(ValueError, "require.*outlier_rule"):
            clean_numeric([1, 2], upper_bound=2)
        with self.assertRaisesRegex(ValueError, "no observations after filtering"):
            clean_numeric([1, 2], outlier_rule="bounds", lower_bound=3)


class DescriptiveTests(unittest.TestCase):
    def test_median_quantiles_and_summary(self) -> None:
        self.assertEqual(median([3, 1, 2]), 2.0)
        self.assertEqual(
            quantiles([0, 10], probabilities=(0, 0.5, 1)),
            {"0": 0.0, "0.5": 5.0, "1": 10.0},
        )
        result = summarize([0, 10, None], probabilities=(0.5,))
        self.assertEqual(result.n, 2)
        self.assertEqual(result.median, 5.0)
        self.assertEqual(result.filter_report.invalid_count, 1)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_duplicate_or_invalid_quantiles_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            quantiles([1, 2], (0.5, 0.5))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            quantiles([1, 2], (-0.1,))


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_ci_is_seeded_and_fully_described(self) -> None:
        first = bootstrap_ci(
            [1, 2, 3, 4, 5], confidence=0.9, resamples=300, seed=17
        )
        second = bootstrap_ci(
            [1, 2, 3, 4, 5], confidence=0.9, resamples=300, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first.effect_size, 3.0)
        self.assertEqual(first.n, 5)
        self.assertEqual(first.sample_size, 5)
        self.assertEqual(first.confidence, 0.9)
        self.assertEqual(first.resamples, 300)
        self.assertEqual(first.seed, 17)
        self.assertLessEqual(first.ci_low, first.effect_size)
        self.assertGreaterEqual(first.ci_high, first.effect_size)
        json.dumps(first.to_dict(), allow_nan=False)

    def test_custom_statistic_and_sample_size(self) -> None:
        result = bootstrap_ci(
            [1, 2, 9],
            statistic=np.mean,
            statistic_name="mean",
            sample_size=2,
            resamples=50,
            seed=2,
        )
        self.assertAlmostEqual(result.effect_size, 4.0)
        self.assertEqual(result.sample_size, 2)

    def test_invalid_bootstrap_configuration_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between"):
            bootstrap_ci([1], confidence=1, resamples=10, seed=0)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            bootstrap_ci([1], confidence=0.95, resamples=0, seed=0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bootstrap_ci([1], confidence=0.95, resamples=10, seed=-1)


class DifferenceTests(unittest.TestCase):
    def test_independent_difference_in_medians_is_a_minus_b(self) -> None:
        result = difference_in_medians(
            [10, 11, 12], [1, 2, 3], confidence=0.9, resamples=250, seed=9
        )
        self.assertEqual(result.effect_size, 9.0)
        self.assertEqual(result.direction, "a_minus_b")
        self.assertEqual(result.mode, "independent")
        self.assertEqual(result.n, {"a": 3, "b": 3})
        self.assertIsNotNone(result.filter_report_a)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_paired_resampling_drops_incomplete_pairs_and_preserves_alignment(self) -> None:
        result = difference_in_medians(
            [10, None, 30, 40],
            [1, 2, 3, 4],
            paired=True,
            confidence=0.9,
            resamples=200,
            seed=4,
        )
        self.assertEqual(result.effect_size, 27.0)
        self.assertEqual(result.n, {"a": 3, "b": 3, "pairs": 3})
        self.assertEqual(result.paired_filter_report.invalid_pairs, 1)
        self.assertEqual(result.paired_filter_report.invalid_pair_indices, (1,))

    def test_paired_length_mismatch_is_rejected_before_analysis(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal input lengths"):
            difference_in_medians([1, 2], [1], paired=True, resamples=10, seed=0)

    def test_generic_statistic_and_drift_wrapper(self) -> None:
        generic = difference_in_statistic(
            [2, 4],
            [1, 1],
            statistic=np.mean,
            statistic_name="mean",
            resamples=100,
            seed=3,
        )
        drift = difference_in_drift(
            [2, 4], [1, 1], center="mean", resamples=100, seed=3
        )
        self.assertEqual(generic.effect_size, 2.0)
        self.assertEqual(drift.effect_size, 2.0)
        self.assertEqual(drift.statistic, "mean_drift")

    def test_one_independent_bootstrap_size_applies_to_both_groups(self) -> None:
        result = difference_in_medians(
            [1, 2, 3], [0, 1], sample_size=4, resamples=20, seed=1
        )
        self.assertEqual(result.sample_size_a, 4)
        self.assertEqual(result.sample_size_b, 4)


if __name__ == "__main__":
    unittest.main()
