import copy
import json
import unittest
from pathlib import Path

from experiments.E03_sentence_resampling.design import (
    audit_design,
    load_design,
    phase_budget,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "E03_sentence_resampling" / "configs" / "qwen122b_design.v0.1.json"


class SentenceResamplingDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = load_design(CONFIG)

    def test_committed_design_passes_offline_and_remains_disabled(self) -> None:
        report = audit_design(self.design)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["sampling_enabled"])
        self.assertEqual(report["implementation_status"], "NOT_READY_FOR_REAL_SAMPLING")
        self.assertIn("real_backend_adapter_implemented", report["open_readiness_gates"])
        self.assertEqual(
            report["cost_status"],
            "BLOCKED_PENDING_PILOT_TOKEN_MEASUREMENT_AND_HARD_USD_CAP",
        )
        self.assertGreaterEqual(len(report["passed_invariants"]), 8)

    def test_request_budgets_are_derived_from_nested_design(self) -> None:
        pilot = phase_budget(self.design, "technical_pilot")
        self.assertEqual(pilot.generated_arms, 6)
        self.assertEqual(pilot.accepted_replacement_calls, 36)
        self.assertEqual(pilot.maximum_replacement_calls, 72)
        self.assertEqual(pilot.original_continuation_calls, 18)
        self.assertEqual(pilot.replacement_continuation_calls, 108)
        self.assertEqual(pilot.planned_calls, 162)
        self.assertEqual(pilot.hard_call_upper_bound, 218)

        main = phase_budget(self.design, "main")
        self.assertEqual(main.accepted_replacement_calls, 288)
        self.assertEqual(main.maximum_replacement_calls, 576)
        self.assertEqual(main.original_continuation_calls, 144)
        self.assertEqual(main.replacement_continuation_calls, 1152)
        self.assertEqual(main.planned_calls, 1584)
        self.assertEqual(main.hard_call_upper_bound, 2060)

    def test_enabling_sampling_or_provider_fallback_fails_closed(self) -> None:
        enabled = copy.deepcopy(self.design)
        enabled["execution"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "sampling-disabled"):
            validate_design(enabled)

        fallback = copy.deepcopy(self.design)
        fallback["target"]["allow_provider_fallbacks"] = True
        with self.assertRaisesRegex(ValueError, "provider must be pinned"):
            validate_design(fallback)

    def test_hidden_state_overclaim_and_missing_original_arm_fail(self) -> None:
        overclaim = copy.deepcopy(self.design)
        overclaim["causal_scope"]["claims_hidden_state_intervention"] = True
        with self.assertRaisesRegex(ValueError, "must be false"):
            validate_design(overclaim)

        no_original = copy.deepcopy(self.design)
        no_original["targets"][0]["arms"] = [
            arm for arm in no_original["targets"][0]["arms"] if arm["arm_id"] != "original"
        ]
        with self.assertRaisesRegex(ValueError, "exactly one original replay"):
            validate_design(no_original)

    def test_post_treatment_recovery_filter_fails(self) -> None:
        invalid = copy.deepcopy(self.design)
        invalid["analysis"]["recovery_conditioned_primary_analysis"] = True
        with self.assertRaisesRegex(ValueError, "post-treatment recovery"):
            validate_design(invalid)

    def test_cross_field_budget_inconsistencies_fail(self) -> None:
        underpowered_pilot = copy.deepcopy(self.design)
        underpowered_pilot["phases"]["technical_pilot"][
            "accepted_candidates_per_generated_arm"
        ] = 2
        with self.assertRaisesRegex(ValueError, "pilot candidate count"):
            validate_design(underpowered_pilot)

        unbalanced_main = copy.deepcopy(self.design)
        unbalanced_main["phases"]["main"]["sources_per_condition"] = 5
        with self.assertRaisesRegex(ValueError, "balanced across two conditions"):
            validate_design(unbalanced_main)

        undercounted_arm = copy.deepcopy(self.design)
        undercounted_arm["targets"][1]["arms"][1]["generated"] = False
        with self.assertRaisesRegex(ValueError, "arm id/kind/generated"):
            validate_design(undercounted_arm)

        wrong_kind = copy.deepcopy(self.design)
        wrong_kind["targets"][0]["arms"][1]["kind"] = "generic_paraphrase"
        with self.assertRaisesRegex(ValueError, "arm id/kind/generated"):
            validate_design(wrong_kind)

        unrecorded_rejections = copy.deepcopy(self.design)
        unrecorded_rejections["replacement_policy"]["record_all_rejections"] = False
        with self.assertRaisesRegex(ValueError, "record_all_rejections"):
            validate_design(unrecorded_rejections)

    def test_json_is_round_trip_stable(self) -> None:
        reloaded = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(self.design, reloaded)


if __name__ == "__main__":
    unittest.main()
