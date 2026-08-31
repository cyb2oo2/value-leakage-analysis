"""Offline audit for the E03 preregistration design.

This module never imports an API client, reads environment variables, or sends
requests. It validates fail-closed invariants and computes request counts from
the machine-readable design rather than relying on prose or magic constants.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DESIGN_SCHEMA_VERSION = "0.1"
DISABLED_STATUS = "design_only_sampling_disabled"


@dataclass(frozen=True)
class PhaseBudget:
    phase: str
    sources: int
    target_types: int
    generated_arms: int
    accepted_replacement_calls: int
    maximum_replacement_calls: int
    original_continuation_calls: int
    replacement_continuation_calls: int
    planned_calls: int
    maximum_calls_before_retries: int
    retry_call_cap: int
    hard_call_upper_bound: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "phase": self.phase,
            "sources": self.sources,
            "target_types": self.target_types,
            "generated_arms": self.generated_arms,
            "accepted_replacement_calls": self.accepted_replacement_calls,
            "maximum_replacement_calls": self.maximum_replacement_calls,
            "original_continuation_calls": self.original_continuation_calls,
            "replacement_continuation_calls": self.replacement_continuation_calls,
            "planned_calls": self.planned_calls,
            "maximum_calls_before_retries": self.maximum_calls_before_retries,
            "retry_call_cap": self.retry_call_cap,
            "hard_call_upper_bound": self.hard_call_upper_bound,
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rate(value: Any, name: str, *, upper_inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    upper_ok = number <= 1 if upper_inclusive else number < 1
    if number < 0 or not upper_ok:
        raise ValueError(f"{name} must be in [0, 1{' ' if upper_inclusive else ')'}]")
    return number


def load_design(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("design config must be a JSON object")
    return value


def validate_design(design: Mapping[str, Any]) -> list[str]:
    """Return passed invariant names; raise on the first fail-closed violation."""

    passed: list[str] = []
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {DESIGN_SCHEMA_VERSION!r}")
    passed.append("schema_version")

    execution = _mapping(design.get("execution"), "execution")
    if design.get("status") != DISABLED_STATUS or execution.get("enabled") is not False:
        raise ValueError("design must remain sampling-disabled")
    if execution.get("requires_explicit_user_authorization") is not True:
        raise ValueError("explicit user authorization must be required")
    if execution.get("read_api_keys") is not False:
        raise ValueError("offline design audit must not read API keys")
    passed.append("offline_sampling_disabled")

    readiness = _mapping(design.get("readiness"), "readiness")
    if not readiness or any(value is not False for value in readiness.values()):
        raise ValueError("v0.1 readiness gates must explicitly remain false")
    passed.append("not_misrepresented_as_execution_ready")

    target = _mapping(design.get("target"), "target")
    if target.get("model_id") != "qwen/qwen3.5-122b-a10b":
        raise ValueError("unexpected target model")
    if target.get("provider") != "deepinfra/fp4" or target.get("allow_provider_fallbacks") is not False:
        raise ValueError("provider must be pinned to deepinfra/fp4 with fallback disabled")
    passed.append("target_and_provider_pin")

    scope = _mapping(design.get("causal_scope"), "causal_scope")
    required_true = (
        "exact_task_messages_required",
        "verbatim_prefix_bytes_required",
        "target_character_span_required",
        "prefix_sha256_required",
        "original_replay_distribution_required",
    )
    for field in required_true:
        if scope.get(field) is not True:
            raise ValueError(f"causal_scope.{field} must be true")
    forbidden_claims = (
        "claims_hidden_state_intervention",
        "claims_natural_mediation",
        "single_observed_source_answer_is_control",
    )
    for field in forbidden_claims:
        if scope.get(field) is not False:
            raise ValueError(f"causal_scope.{field} must be false")
    passed.append("causal_claim_boundary")

    source_selection = _mapping(design.get("source_selection"), "source_selection")
    main_sources = _positive_int(source_selection.get("main_sources"), "main_sources")
    sources_per_condition = _positive_int(
        source_selection.get("sources_per_condition"), "sources_per_condition"
    )
    confirmatory_sources = _positive_int(
        source_selection.get("confirmatory_sources_per_condition"),
        "confirmatory_sources_per_condition",
    )
    minimum_case_series = _positive_int(
        source_selection.get("minimum_case_series_sources_per_condition"),
        "minimum_case_series_sources_per_condition",
    )
    if main_sources != 2 * sources_per_condition:
        raise ValueError("main_sources must equal two balanced condition cells")
    if confirmatory_sources != sources_per_condition:
        raise ValueError("confirmatory subtype count must equal the main per-condition count")
    if minimum_case_series >= confirmatory_sources:
        raise ValueError("case-series minimum must be below the confirmatory count")
    if source_selection.get("primary_policy_subtype") != "impartiality_commitment":
        raise ValueError("the v0.1 confirmatory subtype must be impartiality_commitment")
    passed.append("balanced_primary_subtype_gate")

    targets = design.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("targets must be a non-empty array")
    target_types: set[str] = set()
    expected_arms = {
        "explicit_policy": {
            "original": ("original_replay", False),
            "preserve": ("policy_preserving_paraphrase", True),
            "counterpolicy": (
                "strategic_to_impartial_or_impartial_commitment_removed_by_frozen_subtype_rule",
                True,
            ),
        },
        "numerical_pivot": {
            "original": ("original_replay", False),
            "same_number_sham": ("meaning_preserving_same_number", True),
            "plausible_low": ("condition_symmetric_plausible_lower_value", True),
            "plausible_high": ("condition_symmetric_plausible_higher_value", True),
        },
        "ordinary_control": {
            "original": ("original_replay", False),
            "meaning_preserving_sham": ("meaning_preserving_paraphrase", True),
        },
    }
    observed_arms: dict[str, dict[str, tuple[str, bool]]] = {}
    for target_index, target_raw in enumerate(targets):
        target_item = _mapping(target_raw, f"targets[{target_index}]")
        target_type = str(target_item.get("target_type", ""))
        if not target_type or target_type in target_types:
            raise ValueError("target_type values must be non-empty and unique")
        target_types.add(target_type)
        arms = target_item.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ValueError(f"target {target_type!r} must define arms")
        arm_ids: set[str] = set()
        target_arm_map: dict[str, tuple[str, bool]] = {}
        original_count = 0
        for arm_index, arm_raw in enumerate(arms):
            arm = _mapping(arm_raw, f"{target_type}.arms[{arm_index}]")
            arm_id = str(arm.get("arm_id", ""))
            if not arm_id or arm_id in arm_ids:
                raise ValueError(f"arm ids for {target_type!r} must be non-empty and unique")
            arm_ids.add(arm_id)
            if not isinstance(arm.get("generated"), bool):
                raise ValueError(f"arm {target_type}.{arm_id} generated must be boolean")
            target_arm_map[arm_id] = (str(arm.get("kind", "")), arm["generated"])
            if arm.get("kind") == "original_replay" and arm.get("generated") is False:
                original_count += 1
        if original_count != 1:
            raise ValueError(f"target {target_type!r} must have exactly one original replay arm")
        observed_arms[target_type] = target_arm_map
    required_targets = {"explicit_policy", "numerical_pivot", "ordinary_control"}
    if target_types != required_targets:
        raise ValueError(f"targets must be exactly {sorted(required_targets)}")
    if observed_arms != expected_arms:
        raise ValueError("target arm id/kind/generated triples do not match the frozen design")
    explicit_target = next(item for item in targets if item["target_type"] == "explicit_policy")
    if explicit_target.get("primary_contrast") != "impartiality_commitment_removed_minus_preserved":
        raise ValueError("unexpected confirmatory explicit-policy contrast")
    passed.append("target_arms_and_original_controls")

    replacement = _mapping(design.get("replacement_policy"), "replacement_policy")
    required_replacement_values = {
        "primary_generation_model": "same_as_target",
        "on_policy_prefix_conditioned": True,
        "freeze_candidates_before_continuations": True,
        "outcome_blind_validation": True,
        "record_all_rejections": True,
        "human_written_replacements_primary": False,
    }
    for field, expected in required_replacement_values.items():
        if replacement.get(field) != expected:
            raise ValueError(f"replacement_policy.{field} must equal {expected!r}")
    _positive_int(
        replacement.get("maximum_attempts_per_accepted_candidate"),
        "maximum_attempts_per_accepted_candidate",
    )
    minimum_candidates = _positive_int(
        replacement.get("minimum_accepted_candidates_per_generated_arm"),
        "minimum_accepted_candidates_per_generated_arm",
    )
    passed.append("replacement_freeze_and_validation")

    phases = _mapping(design.get("phases"), "phases")
    for phase_name in ("capability_gate", "technical_pilot", "main"):
        phase = _mapping(phases.get(phase_name), f"phases.{phase_name}")
        if phase.get("enabled") is not False:
            raise ValueError(f"phase {phase_name!r} must remain disabled")
    if _positive_int(
        _mapping(phases["capability_gate"], "capability_gate").get("maximum_total_calls"),
        "capability_gate.maximum_total_calls",
    ) > 12:
        raise ValueError("capability gate may use at most 12 calls")
    main = _mapping(phases["main"], "phases.main")
    pilot = _mapping(phases["technical_pilot"], "phases.technical_pilot")
    for phase_name, phase in (("technical_pilot", pilot), ("main", main)):
        phase_sources = _positive_int(phase.get("sources"), f"{phase_name}.sources")
        phase_per_condition = _positive_int(
            phase.get("sources_per_condition"), f"{phase_name}.sources_per_condition"
        )
        if phase_sources != 2 * phase_per_condition:
            raise ValueError(f"{phase_name} sources must be balanced across two conditions")
    if _positive_int(main.get("sources"), "main.sources") != main_sources:
        raise ValueError("main phase source count must match source_selection.main_sources")
    if _positive_int(
        pilot.get("accepted_candidates_per_generated_arm"), "pilot candidates"
    ) < minimum_candidates:
        raise ValueError("pilot candidate count is below the replacement-policy minimum")
    if _positive_int(main.get("accepted_candidates_per_generated_arm"), "main candidates") < minimum_candidates:
        raise ValueError("main candidate count is below the replacement-policy minimum")
    passed.append("all_paid_phases_disabled")

    analysis = _mapping(design.get("analysis"), "analysis")
    if analysis.get("top_level_unit") != "source_rollout" or analysis.get("equal_source_weight") is not True:
        raise ValueError("source rollout must be the equally weighted top-level unit")
    if analysis.get("recovery_conditioned_primary_analysis") is not False:
        raise ValueError("primary analysis cannot condition on post-treatment recovery")
    bootstrap = _mapping(analysis.get("bootstrap"), "analysis.bootstrap")
    if bootstrap.get("resample_sources_within_condition") is not True:
        raise ValueError("bootstrap must resample sources within condition")
    passed.append("source_level_inference")

    gates = _mapping(design.get("technical_gates"), "technical_gates")
    _rate(gates.get("minimum_request_completion_rate"), "minimum_request_completion_rate")
    _rate(gates.get("minimum_overall_parse_rate"), "minimum_overall_parse_rate")
    _rate(gates.get("maximum_parse_rate_arm_gap"), "maximum_parse_rate_arm_gap")
    _rate(gates.get("minimum_candidate_validity_rate"), "minimum_candidate_validity_rate")
    _rate(gates.get("maximum_truncation_rate"), "maximum_truncation_rate")
    if gates.get("advance_based_on_effect_sign_or_size") is not False:
        raise ValueError("advancement cannot depend on a favorable research effect")
    if _positive_int(
        gates.get("minimum_valid_candidates_per_target_arm"),
        "minimum_valid_candidates_per_target_arm",
    ) != minimum_candidates:
        raise ValueError("technical candidate gate must equal the replacement-policy minimum")
    passed.append("outcome_independent_gates")

    budget = _mapping(design.get("budget"), "budget")
    if budget.get("sampling_must_remain_disabled_without_hard_cap") is not True:
        raise ValueError("missing hard budget must keep sampling disabled")
    if budget.get("hard_usd_cap") is not None:
        cap = budget["hard_usd_cap"]
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
            raise ValueError("hard_usd_cap must be null or a positive number")
    passed.append("budget_gate")
    return passed


def phase_budget(design: Mapping[str, Any], phase_name: str) -> PhaseBudget:
    validate_design(design)
    if phase_name not in {"technical_pilot", "main"}:
        raise ValueError("phase_name must be technical_pilot or main")

    phase = _mapping(_mapping(design["phases"], "phases")[phase_name], f"phases.{phase_name}")
    replacement = _mapping(design["replacement_policy"], "replacement_policy")
    targets = design["targets"]
    sources = _positive_int(phase.get("sources"), f"{phase_name}.sources")
    replacements = _positive_int(
        phase.get("accepted_candidates_per_generated_arm"),
        f"{phase_name}.accepted_candidates_per_generated_arm",
    )
    continuations = _positive_int(
        phase.get("continuations_per_candidate"),
        f"{phase_name}.continuations_per_candidate",
    )
    originals = _positive_int(
        phase.get("original_replays_per_target"),
        f"{phase_name}.original_replays_per_target",
    )
    max_attempts = _positive_int(
        replacement.get("maximum_attempts_per_accepted_candidate"),
        "maximum_attempts_per_accepted_candidate",
    )
    retry_fraction = _rate(phase.get("retry_fraction_cap"), f"{phase_name}.retry_fraction_cap")

    generated_arms = sum(
        1
        for target in targets
        for arm in target["arms"]
        if arm.get("generated") is True
    )
    target_count = len(targets)
    accepted_replacement_calls = sources * generated_arms * replacements
    maximum_replacement_calls = accepted_replacement_calls * max_attempts
    original_continuation_calls = sources * target_count * originals
    replacement_continuation_calls = accepted_replacement_calls * continuations
    planned_calls = (
        accepted_replacement_calls
        + original_continuation_calls
        + replacement_continuation_calls
    )
    maximum_calls_before_retries = (
        maximum_replacement_calls
        + original_continuation_calls
        + replacement_continuation_calls
    )
    retry_call_cap = math.ceil(maximum_calls_before_retries * retry_fraction)
    return PhaseBudget(
        phase=phase_name,
        sources=sources,
        target_types=target_count,
        generated_arms=generated_arms,
        accepted_replacement_calls=accepted_replacement_calls,
        maximum_replacement_calls=maximum_replacement_calls,
        original_continuation_calls=original_continuation_calls,
        replacement_continuation_calls=replacement_continuation_calls,
        planned_calls=planned_calls,
        maximum_calls_before_retries=maximum_calls_before_retries,
        retry_call_cap=retry_call_cap,
        hard_call_upper_bound=maximum_calls_before_retries + retry_call_cap,
    )


def audit_design(design: Mapping[str, Any]) -> dict[str, Any]:
    passed = validate_design(design)
    budget = _mapping(design["budget"], "budget")
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_id": design.get("design_id"),
        "status": "PASS",
        "sampling_enabled": False,
        "implementation_status": "NOT_READY_FOR_REAL_SAMPLING",
        "open_readiness_gates": list(_mapping(design["readiness"], "readiness")),
        "passed_invariants": passed,
        "phase_budgets": {
            name: phase_budget(design, name).to_dict()
            for name in ("technical_pilot", "main")
        },
        "cost_status": (
            "BLOCKED_PENDING_PILOT_TOKEN_MEASUREMENT_AND_HARD_USD_CAP"
            if budget.get("hard_usd_cap") is None
            else "HARD_CAP_RECORDED_BUT_SAMPLING_STILL_REQUIRES_AUTHORIZATION"
        ),
        "causal_claim": _mapping(design["causal_scope"], "causal_scope").get("estimand"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_design(load_design(args.config))
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"design audit failed: {exc}") from exc
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
