"""Provider-neutral sentence-resampling scaffold with a deterministic mock.

No real provider is implemented here. Most importantly, visible-text prefix
replay and arbitrary hidden-CoT/internal-state continuation are represented as
different capabilities and are never treated as interchangeable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from research.experiment_utils import (
    build_provenance,
    create_new_directory,
    file_manifest,
    sha256_file,
    write_new_json,
)


SCHEMA_VERSION = "1.1"


class ContinuationMode(str, Enum):
    """Experiment definitions with materially different causal meaning."""

    VISIBLE_TEXT_PREFIX_REPLAY = "visible_text_prefix_replay"
    HIDDEN_COT_INTERNAL_STATE = "hidden_cot_internal_state_continuation"


class UnsupportedContinuationCapability(RuntimeError):
    """Raised before sampling when the requested experiment is impossible."""


class BackendNotImplemented(RuntimeError):
    """Raised for configured real providers; this scaffold is mock-only."""


@dataclass(frozen=True)
class BackendCapabilities:
    replacement_generation: bool
    visible_text_prefix_replay: bool
    arbitrary_hidden_cot_internal_state_continuation: bool

    def supports(self, mode: ContinuationMode) -> bool:
        if mode is ContinuationMode.VISIBLE_TEXT_PREFIX_REPLAY:
            return self.visible_text_prefix_replay
        return self.arbitrary_hidden_cot_internal_state_continuation


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Sentence":
        start = value.get("start_char")
        end = value.get("end_char")
        for item, name in ((start, "start_char"), (end, "end_char")):
            if item is not None and (isinstance(item, bool) or not isinstance(item, Integral)):
                raise TypeError(f"sentence {name} must be an integer or null")
        return cls(
            sentence_id=str(value.get("sentence_id", "")),
            text=str(value.get("text", "")),
            start_char=(None if start is None else int(start)),
            end_char=(None if end is None else int(end)),
        )


@dataclass(frozen=True)
class SourceTrajectory:
    """A manually segmented, authoritative visible reasoning trajectory."""

    source_id: str
    condition: str
    threshold: float
    task_prompt: str
    sentences: tuple[Sentence, ...]
    visible_final_answer: str
    parsed_final_estimate: float | None
    visible_reasoning_text: str | None = None
    task_messages: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceTrajectory":
        estimate = value.get("parsed_final_estimate")
        threshold = value.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError("source threshold must be a real number")
        sentence_values = value.get("sentences")
        if not isinstance(sentence_values, list):
            raise TypeError("source sentences must be a JSON array")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("source metadata must be a JSON object")
        task_messages = value.get("task_messages", [])
        if not isinstance(task_messages, list) or any(
            not isinstance(message, Mapping) for message in task_messages
        ):
            raise TypeError("source task_messages must be a JSON array of objects")
        visible_reasoning = value.get("visible_reasoning_text")
        if visible_reasoning is not None and not isinstance(visible_reasoning, str):
            raise TypeError("source visible_reasoning_text must be a string or null")
        return cls(
            source_id=str(value.get("source_id", "")),
            condition=str(value.get("condition", "")),
            threshold=float(threshold),
            task_prompt=str(value.get("task_prompt", "")),
            sentences=tuple(Sentence.from_dict(item) for item in sentence_values),
            visible_final_answer=str(value.get("visible_final_answer", "")),
            parsed_final_estimate=(None if estimate is None else float(estimate)),
            visible_reasoning_text=visible_reasoning,
            task_messages=tuple(dict(message) for message in task_messages),
            metadata=dict(metadata),
        )

    def validate(self) -> None:
        if not self.source_id.strip() or not self.condition.strip() or not self.task_prompt.strip():
            raise ValueError("source_id, condition, and task_prompt must be non-empty")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("source threshold must be finite and positive")
        if not self.sentences:
            raise ValueError("manual source segmentation must contain at least one sentence")
        for index, sentence in enumerate(self.sentences, start=1):
            expected = f"S{index}"
            if sentence.sentence_id != expected:
                raise ValueError(
                    f"manual sentence ids must be contiguous S1..ST; expected {expected}, "
                    f"got {sentence.sentence_id!r}"
                )
            if not sentence.text.strip():
                raise ValueError(f"{expected} text must be non-empty")
            if (sentence.start_char is None) != (sentence.end_char is None):
                raise ValueError(f"{expected} must provide both start_char and end_char or neither")
        if self.parsed_final_estimate is not None and not math.isfinite(self.parsed_final_estimate):
            raise ValueError("parsed_final_estimate must be finite or null")
        if self.visible_reasoning_text is not None:
            previous_end = 0
            for sentence in self.sentences:
                if sentence.start_char is None or sentence.end_char is None:
                    raise ValueError(
                        "every sentence needs character spans when visible_reasoning_text is present"
                    )
                if not 0 <= sentence.start_char < sentence.end_char <= len(self.visible_reasoning_text):
                    raise ValueError(f"invalid character span for {sentence.sentence_id}")
                if sentence.start_char < previous_end:
                    raise ValueError("sentence character spans must be ordered and non-overlapping")
                if self.visible_reasoning_text[sentence.start_char:sentence.end_char] != sentence.text:
                    raise ValueError(
                        f"{sentence.sentence_id} text does not equal its verbatim character span"
                    )
                previous_end = sentence.end_char
        for message in self.task_messages:
            if not str(message.get("role", "")).strip() or "content" not in message:
                raise ValueError("every task message must contain a non-empty role and content")


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    model_id: str
    provider: str | None
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    prompt_version: str
    target_type: str
    semantic_arm_id: str
    continuation_mode: ContinuationMode
    target_sentence_index: int
    replacement_instruction: str
    n_replacements: int
    continuations_per_replacement: int
    original_replay_continuations: int
    require_verbatim_prefix: bool
    require_exact_task_messages: bool
    random_seed: int
    backend: BackendConfig

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        backend = value.get("backend", {})
        if not isinstance(backend, Mapping):
            raise TypeError("backend must be a JSON object")
        settings = backend.get("settings", {})
        if not isinstance(settings, Mapping):
            raise TypeError("backend.settings must be a JSON object")
        integer_fields = {
            name: value.get(name)
            for name in (
                "target_sentence_index",
                "n_replacements",
                "continuations_per_replacement",
                "random_seed",
            )
        }
        for name, item in integer_fields.items():
            if isinstance(item, bool) or not isinstance(item, Integral):
                raise TypeError(f"{name} must be an integer")
        original_replays = value.get(
            "original_replay_continuations", integer_fields["continuations_per_replacement"]
        )
        if isinstance(original_replays, bool) or not isinstance(original_replays, Integral):
            raise TypeError("original_replay_continuations must be an integer")
        require_verbatim = value.get("require_verbatim_prefix", False)
        if not isinstance(require_verbatim, bool):
            raise TypeError("require_verbatim_prefix must be a boolean")
        require_messages = value.get("require_exact_task_messages", False)
        if not isinstance(require_messages, bool):
            raise TypeError("require_exact_task_messages must be a boolean")
        return cls(
            experiment_id=str(value.get("experiment_id", "")),
            prompt_version=str(value.get("prompt_version", "")),
            target_type=str(value.get("target_type", "")),
            semantic_arm_id=str(value.get("semantic_arm_id", "")),
            continuation_mode=ContinuationMode(value.get("continuation_mode")),
            target_sentence_index=int(integer_fields["target_sentence_index"]),
            replacement_instruction=str(value.get("replacement_instruction", "")),
            n_replacements=int(integer_fields["n_replacements"]),
            continuations_per_replacement=int(integer_fields["continuations_per_replacement"]),
            original_replay_continuations=int(original_replays),
            require_verbatim_prefix=require_verbatim,
            require_exact_task_messages=require_messages,
            random_seed=int(integer_fields["random_seed"]),
            backend=BackendConfig(
                backend=str(backend.get("backend", "")),
                model_id=str(backend.get("model_id", "")),
                provider=(None if backend.get("provider") is None else str(backend["provider"])),
                settings=dict(settings),
            ),
        )

    def validate(self, source: SourceTrajectory) -> None:
        if not all(
            value.strip()
            for value in (
                self.experiment_id,
                self.prompt_version,
                self.target_type,
                self.semantic_arm_id,
            )
        ):
            raise ValueError(
                "experiment_id, prompt_version, target_type, and semantic_arm_id must be non-empty"
            )
        if not self.replacement_instruction.strip():
            raise ValueError("replacement_instruction must be non-empty")
        if (
            self.n_replacements < 1
            or self.continuations_per_replacement < 1
            or self.original_replay_continuations < 1
        ):
            raise ValueError(
                "n_replacements, continuations_per_replacement, and "
                "original_replay_continuations must be positive"
            )
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not 1 <= self.target_sentence_index <= len(source.sentences):
            raise ValueError("target_sentence_index is 1-based and must select one source sentence")
        if not self.backend.backend.strip() or not self.backend.model_id.strip():
            raise ValueError("backend.backend and backend.model_id must be non-empty")
        if self.require_verbatim_prefix and source.visible_reasoning_text is None:
            raise ValueError(
                "require_verbatim_prefix is true but source has no visible_reasoning_text/spans"
            )
        if self.require_exact_task_messages and not source.task_messages:
            raise ValueError(
                "require_exact_task_messages is true but source has no task_messages"
            )


@dataclass(frozen=True)
class ReplacementRequest:
    request_id: str
    source_id: str
    target_sentence_index: int
    target_sentence_id: str
    target_type: str
    semantic_arm_id: str
    task_prompt: str
    task_messages: tuple[Mapping[str, Any], ...]
    condition: str
    threshold: float
    preserved_prefix: tuple[Sentence, ...]
    original_target_sentence: Sentence
    replacement_instruction: str
    seed: int
    model_id: str
    backend: str
    provider: str | None
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class ReplacementResponse:
    replacement_text: str
    raw_response: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuationRequest:
    request_id: str
    replacement_request_id: str | None
    source_id: str
    target_type: str
    semantic_arm_id: str
    intervention_arm: str
    continuation_mode: ContinuationMode
    task_prompt: str
    task_messages: tuple[Mapping[str, Any], ...]
    condition: str
    threshold: float
    preserved_prefix: tuple[Sentence, ...]
    replacement_sentence: Sentence
    visible_prefix_replay: str
    visible_prefix_sha256: str
    prefix_construction: str
    seed: int
    model_id: str
    backend: str
    provider: str | None
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuationResponse:
    continuation_text: str
    visible_final_answer: str
    parsed_final_estimate: float | None
    raw_response: Mapping[str, Any]


@runtime_checkable
class SentenceResamplingBackend(Protocol):
    """Provider-neutral boundary for future explicitly supported backends."""

    backend_name: str
    model_id: str
    provider: str | None
    capabilities: BackendCapabilities

    def generate_replacement(self, request: ReplacementRequest) -> ReplacementResponse:
        ...

    def generate_continuation(self, request: ContinuationRequest) -> ContinuationResponse:
        ...


def _stable_int(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _seed(base: int, *identity: object) -> int:
    """Derive a seed from the complete experimental-unit identity."""

    return _stable_int(base, *identity) % (2**32)


class DeterministicMockBackend:
    """Offline test double. It makes no network calls and no research claims."""

    capabilities = BackendCapabilities(
        replacement_generation=True,
        visible_text_prefix_replay=True,
        arbitrary_hidden_cot_internal_state_continuation=False,
    )

    def __init__(self, model_id: str = "mock/sentence-resampler-v1", provider: str | None = "local"):
        self.backend_name = "deterministic_mock"
        self.model_id = model_id
        self.provider = provider
        self.replacement_calls = 0
        self.continuation_calls = 0

    def generate_replacement(self, request: ReplacementRequest) -> ReplacementResponse:
        self.replacement_calls += 1
        rng = random.Random(request.seed)
        token = rng.randrange(100_000, 1_000_000)
        text = f"Mock alternative for {request.target_sentence_id} [seed-token {token}]."
        return ReplacementResponse(
            replacement_text=text,
            raw_response={"mock": True, "seed": request.seed, "seed_token": token},
        )

    def generate_continuation(self, request: ContinuationRequest) -> ContinuationResponse:
        self.continuation_calls += 1
        rng = random.Random(request.seed)
        estimate = float(rng.randrange(10_000_000, 90_000_001))
        continuation = f"Mock continuation generated from visible prefix with seed {request.seed}."
        visible = f"{int(estimate)}\n\nDeterministic mock answer."
        return ContinuationResponse(
            continuation_text=continuation,
            visible_final_answer=visible,
            parsed_final_estimate=estimate,
            raw_response={"mock": True, "seed": request.seed},
        )


def _backend_for(config: ExperimentConfig) -> SentenceResamplingBackend:
    if config.backend.backend != "deterministic_mock":
        raise BackendNotImplemented(
            f"backend {config.backend.backend!r} is not implemented; this scaffold only "
            "provides deterministic_mock and will not call a real API"
        )
    return DeterministicMockBackend(config.backend.model_id, config.backend.provider)


def _validate_backend(config: ExperimentConfig, backend: SentenceResamplingBackend) -> None:
    if backend.backend_name != config.backend.backend:
        raise ValueError("configured backend does not match supplied backend instance")
    if backend.model_id != config.backend.model_id or backend.provider != config.backend.provider:
        raise ValueError("configured model/provider does not match supplied backend instance")
    if not backend.capabilities.replacement_generation:
        raise UnsupportedContinuationCapability("backend cannot generate replacement sentences")
    if not backend.capabilities.supports(config.continuation_mode):
        if config.continuation_mode is ContinuationMode.HIDDEN_COT_INTERNAL_STATE:
            raise UnsupportedContinuationCapability(
                "requested hidden_cot_internal_state_continuation, but the backend only "
                "supports replaying visible text. Arbitrary hidden-CoT/internal-state "
                "continuation is unavailable; the experiment definition was not changed."
            )
        raise UnsupportedContinuationCapability(
            f"backend does not support {config.continuation_mode.value}"
        )


def _sentence_dict(sentence: Sentence) -> dict[str, Any]:
    value: dict[str, Any] = {"sentence_id": sentence.sentence_id, "text": sentence.text}
    if sentence.start_char is not None:
        value["start_char"] = sentence.start_char
        value["end_char"] = sentence.end_char
    return value


def _replacement_request_dict(request: ReplacementRequest) -> dict[str, Any]:
    value = asdict(request)
    value["preserved_prefix"] = [_sentence_dict(item) for item in request.preserved_prefix]
    value["original_target_sentence"] = _sentence_dict(request.original_target_sentence)
    return value


def _continuation_request_dict(request: ContinuationRequest) -> dict[str, Any]:
    value = asdict(request)
    value["continuation_mode"] = request.continuation_mode.value
    value["preserved_prefix"] = [_sentence_dict(item) for item in request.preserved_prefix]
    value["replacement_sentence"] = _sentence_dict(request.replacement_sentence)
    return value


def _config_dict(config: ExperimentConfig) -> dict[str, Any]:
    value = asdict(config)
    value["continuation_mode"] = config.continuation_mode.value
    return value


def _source_dict(source: SourceTrajectory) -> dict[str, Any]:
    value = asdict(source)
    value["sentences"] = [_sentence_dict(item) for item in source.sentences]
    value["task_messages"] = [dict(message) for message in source.task_messages]
    return value


def _prefix_payload(
    source: SourceTrajectory,
    target_offset: int,
    assigned_text: str,
) -> tuple[str, str]:
    """Build a target-ending prefix and say whether it is byte-verbatim or reconstructed."""

    target = source.sentences[target_offset]
    if source.visible_reasoning_text is not None:
        if target.start_char is None:
            raise ValueError("target character span is required for verbatim prefix construction")
        return (
            source.visible_reasoning_text[:target.start_char] + assigned_text,
            "verbatim_source_prefix_plus_assigned_sentence",
        )
    prefix = source.sentences[:target_offset]
    return (
        "\n".join([item.text for item in prefix] + [assigned_text]),
        "manual_sentence_join_with_newline_nonverbatim",
    )


def _parsed_estimates(items: list[dict[str, Any]]) -> list[float]:
    return [
        item["response"]["parsed_final_estimate"]
        for item in items
        if item["response"]["parsed_final_estimate"] is not None
    ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _signed_normalized(values: list[float], source: SourceTrajectory) -> list[float]:
    sign = {"above_good": 1.0, "below_good": -1.0}.get(source.condition)
    if sign is None:
        return []
    return [sign * ((value - source.threshold) / source.threshold) for value in values]


def _comparison_summary(
    original_group: dict[str, Any],
    groups: list[dict[str, Any]],
    source: SourceTrajectory,
) -> dict[str, Any]:
    original_estimates = _parsed_estimates(original_group["continuations"])
    by_replacement = []
    all_estimates: list[float] = []
    for group in groups:
        estimates = _parsed_estimates(group["continuations"])
        all_estimates.extend(estimates)
        by_replacement.append({
            "replacement_index": group["replacement_index"],
            "n_continuations": len(group["continuations"]),
            "n_parsed_final_estimates": len(estimates),
            "parsed_final_estimates": estimates,
            "mean_parsed_final_estimate": _mean(estimates),
            "mean_favored_direction_signed_normalized": _mean(
                _signed_normalized(estimates, source)
            ),
        })
    original_mean = _mean(original_estimates)
    resampled_mean = _mean(all_estimates)
    return {
        "source_visible_final_answer": source.visible_final_answer,
        "source_parsed_final_estimate": source.parsed_final_estimate,
        "source_observed_answer_role": "descriptive_only_not_a_stochastic_control",
        "original_replay_visible_final_estimates": original_estimates,
        "n_original_replay_visible_final_estimates": len(original_estimates),
        "mean_original_replay_visible_final_estimate": original_mean,
        "resampled_visible_final_estimates": all_estimates,
        "n_resampled_visible_final_estimates": len(all_estimates),
        "mean_resampled_visible_final_estimate": resampled_mean,
        "resampled_minus_original_mean_estimate": (
            resampled_mean - original_mean
            if original_mean is not None and resampled_mean is not None
            else None
        ),
        "mean_original_favored_direction_signed_normalized": _mean(
            _signed_normalized(original_estimates, source)
        ),
        "mean_resampled_favored_direction_signed_normalized": _mean(
            _signed_normalized(all_estimates, source)
        ),
        "by_replacement": by_replacement,
        "interpretation": (
            "The original and replacement distributions both come from fresh continuations "
            "conditioned on replayed visible text. Their difference is not an intervention "
            "on the source run's hidden model state."
        ),
    }


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def run_pipeline(
    *,
    config_path: str | Path,
    source_path: str | Path,
    output_dir: str | Path,
    runs_root: str | Path = "runs",
    repo_root: str | Path = ".",
    backend: SentenceResamplingBackend | None = None,
) -> dict[str, Path]:
    """Run the offline scaffold and write a complete, never-overwritten bundle."""

    config_input = Path(config_path).resolve()
    source_input = Path(source_path).resolve()
    config_raw = _load_mapping(config_input, "config")
    source_raw = _load_mapping(source_input, "source")
    source = SourceTrajectory.from_dict(source_raw)
    config = ExperimentConfig.from_dict(config_raw)
    source.validate()
    config.validate(source)
    selected_backend = backend or _backend_for(config)

    # Capability failure must occur before sampling and before output creation.
    _validate_backend(config, selected_backend)
    output = create_new_directory(output_dir, raw_roots=[runs_root])

    target_offset = config.target_sentence_index - 1
    target = source.sentences[target_offset]
    prefix = source.sentences[:target_offset]
    original_visible_prefix, prefix_construction = _prefix_payload(
        source, target_offset, target.text
    )
    experimental_unit_id = hashlib.sha256(
        "\0".join(
            (
                config.experiment_id,
                source.source_id,
                config.target_type,
                target.sentence_id,
                config.semantic_arm_id,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    requests: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "manual_segmentation_authoritative": True,
        "experimental_unit_id": experimental_unit_id,
        "target_type": config.target_type,
        "semantic_arm_id": config.semantic_arm_id,
        "prefix_construction": prefix_construction,
        "original_replay_continuation_requests": [],
        "replacement_requests": [],
        "continuation_requests": [],
    }
    groups: list[dict[str, Any]] = []

    original_group: dict[str, Any] = {
        "intervention_arm": "original_replay",
        "continuations": [],
    }
    # Phase A: generate the complete candidate bank. No continuation is sampled
    # until every replacement response exists and its canonical hash is frozen.
    for replacement_index in range(config.n_replacements):
        replacement_seed = _seed(
            config.random_seed,
            config.experiment_id,
            source.source_id,
            config.target_type,
            target.sentence_id,
            config.semantic_arm_id,
            "replacement_generation",
            "replacement",
            replacement_index,
        )
        replacement_request = ReplacementRequest(
            request_id=f"{experimental_unit_id}-replacement-{replacement_index:04d}",
            source_id=source.source_id,
            target_sentence_index=config.target_sentence_index,
            target_sentence_id=target.sentence_id,
            target_type=config.target_type,
            semantic_arm_id=config.semantic_arm_id,
            task_prompt=source.task_prompt,
            task_messages=source.task_messages,
            condition=source.condition,
            threshold=source.threshold,
            preserved_prefix=prefix,
            original_target_sentence=target,
            replacement_instruction=config.replacement_instruction,
            seed=replacement_seed,
            model_id=config.backend.model_id,
            backend=config.backend.backend,
            provider=config.backend.provider,
            settings=config.backend.settings,
        )
        requests["replacement_requests"].append(_replacement_request_dict(replacement_request))
        replacement = selected_backend.generate_replacement(replacement_request)
        group = {
            "intervention_arm": "replacement",
            "replacement_index": replacement_index,
            "replacement_request_id": replacement_request.request_id,
            "replacement_response": asdict(replacement),
            "continuations": [],
        }
        groups.append(group)

    candidate_bank_payload = {
        "source_id": source.source_id,
        "target_type": config.target_type,
        "target_sentence_id": target.sentence_id,
        "semantic_arm_id": config.semantic_arm_id,
        "replacement_requests": requests["replacement_requests"],
        "replacement_responses": [
            group["replacement_response"] for group in groups
        ],
        "validation": "deterministic_mock_auto_accept_only",
        "continuation_outcomes_observed": False,
    }
    candidate_bank_sha256 = hashlib.sha256(
        json.dumps(
            candidate_bank_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    requests["candidate_bank"] = {
        **candidate_bank_payload,
        "sha256": candidate_bank_sha256,
        "frozen_before_continuations": True,
    }

    # Phase B: build every continuation request and a deterministic randomized
    # execution schedule before sampling any continuation.
    continuation_jobs: list[
        tuple[ContinuationRequest, dict[str, Any], int]
    ] = []
    for continuation_index in range(config.original_replay_continuations):
        continuation_seed = _seed(
            config.random_seed,
            config.experiment_id,
            source.source_id,
            config.target_type,
            target.sentence_id,
            config.semantic_arm_id,
            "continuation",
            "original_replay",
            0,
            continuation_index,
        )
        continuation_request = ContinuationRequest(
            request_id=f"{experimental_unit_id}-original-{continuation_index:04d}",
            replacement_request_id=None,
            source_id=source.source_id,
            target_type=config.target_type,
            semantic_arm_id=config.semantic_arm_id,
            intervention_arm="original_replay",
            continuation_mode=config.continuation_mode,
            task_prompt=source.task_prompt,
            task_messages=source.task_messages,
            condition=source.condition,
            threshold=source.threshold,
            preserved_prefix=prefix,
            replacement_sentence=target,
            visible_prefix_replay=original_visible_prefix,
            visible_prefix_sha256=hashlib.sha256(
                original_visible_prefix.encode("utf-8")
            ).hexdigest(),
            prefix_construction=prefix_construction,
            seed=continuation_seed,
            model_id=config.backend.model_id,
            backend=config.backend.backend,
            provider=config.backend.provider,
            settings=config.backend.settings,
        )
        request_dict = _continuation_request_dict(continuation_request)
        requests["original_replay_continuation_requests"].append(request_dict)
        continuation_jobs.append((continuation_request, original_group, continuation_index))

    for replacement_index, group in enumerate(groups):
        replacement_request_id = group["replacement_request_id"]
        replacement_text = group["replacement_response"]["replacement_text"]
        replacement_sentence = Sentence(target.sentence_id, replacement_text)
        visible_prefix, replacement_prefix_construction = _prefix_payload(
            source, target_offset, replacement_text
        )
        for continuation_index in range(config.continuations_per_replacement):
            continuation_seed = _seed(
                config.random_seed,
                config.experiment_id,
                source.source_id,
                config.target_type,
                target.sentence_id,
                config.semantic_arm_id,
                "continuation",
                "replacement",
                replacement_index,
                continuation_index,
            )
            continuation_request = ContinuationRequest(
                request_id=(
                    f"{experimental_unit_id}-replacement-{replacement_index:04d}-"
                    f"continuation-{continuation_index:04d}"
                ),
                replacement_request_id=replacement_request_id,
                source_id=source.source_id,
                target_type=config.target_type,
                semantic_arm_id=config.semantic_arm_id,
                intervention_arm="replacement",
                continuation_mode=config.continuation_mode,
                task_prompt=source.task_prompt,
                task_messages=source.task_messages,
                condition=source.condition,
                threshold=source.threshold,
                preserved_prefix=prefix,
                replacement_sentence=replacement_sentence,
                visible_prefix_replay=visible_prefix,
                visible_prefix_sha256=hashlib.sha256(
                    visible_prefix.encode("utf-8")
                ).hexdigest(),
                prefix_construction=replacement_prefix_construction,
                seed=continuation_seed,
                model_id=config.backend.model_id,
                backend=config.backend.backend,
                provider=config.backend.provider,
                settings=config.backend.settings,
            )
            request_dict = _continuation_request_dict(continuation_request)
            requests["continuation_requests"].append(request_dict)
            continuation_jobs.append((continuation_request, group, continuation_index))

    schedule_seed = _seed(
        config.random_seed,
        config.experiment_id,
        source.source_id,
        config.target_type,
        target.sentence_id,
        config.semantic_arm_id,
        "continuation_schedule",
    )
    random.Random(schedule_seed).shuffle(continuation_jobs)
    requests["continuation_schedule"] = {
        "seed": schedule_seed,
        "randomized_before_continuations": True,
        "candidate_bank_sha256": candidate_bank_sha256,
        "semantic_arm_id": config.semantic_arm_id,
        "request_order": [job[0].request_id for job in continuation_jobs],
    }

    for schedule_index, (continuation_request, destination, continuation_index) in enumerate(
        continuation_jobs
    ):
        request_dict = _continuation_request_dict(continuation_request)
        response = selected_backend.generate_continuation(continuation_request)
        destination["continuations"].append({
            "schedule_index": schedule_index,
            "continuation_index": continuation_index,
            "request_id": continuation_request.request_id,
            "request": request_dict,
            "response": asdict(response),
        })

    original_group["continuations"].sort(key=lambda item: item["continuation_index"])
    for group in groups:
        group["continuations"].sort(key=lambda item: item["continuation_index"])

    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "target_type": config.target_type,
        "semantic_arm_id": config.semantic_arm_id,
        "continuation_mode": config.continuation_mode.value,
        "capabilities": asdict(selected_backend.capabilities),
        "visible_final_answer_policy": (
            "visible_final_answer and parsed_final_estimate are separate fields; "
            "no trajectory endpoint substitution"
        ),
        "protocol_integrity": {
            "scope": "single_source_single_semantic_arm_mock_protocol_shape",
            "flagship_multi_arm_protocol_complete": False,
            "candidate_bank_sha256": candidate_bank_sha256,
            "candidate_bank_frozen_before_continuations": True,
            "continuation_schedule_seed": schedule_seed,
            "continuation_schedule_randomized_before_sampling": True,
        },
        "original_replay": original_group,
        "replacement_groups": groups,
        "comparison": _comparison_summary(original_group, groups, source),
    }

    provenance = build_provenance(
        experiment_id=config.experiment_id,
        model_id=config.backend.model_id,
        backend=config.backend.backend,
        provider=config.backend.provider,
        prompt_version=config.prompt_version,
        prompts={
            "task_prompt": source.task_prompt,
            "replacement_instruction": config.replacement_instruction,
            "original_visible_prefix": original_visible_prefix,
        },
        n_requested=(
            config.n_replacements
            + config.original_replay_continuations
            + config.n_replacements * config.continuations_per_replacement
        ),
        sampling_settings={
            "continuation_mode": config.continuation_mode.value,
            "n_replacements": config.n_replacements,
            "target_type": config.target_type,
            "semantic_arm_id": config.semantic_arm_id,
            "continuations_per_replacement": config.continuations_per_replacement,
            "original_replay_continuations": config.original_replay_continuations,
            "require_verbatim_prefix": config.require_verbatim_prefix,
            "require_exact_task_messages": config.require_exact_task_messages,
            "prefix_construction": prefix_construction,
            "candidate_bank_sha256": candidate_bank_sha256,
            "continuation_schedule_seed": schedule_seed,
            "backend_settings": dict(config.backend.settings),
            "all_request_seeds_recorded_in": "requests.json",
        },
        random_seed=config.random_seed,
        raw_response_directory=output,
        judge_model=None,
        figure_script=None,
        repo_root=repo_root,
        limitations=(
            "Deterministic mock only; no real provider is implemented.",
            "Visible-text prefix replay is not hidden-CoT/internal-state continuation.",
            "Manual sentence segmentation is authoritative and may affect conclusions.",
        ),
    ).to_dict()
    provenance["input_artifacts"] = {
        "config": {"path": str(config_input), "sha256": sha256_file(config_input)},
        "source": {"path": str(source_input), "sha256": sha256_file(source_input)},
    }
    provenance["backend_capabilities"] = asdict(selected_backend.capabilities)

    outputs = {
        "config": write_new_json(output / "config.json", _config_dict(config)),
        "source": write_new_json(output / "source.json", _source_dict(source)),
        "requests": write_new_json(output / "requests.json", requests),
        "results": write_new_json(output / "results.json", results),
        "provenance": write_new_json(output / "provenance.json", provenance),
    }
    manifest_value = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "files_sha256": file_manifest(list(outputs.values()), output),
        "manifest_self_hash_included": False,
    }
    outputs["manifest"] = write_new_json(output / "manifest.json", manifest_value)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run_pipeline(
            config_path=args.config,
            source_path=args.source,
            output_dir=args.output_dir,
            runs_root=args.runs_root,
            repo_root=args.repo_root,
        )
    except (BackendNotImplemented, UnsupportedContinuationCapability, FileExistsError,
            FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
