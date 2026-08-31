"""Provider-neutral, mock-only pilot runner for E01.

This stage intentionally has no Fireworks or OpenRouter implementation.  It
does not read environment variables, construct network clients, or call APIs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Protocol, runtime_checkable

from research.experiment_utils import (
    build_provenance,
    create_new_directory,
    sha256_file,
    sha256_text,
    write_new_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_SPEC = Path(__file__).with_name("prompt_variants.v1.json")
SUPPORTED_MOCK_BACKEND = "mock"
DISABLED_REAL_BACKENDS = frozenset({"fireworks", "openrouter"})
REQUIRED_CONDITIONS = frozenset(
    {"above_good", "below_good", "neutral_equal_good"}
)
CONDITION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class PromptSpec:
    schema_version: str
    prompt_version: str
    task: Mapping[str, Any]
    design: Mapping[str, Any]
    variants: Mapping[str, Mapping[str, Any]]
    source_path: Path


@dataclass(frozen=True)
class PilotConfig:
    schema_version: str
    experiment_id: str
    prompt_spec: str
    prompt_version: str
    conditions: tuple[str, ...]
    threshold: int
    model_id: str
    backend: str
    provider: str | None
    n: int
    temperature: float
    top_p: float
    reasoning: Mapping[str, Any]
    max_tokens: int
    seed: int
    judge: Mapping[str, Any]
    output_directory: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["conditions"] = list(self.conditions)
        return value


@dataclass(frozen=True)
class SamplingRequest:
    condition: str
    rollout_index: int
    prompt: str
    threshold: int
    model_id: str
    temperature: float
    top_p: float
    reasoning: Mapping[str, Any]
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class SamplingResponse:
    condition: str
    rollout_index: int
    reasoning: str
    content: str
    finish_reason: str
    usage: Mapping[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "i": self.rollout_index,
            "condition": self.condition,
            "reasoning": self.reasoning,
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": None if self.usage is None else dict(self.usage),
        }


@runtime_checkable
class SamplingBackend(Protocol):
    """Minimal interface a future provider adapter must implement."""

    backend_name: str
    provider_name: str | None

    def sample(self, requests: Sequence[SamplingRequest]) -> list[SamplingResponse]:
        """Return exactly one response for every request, in request order."""


class MockBackend:
    """Deterministic offline backend for pipeline smoke tests only."""

    backend_name = "mock"
    provider_name = "local-deterministic-mock"

    def sample(self, requests: Sequence[SamplingRequest]) -> list[SamplingResponse]:
        return [self._sample_one(request) for request in requests]

    @staticmethod
    def _sample_one(request: SamplingRequest) -> SamplingResponse:
        identity = "\0".join(
            (
                str(request.seed),
                request.condition,
                str(request.rollout_index),
                request.model_id,
                request.prompt,
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        span = max(1, request.threshold // 20)
        initial_offset = int.from_bytes(digest[:8], "big") % (2 * span + 1) - span
        revision_offset = int.from_bytes(digest[8:16], "big") % (span + 1) - span // 2
        first_estimate = max(1, request.threshold + initial_offset)
        final_estimate = max(1, first_estimate + revision_offset)
        reasoning = (
            "Synthetic mock trace for pipeline validation only. "
            f"Initial estimate: {first_estimate:,}. "
            f"After a deterministic mock revision: {final_estimate:,}."
        )
        content = f"Final estimate: {final_estimate:,}"
        return SamplingResponse(
            condition=request.condition,
            rollout_index=request.rollout_index,
            reasoning=reasoning,
            content=content,
            finish_reason="mock_stop",
            usage={
                "prompt_characters": len(request.prompt),
                "completion_characters": len(reasoning) + len(content),
            },
        )


@dataclass(frozen=True)
class PilotRun:
    output_directory: Path
    effective_config: Path
    provenance: Path
    resolved_prompts: Path
    raw_response_files: tuple[Path, ...]
    manifest: Path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: object, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    integer = int(value)
    minimum = 0 if allow_zero else 1
    if integer < minimum:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{field} must be {comparator}")
    return integer


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    return result


def load_prompt_spec(path: Path | str = DEFAULT_PROMPT_SPEC) -> PromptSpec:
    source = Path(path).resolve()
    raw = _load_json_object(source, "prompt spec")
    schema_version = _nonempty_string(raw.get("schema_version"), "schema_version")
    prompt_version = _nonempty_string(raw.get("prompt_version"), "prompt_version")
    task = raw.get("task")
    design = raw.get("design")
    variants = raw.get("variants")
    if not isinstance(task, dict) or not isinstance(design, dict):
        raise ValueError("prompt spec task and design must be JSON objects")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("prompt spec variants must be a non-empty JSON object")
    missing = REQUIRED_CONDITIONS.difference(variants)
    if missing:
        raise ValueError(f"prompt spec is missing required conditions: {sorted(missing)}")
    question = _nonempty_string(task.get("question"), "task.question")
    placeholder = _nonempty_string(
        design.get("threshold_placeholder"), "design.threshold_placeholder"
    )
    if placeholder != "{threshold}":
        raise ValueError("design.threshold_placeholder must be exactly '{threshold}'")
    for condition, variant in variants.items():
        if not isinstance(condition, str) or CONDITION_NAME.fullmatch(condition) is None:
            raise ValueError(f"unsafe condition name in prompt spec: {condition!r}")
        if not isinstance(variant, dict):
            raise ValueError(f"variant {condition!r} must be a JSON object")
        template = _nonempty_string(variant.get("template"), f"variants.{condition}.template")
        if template.count(placeholder) != 1:
            raise ValueError(f"variant {condition!r} must contain one threshold placeholder")
        if question not in template:
            raise ValueError(f"variant {condition!r} changes or omits the task question")
    neutral = variants["neutral_equal_good"]
    if neutral.get("preferred_side") is not None:
        raise ValueError("neutral_equal_good must not declare a preferred side")
    neutral_design = design.get("neutral_control")
    if not isinstance(neutral_design, dict):
        raise ValueError("design.neutral_control must be a JSON object")
    if neutral_design.get("directional_preference") is not None:
        raise ValueError("neutral control must have no directional preference")
    return PromptSpec(
        schema_version=schema_version,
        prompt_version=prompt_version,
        task=task,
        design=design,
        variants=variants,
        source_path=source,
    )


def resolve_prompts(
    spec: PromptSpec,
    threshold: int,
    conditions: Sequence[str] | None = None,
) -> dict[str, str]:
    threshold_value = _positive_integer(threshold, "threshold")
    requested = tuple(conditions) if conditions is not None else tuple(spec.variants)
    if not requested:
        raise ValueError("at least one condition is required")
    if len(set(requested)) != len(requested):
        raise ValueError("conditions must be unique")
    rendered_threshold = f"{threshold_value:,}"
    prompts: dict[str, str] = {}
    for condition in requested:
        if condition not in spec.variants:
            raise ValueError(f"condition {condition!r} is absent from prompt spec")
        template = spec.variants[condition]["template"]
        try:
            prompt = template.format(threshold=rendered_threshold)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"could not render prompt for {condition!r}: {exc}") from exc
        if "{threshold}" in prompt or rendered_threshold not in prompt:
            raise ValueError(f"threshold was not resolved exactly for {condition!r}")
        prompts[condition] = prompt
    return prompts


def load_pilot_config(path: Path | str) -> PilotConfig:
    source = Path(path).resolve()
    raw = _load_json_object(source, "pilot config")
    required = {
        "schema_version",
        "experiment_id",
        "prompt_spec",
        "prompt_version",
        "conditions",
        "threshold",
        "model_id",
        "backend",
        "provider",
        "n",
        "temperature",
        "top_p",
        "reasoning",
        "max_tokens",
        "seed",
        "judge",
        "output_directory",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"pilot config is missing required fields: {sorted(missing)}")
    conditions_raw = raw["conditions"]
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("conditions must be a non-empty JSON array")
    conditions = tuple(_nonempty_string(value, "condition") for value in conditions_raw)
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique")
    reasoning = raw["reasoning"]
    judge = raw["judge"]
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be a JSON object")
    if not isinstance(judge, dict) or "enabled" not in judge or "model_id" not in judge:
        raise ValueError("judge must record enabled and model_id")
    if not isinstance(judge["enabled"], bool):
        raise ValueError("judge.enabled must be a bool")
    if judge["model_id"] is not None and not isinstance(judge["model_id"], str):
        raise ValueError("judge.model_id must be a string or null")
    temperature = _finite_number(raw["temperature"], "temperature")
    top_p = _finite_number(raw["top_p"], "top_p")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    provider = raw["provider"]
    if provider is not None and (not isinstance(provider, str) or not provider.strip()):
        raise ValueError("provider must be a non-empty string or null")
    return PilotConfig(
        schema_version=_nonempty_string(raw["schema_version"], "schema_version"),
        experiment_id=_nonempty_string(raw["experiment_id"], "experiment_id"),
        prompt_spec=_nonempty_string(raw["prompt_spec"], "prompt_spec"),
        prompt_version=_nonempty_string(raw["prompt_version"], "prompt_version"),
        conditions=conditions,
        threshold=_positive_integer(raw["threshold"], "threshold"),
        model_id=_nonempty_string(raw["model_id"], "model_id"),
        backend=_nonempty_string(raw["backend"], "backend"),
        provider=provider,
        n=_positive_integer(raw["n"], "n"),
        temperature=temperature,
        top_p=top_p,
        reasoning=dict(reasoning),
        max_tokens=_positive_integer(raw["max_tokens"], "max_tokens"),
        seed=_positive_integer(raw["seed"], "seed", allow_zero=True),
        judge=dict(judge),
        output_directory=_nonempty_string(raw["output_directory"], "output_directory"),
    )


def _configured_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _repo_source_files(candidates: Sequence[Path], repo_root: Path) -> list[Path]:
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == repo_root or repo_root in resolved.parents:
            result.append(resolved)
    return result


def run_pilot(
    config_path: Path | str,
    *,
    output_directory: Path | str | None = None,
    repo_root: Path | str = REPO_ROOT,
    backend: SamplingBackend | None = None,
) -> PilotRun:
    """Run an offline smoke pilot and create a new, immutable artifact bundle."""

    repo = Path(repo_root).resolve()
    config_source = Path(config_path).resolve()
    config = load_pilot_config(config_source)
    if config.backend in DISABLED_REAL_BACKENDS:
        raise NotImplementedError(
            f"backend {config.backend!r} is intentionally disabled in Stage 6; "
            "this scaffold cannot make real API calls"
        )
    if config.backend != SUPPORTED_MOCK_BACKEND:
        raise NotImplementedError(
            f"only backend='mock' is implemented in Stage 6, got {config.backend!r}"
        )
    selected_backend: SamplingBackend = backend if backend is not None else MockBackend()
    if selected_backend.backend_name != config.backend:
        raise ValueError(
            f"backend object reports {selected_backend.backend_name!r}, "
            f"but config requests {config.backend!r}"
        )
    if config.provider != selected_backend.provider_name:
        raise ValueError(
            f"backend provider {selected_backend.provider_name!r} does not match "
            f"config provider {config.provider!r}"
        )

    prompt_spec_path = _configured_path(config.prompt_spec, repo)
    prompt_spec = load_prompt_spec(prompt_spec_path)
    if config.prompt_version != prompt_spec.prompt_version:
        raise ValueError(
            f"config prompt_version {config.prompt_version!r} does not match "
            f"prompt spec {prompt_spec.prompt_version!r}"
        )
    prompts = resolve_prompts(prompt_spec, config.threshold, config.conditions)
    configured_output = (
        Path(output_directory)
        if output_directory is not None
        else Path(config.output_directory)
    )
    output = (
        configured_output.resolve()
        if configured_output.is_absolute()
        else (repo / configured_output).resolve()
    )

    # Reserve the destination before invoking a backend. This ensures an
    # existing/unsafe path is rejected before any future adapter could sample.
    create_new_directory(output, raw_roots=[repo / "runs"])

    requests = [
        SamplingRequest(
            condition=condition,
            rollout_index=index,
            prompt=prompts[condition],
            threshold=config.threshold,
            model_id=config.model_id,
            temperature=config.temperature,
            top_p=config.top_p,
            reasoning=config.reasoning,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        for condition in config.conditions
        for index in range(config.n)
    ]
    responses = selected_backend.sample(requests)
    if len(responses) != len(requests):
        raise ValueError(
            f"backend returned {len(responses)} responses for {len(requests)} requests"
        )
    for request, response in zip(requests, responses, strict=True):
        if (response.condition, response.rollout_index) != (
            request.condition,
            request.rollout_index,
        ):
            raise ValueError("backend responses are missing or out of request order")

    raw_directory = output / "raw"
    raw_directory.mkdir(exist_ok=False)
    effective_config = config.to_dict()
    effective_config.update(
        {
            "input_config": str(config_source),
            "input_config_sha256": sha256_file(config_source),
            "prompt_spec": str(prompt_spec_path),
            "prompt_spec_sha256": sha256_file(prompt_spec_path),
            "output_directory": str(output),
            "raw_response_directory": str(raw_directory),
        }
    )
    config_output = write_new_json(output / "config.json", effective_config)
    resolved_output = write_new_json(
        output / "resolved_prompts.json",
        {
            "schema_version": "value-leakage.resolved-prompts/v1",
            "prompt_version": prompt_spec.prompt_version,
            "prompt_spec": str(prompt_spec_path),
            "prompt_spec_sha256": sha256_file(prompt_spec_path),
            "threshold": config.threshold,
            "prompts": prompts,
            "prompt_sha256": {
                condition: sha256_text(prompt) for condition, prompt in prompts.items()
            },
        },
    )
    sampling_settings = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "reasoning": dict(config.reasoning),
        "max_tokens": config.max_tokens,
    }
    raw_files: list[Path] = []
    for condition in config.conditions:
        rows = [
            response.to_dict()
            for response in responses
            if response.condition == condition
        ]
        raw_files.append(
            write_new_json(
                raw_directory / f"{condition}.json",
                {
                    "schema_version": "value-leakage.raw-sampling/v1",
                    "experiment_id": config.experiment_id,
                    "model_id": config.model_id,
                    "backend": config.backend,
                    "provider": config.provider,
                    "condition": condition,
                    "threshold": config.threshold,
                    "prompt_version": config.prompt_version,
                    "prompt": prompts[condition],
                    "n_requested": config.n,
                    "sampling_settings": sampling_settings,
                    "seed": config.seed,
                    "rows": rows,
                },
            )
        )
    provenance_value = build_provenance(
        experiment_id=config.experiment_id,
        model_id=config.model_id,
        backend=config.backend,
        provider=config.provider,
        prompt_version=config.prompt_version,
        prompts=prompts,
        n_requested=config.n,
        sampling_settings=sampling_settings,
        random_seed=config.seed,
        raw_response_directory=raw_directory,
        judge_model=config.judge.get("model_id"),
        figure_script=None,
        source_files=_repo_source_files(
            [prompt_spec_path, Path(__file__), config_source], repo
        ),
        repo_root=repo,
        limitations=(
            "Mock responses are synthetic and cannot support a behavioral conclusion.",
            "Fireworks and OpenRouter execution are intentionally not implemented.",
            "No trajectory judge is run in this scaffold.",
        ),
    ).to_dict()
    provenance_value.update(
        {
            "input_config": str(config_source),
            "input_config_sha256": sha256_file(config_source),
            "prompt_spec": str(prompt_spec_path),
            "prompt_spec_sha256": sha256_file(prompt_spec_path),
            "n_per_condition": config.n,
            "conditions": list(config.conditions),
            "judge": dict(config.judge),
        }
    )
    provenance_output = write_new_json(output / "provenance.json", provenance_value)
    artifacts = [config_output, resolved_output, provenance_output, *raw_files]
    manifest_output = write_new_json(
        output / "manifest.json",
        {
            "schema_version": "value-leakage.artifact-manifest/v1",
            "files_sha256": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in artifacts
            },
        },
    )
    return PilotRun(
        output_directory=output,
        effective_config=config_output,
        provenance=provenance_output,
        resolved_prompts=resolved_output,
        raw_response_files=tuple(raw_files),
        manifest=manifest_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional new output directory; the effective path is recorded.",
    )
    args = parser.parse_args(argv)
    run = run_pilot(args.config, output_directory=args.output_dir)
    print(f"Mock pilot artifacts written to {run.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
