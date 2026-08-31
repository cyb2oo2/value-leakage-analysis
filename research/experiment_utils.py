"""Small reproducibility and filesystem-safety helpers for research scripts."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit(repo_root: Path | str = ".") -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def git_is_dirty(repo_root: Path | str = ".") -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_manifest(paths: list[Path], root: Path) -> dict[str, str]:
    resolved_root = root.resolve()
    return {
        path.resolve().relative_to(resolved_root).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and legacy NumPy state; return an explicit modern generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_output_outside_raw(output: Path | str, raw_roots: list[Path | str]) -> Path:
    resolved = Path(output).resolve()
    for root in raw_roots:
        raw = Path(root).resolve()
        if resolved == raw or raw in resolved.parents:
            raise ValueError(f"output {resolved} is inside immutable raw root {raw}")
    return resolved


def create_new_directory(path: Path | str, raw_roots: list[Path | str] | None = None) -> Path:
    resolved = ensure_output_outside_raw(path, raw_roots or [])
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def write_new_text(path: Path | str, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return target


def write_new_json(path: Path | str, value: Any) -> Path:
    return write_new_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class ExperimentProvenance:
    """Companion metadata every new behavioral run or derived figure should carry."""

    experiment_id: str
    created_at_utc: str
    code_commit: str | None
    code_dirty: bool | None
    model_id: str
    backend: str
    provider: str | None
    prompt_version: str
    prompt_sha256: Mapping[str, str]
    n_requested: int
    sampling_settings: Mapping[str, Any]
    random_seed: int | None
    raw_response_directory: str
    judge_model: str | None
    figure_script: str | None
    source_files_sha256: Mapping[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_provenance(
    *,
    experiment_id: str,
    model_id: str,
    backend: str,
    provider: str | None,
    prompt_version: str,
    prompts: Mapping[str, str],
    n_requested: int,
    sampling_settings: Mapping[str, Any],
    random_seed: int | None,
    raw_response_directory: Path | str,
    judge_model: str | None,
    figure_script: str | None,
    source_files: list[Path] | None = None,
    repo_root: Path | str = ".",
    limitations: tuple[str, ...] = (),
) -> ExperimentProvenance:
    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    if not model_id.strip() or not backend.strip() or not prompt_version.strip():
        raise ValueError("model_id, backend, and prompt_version must be non-empty")
    if n_requested < 1:
        raise ValueError("n_requested must be positive")
    if not prompts:
        raise ValueError("at least one exact prompt is required")
    source_files = source_files or []
    repo = Path(repo_root).resolve()
    return ExperimentProvenance(
        experiment_id=experiment_id,
        created_at_utc=utc_now(),
        code_commit=git_commit(repo),
        code_dirty=git_is_dirty(repo),
        model_id=model_id,
        backend=backend,
        provider=provider,
        prompt_version=prompt_version,
        prompt_sha256={name: sha256_text(text) for name, text in sorted(prompts.items())},
        n_requested=n_requested,
        sampling_settings=dict(sampling_settings),
        random_seed=random_seed,
        raw_response_directory=str(Path(raw_response_directory)),
        judge_model=judge_model,
        figure_script=figure_script,
        source_files_sha256=file_manifest(source_files, repo) if source_files else {},
        limitations=limitations,
    )

