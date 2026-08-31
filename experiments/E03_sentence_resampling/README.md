# E03 Visible-Prefix Sentence Intervention

This directory defines the future sentence-resampling experiment without
claiming that currently available APIs expose hidden chain-of-thought state.
It is offline by default and implements only `deterministic_mock`. The flagship
mechanism design is specified in:

- `FORMAL_HYPOTHESES_v0.1.md` — four falsifiable dominant-pathway models;
- `PREREGISTRATION_DRAFT_v0.1.md` — source selection, arms, estimands,
  hierarchy, gates, and claim boundaries;
- `configs/qwen122b_design.v0.1.json` — machine-readable design with all paid
  phases disabled;
- `design.py` — an offline fail-closed audit and request-budget calculator.

The design draft is not yet a registered confirmatory protocol. It cannot be
locked until the human E02 discovery step and its target-selection freeze are
complete.

## Experiment object

The source trajectory is manually split into `S1 ... ST`. That segmentation is
authoritative: the pipeline does not split, merge, relabel, or classify
sentences. `target_sentence_index` is one-based. For target `Si`, the request
records the exact task prompt/messages, preserved visible prefix `S1 ...
S(i-1)`, the original `Si`,
the replacement instruction, replacement text, every seed, and every generated
continuation.

Every target now receives fresh `P + original Si` continuations as an
`original_replay` distribution. The single final answer observed in the source
rollout is descriptive only and is never used as the stochastic control. A
replacement contrast without this replay arm is not considered a causal
sentence-resampling contrast.

The mock runner executes one source × target × semantic arm in two phases: it
first generates and hashes that arm's full candidate bank, then freezes every
continuation request and a deterministic randomized schedule before any
continuation call. Seeds and globally scoped request IDs bind experiment,
source, target type/span, arm, candidate, and continuation identities. This
tests a single-arm protocol shape; it is **not** the flagship multi-arm
orchestrator. Mock auto-acceptance is not semantic candidate validation. A
future network runner must freeze all source × target × arm banks together,
create one interleaved schedule, and checkpoint each response atomically so an
interrupted paid run is auditable and resumable.

For real work, set `require_verbatim_prefix: true` and provide exact
`visible_reasoning_text` plus validated character spans for every sentence. The
request records the UTF-8 SHA-256 of the final visible prefix. Reconstructing a
prefix by joining manual sentences with newlines is permitted only for mock
scaffolding and is explicitly labeled non-verbatim.

The current executable mode is `visible_text_prefix_replay`. It supplies the
preserved visible sentences plus the replacement sentence as text. This is not
equivalent to resuming the original model's hidden chain of thought, KV cache,
activations, RNG state, or other internal state. Therefore differences in final
estimate distributions support a causal claim about conditioning on replayed
visible text, not a clean intervention on an original hidden reasoning state.
Manual sentence boundaries, prefix formatting, replacement plausibility, and
distribution shift remain confounds.

`hidden_cot_internal_state_continuation` exists as a distinct requested
capability. The mock backend declares it unsupported. Such a config fails before
sampling and before creating an output directory; the pipeline never silently
falls back to visible-prefix replay.

## Offline smoke run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m experiments.E03_sentence_resampling.pipeline `
  --config experiments/E03_sentence_resampling/configs/mock_example.json `
  --source experiments/E03_sentence_resampling/sources/example_source.json `
  --output-dir derived/E03/mock_example `
  --runs-root runs
```

The output directory must be new and outside `runs/`. It contains exact copies
of normalized `config.json` and `source.json`, plus `requests.json`,
`results.json`, `provenance.json`, and a SHA-256 `manifest.json`. Visible final
answer text and parsed final estimate are always separate fields. A missing
parsed final estimate stays `null`; no reasoning endpoint is substituted.

Schemas live under `schemas/`. Real Fireworks, OpenRouter, Anthropic, and local
model providers are intentionally not implemented in this stage.

## Offline design audit

```powershell
.\.venv\Scripts\python.exe -m experiments.E03_sentence_resampling.design `
  --config experiments/E03_sentence_resampling/configs/qwen122b_design.v0.1.json
```

The audit refuses enabled sampling, provider fallback, hidden-state overclaims,
missing original replay arms, post-treatment recovery filtering, and a design
that advances based on favorable effect direction. It calculates request counts
from the nested arms and repetitions. It does not import an API client or read
environment variables.
