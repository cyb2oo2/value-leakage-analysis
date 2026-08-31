# E01 — neutral-threshold control

This scaffold tests whether a salient threshold shifts estimates even when its
two consequences have explicitly equal value. It preserves the giraffe-spots
question, threshold number, bet structure, donation consequence, and response
instruction used by the original `above_good` and `below_good` conditions.

The primary control, `neutral_equal_good`, sends the same USD 100 donation to
preselected Cause A above the threshold or preselected Cause B at/below it. The
prompt states that both causes were selected in advance and are regarded as
equally beneficial. Cause labels are abstract to avoid adding a cause-specific
preference. This is an intended-valence control, not proof that the model
internally values the branches equally.

All exact templates and design metadata live in
`prompt_variants.v1.json`. Sampling and provenance settings live in a run
config such as `configs/mock_pilot.json`; prompt prose is not embedded in the
runner.

## Offline smoke test

Choose a fresh output path and run:

```text
uv run python -m experiments.E01_neutral_threshold.pilot --config experiments/E01_neutral_threshold/configs/mock_pilot.json --output-dir derived/E01_neutral_threshold/my_new_mock_smoke
```

The runner creates a new directory containing effective `config.json`, exact
`resolved_prompts.json`, `provenance.json`, a hash `manifest.json`, and one raw
response file per condition under `raw/`. Existing output directories and every
path under repository `runs/` are rejected.

Only the deterministic `mock` backend exists in Stage 6. Configuring
`fireworks` or `openrouter` raises `NotImplementedError` before output creation.
The module never reads API keys or imports provider SDKs. Adding a real adapter
requires a separate decision about exact model IDs and continuation/reasoning
semantics; no real sampling is authorized by this scaffold.

Mock responses validate plumbing only. They are synthetic and must never be
used as behavioral evidence or passed off as Qwen outputs. No judge or figure is
run at this stage.
