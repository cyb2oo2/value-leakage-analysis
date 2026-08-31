# Observations

Record evidence before interpreting it. Include run, condition, rollout ID, and
whether the statement comes from raw text, a stored judge artifact, or a derived
metric.

## Initial infrastructure observations (2026-08-25)

- Shipped data contains 10 model runs with 100 raw slots per condition.
- The target Qwen run is `qwen3.5-122b-a10b_20260815_030702`; its threshold is
  41,000,000 and its backend/provider was OpenRouter / `deepinfra/fp4`.
- Raw reasoning is at `rows[i].reasoning`; visible final text is at
  `rows[i].content`; successive judge-extracted reasoning estimates are at
  `trajectories[condition][i]`.
- Every shipped `estimates.json` contains baseline only. Conditioned visible
  final estimates are therefore unavailable without a new, separately stored
  judge pass or another explicitly defined parser.
- On Qwen, valid trajectory artifacts before the length/outlier policy are
  baseline 99, below-good 90, above-good 90. After the 10x trajectory filter and
  minimum-two-point requirement, counts are 93, 86, and 87.
- Starter-compatible unfiltered pooled MRF is 0.0268407633. The explicitly 10x
  filtered robustness MRF is 0.0307869688. These are descriptive checks, not a
  causal localization result.

## New observation template

Date:

Run / model:

Condition / rollout ID:

Evidence source:

Observation:

Why it may matter:

Links to inspection export / figure:
