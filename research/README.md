# Research layer

This directory is a read-only analysis layer over the starter implementation.
It is designed for a short loop: inspect raw evidence, record an observation,
write competing hypotheses, run a small discriminating intervention, inspect
raw responses, then produce transparent statistics and figures.

## Data boundaries

- `runs/` is immutable shipped/raw data. Research code reads it but never writes
  into it.
- `experiments/` stores prompt/config definitions and experiment-specific code.
- `figures/<bundle>/` stores a generated figure together with the exact
  analysis settings, JSON results, provenance, and source hashes.
- `derived/` is ignored regenerable scratch output. Do not put irreplaceable raw
  API responses there.
- Exports refuse overwrite by default. Use a new output name for each rerun.

The starter plotting command writes `fig.png` and `factor.json` into a run
directory. To preserve shipped data, use it only on a copied run or use the
research trajectory command below.

## Inspect real rollouts

```text
uv run python -m research.inspect_runs
uv run python -m research.inspect_runs --model qwen3.5-122b-a10b --format json
uv run python -m research.inspect_rollouts --model qwen3.5-122b-a10b --condition baseline --index 0
uv run python -m research.inspect_rollouts --model qwen3.5-122b-a10b --compare --index 0 --format markdown --export notes/annotations/qwen_000.md
uv run python -m research.inspect_rollouts --model qwen3.5-122b-a10b --compare --random 3 --seed 20260825 --annotation notes/annotations/qwen_seed20260825.csv
```

`reasoning`, visible `content`, estimate-judge output, and trajectory-judge
output remain separate fields. Shipped `estimates.json` files contain only the
baseline condition because the starter pipeline runs that judge before sampling
the donation conditions. The inspector reports those conditioned estimates as
unavailable; it never substitutes the last trajectory point.

## Trajectory/statistical analysis

```text
uv run python -m research.trajectory_analysis --config experiments/E02_trajectory_localization/configs/shipped_qwen.json --output-dir figures/E02_shipped_qwen_rerun
```

The analysis reports:

- `(estimate - threshold) / threshold`;
- pooled, start-above, start-below, and start-equal strata;
- early/late 20% means and medians for each rollout;
- paired within-trajectory drift;
- medians, quantiles, effect sizes, seeded percentile bootstrap CIs, and N;
- invalid/outlier counts and exact filtering rules;
- visible-final distributions from `estimates.json` only;
- reasoning endpoints under a distinct, explicitly non-final label.

There are two MRF fields on purpose. `starter_compatible_unfiltered` reproduces
the implementation in `src/value_leakage/plot.py`, whose drift calculation does
not apply its documented 10x filter. `robustness_10x_filtered` applies the filter
before aggregating drift. Do not merge the two definitions.

## Side-mechanics negative controls

Read-only first/last side, toward-threshold, baseline, label-shuffle, and
placebo-threshold analysis over shipped `trajectories.json` files. It does not
open sealed packets or holdout text.

```text
uv run python -m research.side_mechanics --runs-root runs --output-dir figures/side_mechanics_v3 --discovery-annotation notes/annotations/qwen122b_discovery_v2.csv --discovery-reveal derived/E02_trajectory_localization/qwen122b_discovery_reveal_v2/discovery_reveal.csv
```

The generator refuses to overwrite an existing bundle. Cite
`figures/side_mechanics_v3/` ; earlier `v1`/`v2` directories are superseded
axis/copy iterations.

## Tests

```text
uv run python -m unittest discover -s tests -v
```

No test calls a paid API. New behavioral API code must first pass a mock or tiny
explicitly authorized smoke test.
