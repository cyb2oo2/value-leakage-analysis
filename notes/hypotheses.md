# Hypothesis record

Copy this block for each question. Keep competing explanations alive until an
intervention distinguishes them.

```text
Observation:

Hypothesis H1:

Hypothesis H2:

Alternative explanation:

Prediction under H1:

Prediction under H2:

Intervention:

What is held constant:

Primary metric:

Result that would favor H1:

Result that would favor H2:

What result would change my mind:

Remaining confounds:
```

## H-side-mechanics (Qwen 122B, post-discovery, shipped trajectories)

Observation:

Shipped Qwen 3.5 122B donation trajectories move toward 41M, and last-side P(estimate>T) diverges to 0.174 vs 0.724, while starter MRF is only 0.027.

Hypothesis H1: constant good-direction push. The donation mapping adds a directional drift (up if above_good, down if below_good) throughout the trace.

Hypothesis H2: threshold-seeking plus favored-side absorption. Trajectories move toward 41M from both sides, then preferentially stop on the prompted good side of T.

Alternative explanation: generic Fermi magnitude regression, trajectory-judge artifacts, or first-estimate anchoring without a later stopping rule.

Prediction under H1: P(up|above_good) > P(up|below_good); pooled P(condition-favored revision) > 0.5; baseline toward-T should not match donation toward-T.

Prediction under H2: P(toward|directional) high in both donation arms; P(up) exchangeable across arms; last-on-favored-side higher than first-on-favored-side; placebo T=41M peaks for donation, not baseline.

Intervention:

Read-only negative controls on shipped judge sequences: no-bet baseline, 2000-permutation label shuffle, and placebo thresholds at 0.25T–4T. No new sampling. Holdout unread.

What is held constant:

Same giraffe question, same shipped traces, same 10x filter, same strict `estimate > T` side definition.

Primary metric:

P(toward|directional), permutation p for delta_early and P(up) contrast, last-on-favored-side, placebo gap-shrink at multiplier 1 vs others.

Result that would favor H1:

P(up) contrast extreme under shuffle; pooled favored revision clearly above 0.5.

Result that would favor H2:

Toward-T high in both donation arms, P(up) null, last-side diverges away from 0.5 onto the good side, placebo peak at true T only in donation arms.

What result would change my mind:

A real E01 equal-value threshold control that still produces the last-side absorption pattern, or holdout/E03 showing that impartiality sentences causally prevent the absorption.

Remaining confounds:

Judge extracts any numeric candidate, not the human first *target total*; baseline removes both value and threshold salience; placebo does not insert 41M into a no-value prompt; n=36 discovery slices are not confirmatory.
