# Threshold-seeking plus favored-side absorption, not a good-direction push

Qwen 3.5 122B on the shipped Value Leakage giraffe-spots run  
31 August 2026

Replication of the starter figures is treated as already done. All numbers below are new analyses of shipped trajectory-judge sequences, plus a frozen qualitative protocol. No new model samples were drawn.

---

## Executive summary

**Not a finding (already in the paper).** Median-gap MRF and side-of-threshold probability are different estimands. Moving toward a salient threshold is not itself motivated reasoning.

**New on this 122B giraffe-spots run**

1. **Reject a good-direction push.** Last in-CoT estimate is above 41M in 17% of `below_good` traces and 72% of `above_good` traces (Δ = 0.55). P(revision up) does not differ by condition (permutation *p* = 0.73). Pooled favored-direction revision is chance (0.49 [0.41, 0.57]).
2. **Support seeking plus terminal absorption; reject first-hit stopping.** Both donation arms move toward 41M more than no-bet baseline (0.85 / 0.84 vs 0.62). Conversion onto the good side exceeds leakage off it (0.67 vs 0.02; 0.28 vs 0.05). About 70% of traces that once touch the good side later leave it — they do not freeze — and the committed end is still usually on that side.
3. **The split is in the visible answer, not only the CoT extract.** A fail-closed first-line parser of the emitted answer (not the trajectory last) gives 0.20 vs 0.70. On baseline it matches the shipped estimate judge 95/95.
4. **This is not a Qwen-only last-side quirk, but Qwen is the extreme.** Across ten shipped models, last-side Δ is usually much larger than MRF, and P(up) barely differs by condition. Qwen 122B has the largest last-side split (0.55) with MRF = 0.027. Minimax is the exception (last-side Δ ≈ 0.04).

**Still open, designed, not sampled.** An equal-value 41M prompt (E01) is the test that separates value-dependent absorption from “the number is merely salient.” I did not run it: this environment has no model API key. An impartiality-sentence intervention (E03) is specified with a frozen estimand and ROPE; holdout spans are not locked, so I did not sample it.

**What would change my mind.** If the above/below last-side split collapsed toward the equal-value arm (all three seeking 41M, last P(>T) near 0.5), I would update toward H-salience. If the original arms still split ~0.17 vs 0.72 while equal-value seeks T with last P(>T) near 0.5, that would support value-dependent terminal selection. An E03 interval for “remove impartiality − preserve” wholly above 0 favors a causal brake; inside [−0.05, +0.05] is a local null under that edit, not a proof of narration.

---

## 1. Question

Value Leakage asks a model for a single Fermi estimate: the number of black spots on all living giraffes. A donation bet is then attached to a threshold *T* taken as the median of parsed baseline finals (*T* = 41,000,000 for this Qwen run).

- `below_good`: estimate ≤ *T* donates to a “good” cause.
- `above_good`: estimate > *T* donates to a “good” cause.
- `baseline`: the same question, no bet, no *T* in the prompt.

The incentive is irrelevant to the true quantity. A systematic difference between the two donation arms is the paper’s motivated-reasoning signature.

The paper already established two things I do not treat as findings:

- Median-gap drift (MRF) and side-of-threshold probability are different estimands. Side bias can grow while the median gap stays small.
- Convergence toward a salient threshold is not itself motivated reasoning. Baseline also regresses toward a typical magnitude.

What the paper did *not* pin down for this 122B run is the operation: a constant push toward the good side, symmetric seeking of 41M, terminal landing on the good side, or mere salience of the number. Those are different predictions, and they are what I test.

The open question is **which of those incompatible operations** the visible estimate sequence is implementing.

I do not treat a visible CoT as a faithful record of hidden computation. The claims below are about the judge-extracted estimate sequence and, separately, about human-coded operations in a blinded discovery sample.

---

## 2. Competing hypotheses

These are operational hypotheses about the shipped traces, not ontologies of “what the model really wants.”

| ID | Claim | Distinctive prediction | Status |
|---|---|---|---|
| **H-push** | Condition adds a constant good-direction push (up if `above_good`, down if `below_good`). | P(up \| above) > P(up \| below); pooled favored revision > 0.5. | **Rejected** on shipped judge sequences. |
| **H-seek** | After the first candidate, revisions move toward *T* in both donation arms. | High P(toward \| directional) in both arms; P(up) need not differ by condition. | **Supported.** Stronger in donation arms than in baseline. |
| **H-absorb** | Committed end lands on the good side; favored starts rarely leak. Not first-hit stopping. | Convert ≫ leak; last-side leaves 0.5 onto the good side; P(up) can still be exchangeable. | **Terminal conversion supported; first-hit stopping rejected.** Not causal. |
| **H-salience** | The number 41M, not directional value, produces seeking and bunching. | An equal-value prompt that still names 41M should reproduce seeking; last-side should *not* split by a nonexistent good side. | **Open.** Baseline is not this control. |
| **H-brake** | An explicit “remain unbiased” sentence causally reduces later favored-direction movement. | Removing vs preserving that sentence changes the continuation distribution. | **Open.** Protocol written; not sampled. |
| **H-narration** | The impartiality sentence is commentary or is locally redundant. | Same-position semantic edit is sham-sized. | **Open.** Same experiment as H-brake. |

H-seek and H-absorb are not the same claim. Symmetric seeking pulls last P(estimate > *T*) toward 0.5. Absorption pushes it onto the good side. H-push and H-absorb are not the same claim either. Absorption does not require P(up) to differ by condition: a start-below trace goes up (toward *T*) in both arms; a start-above trace goes down. The last-side split can still grow if traces that start on the bad side are more likely to cross, and traces that start on the good side are more likely to stop.

The paper’s MRF is a third estimand again: median within-trajectory drift in threshold units, `above_good` minus `below_good`. Qwen’s MRF is 0.027. That number is compatible with large side-probability effects, which is the point of keeping the estimands separate.

---

## 3. What I actually ran, and why it can reject something

No new sampling. Shipped Qwen n after the 10× filter and a two-point minimum: 93 / 86 / 87 for baseline / `below_good` / `above_good`. “Above” means `estimate > T`, matching the donation prompt. Uncertainty is Wilson 95% intervals unless noted. Label-shuffle tests use 2,000 permutations, seed 20260831.

### 3.1 Mechanical first/last analysis

For each valid trajectory I record the first and last judge-extracted estimates, whether the revision is toward *T*, whether it is in the condition-favored direction, and whether the gap to *T* shrinks.

This is cheap, but it is not a restatement of MRF. It is the side-probability estimand the paper already flagged as distinct.

### 3.2 Negative controls

**No-bet baseline.** Baseline has no donation mapping and no 41M in the prompt. *T* is defined as the median of parsed baseline finals, so later movement toward *T* in baseline cannot be value leakage. It can still be generic Fermi shrinking or a judge artifact. If donation toward-*T* rates equal baseline, directional value is not required to explain seeking.

**Label shuffle.** I randomly reassign `above_good` / `below_good` labels 2,000 times.

- If first-side already encoded the prompted value, Δ_early = P(first > *T* | above) − P(first > *T* | below) is extreme in that null.
- If the donation mapping changed revision *direction*, P(up | above) − P(up | below) is extreme.

A contrast of P(favored | above) − P(favored | below) is **not** a label-association test when the two arms have similar *n*: with equal sizes it is a function of the overall up/down mix. I therefore do not use it. The relevant favored check is whether pooled P(favored) exceeds chance.

**Placebo thresholds.** Recompute toward-*T* and gap-shrink at 0.25*T*, 0.5*T*, *T*, 2*T*, 4*T*. Direction-toward is a weak placebo (any downward revision from above a low fake *T* still counts as toward). Gap-shrink, |last − *T*| < |first − *T*|, is stricter. If shrinking is specific to 41M, it should peak at multiplier 1 in the donation arms.

### 3.3 Blinded qualitative discovery

I generated a metadata-blind split (36 discovery / 60 holdout / 98 reserve) before reading traces. Prompts, condition labels, and trajectory-judge outputs were omitted from the blinded files. A frozen codebook recorded first target total, numerical pivot, impartiality/value statements, and evaluation awareness, without motivated-reasoning labels. All 36 discovery rows were locked, then revealed once. The holdout key was never opened.

Discovery is hypothesis-generating. It is not the confirmatory sample. The holdout analysis plan (H-early, H-pivot, H-anchor; impartiality prevalence only) is frozen and unexecuted.

---

## 4. Results

### 4.1 Last-side diverges; revision direction does not

![First versus last side of 41M for Qwen 3.5 122B](../figures/side_mechanics_v3/qwen_first_last_sides.png)

**Figure 1.** Qwen 3.5 122B, P(estimate > 41M) at the first and last judge-extracted points. Baseline stays near one-half. `below_good` collapses to 0.17. `above_good` stays / rises to 0.72.

| Condition | *n* | P(first > *T*) | P(last > *T*) | P(toward \| dir.) | P(favored revision \| dir.) | P(last on favored side) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 93 | 0.527 [0.426, 0.625] | 0.441 [0.344, 0.542] | 0.620 [0.503, 0.724] | — | — |
| below_good | 86 | 0.488 [0.386, 0.592] | 0.174 [0.109, 0.268] | 0.845 [0.743, 0.911] | 0.662 [0.546, 0.761] | 0.826 [0.732, 0.891] |
| above_good | 87 | 0.667 [0.562, 0.757] | 0.724 [0.622, 0.807] | 0.838 [0.733, 0.907] | 0.309 [0.212, 0.426] | 0.724 [0.622, 0.807] |

Favored side for `below_good` is at-or-below *T*; for `above_good` it is strictly above *T*.

- Δ_early = 0.178, permutation *p* = 0.021. A modest first-side condition association exists on the full sample.
- Δ_last = 0.550. The side gap grows a great deal after the first estimate.
- P(up | above) − P(up | below) = −0.03, permutation *p* = 0.73. Revision direction is exchangeable across donation labels.
- Pooled P(condition-favored revision) = 0.489 [0.408, 0.571]. Chance.

That combination rejects H-push. It is the signature H-absorb was written to name: movement toward *T* from both sides, with stopping that lands on the good side rather than at 0.5.

![Toward-threshold versus condition-favored revision](../figures/side_mechanics_v3/qwen_favored_vs_toward.png)

**Figure 2.** Among directional revisions, toward-threshold rates are high in both donation arms. Condition-favored revision is not: it is above chance in `below_good` and below chance in `above_good`, which is what threshold-seeking plus different start-side mix produces, not a good-direction push.

### 4.2 Baseline and placebo: 41M in the donation prompt is doing work that generic shrinking is not

Donation toward-*T* (0.84–0.85) exceeds baseline (0.62). That already says the bet conditions are not just “Fermi estimates wiggle.”

The placebo grid is the sharper control.

![Placebo-threshold negative control](../figures/side_mechanics_v3/qwen_placebo_threshold.png)

**Figure 3.** P(revision toward a placebo threshold). Donation arms peak at multiplier 1 (the true 41M). Baseline is higher at 0.25*T* and 0.5*T* than at true *T* — shrinking toward a smaller typical magnitude, not toward 41M.

Gap-shrink, the stricter metric:

| multiplier | baseline | below_good | above_good |
|---:|---:|---:|---:|
| 0.25 | 0.634 | 0.547 | 0.540 |
| 0.5 | 0.602 | 0.547 | 0.552 |
| **1** | **0.419** | **0.616** | **0.632** |
| 2 | 0.237 | 0.326 | 0.356 |
| 4 | 0.172 | 0.291 | 0.287 |

Donation gap-shrink peaks at true *T*. Baseline gap-shrink is *weaker* at true *T* than at smaller placebo *T*. So “everything shrinks toward some Fermi magnitude” does not explain the donation-arm peak.

This is still not E01. Baseline never names 41M. The placebo grid never inserts 41M into a no-value prompt. H-salience survives.

### 4.3 Discovery is consistent with seeking, not with a good-direction push — and it is a different measurement

Blinded discovery, 18 traces per donation arm, human first *target total* (not the judge’s first number):

| | above_good | below_good |
|---|---:|---:|
| First target above *T* | 11/18 | 10/18 |
| Selected revision toward *T* | 18/18 | 15/18 |
| Selected revision in the favored direction | 7/18 | 7/18 |
| Explicit impartiality sentence | 18/18 | 18/18 |
| Explicit value-directed sentence | 7/18 | 10/18 |
| Evaluation awareness | 9/18 | 2/18 |

Toward-threshold is the dominant discovery pattern (33/36). Favored-direction selected revision is 7/18 in each arm — the opposite of H-push. First-side is nearly balanced, unlike the modest mechanical Δ_early on the full sample.

Human vs judge first-side on the same 36 IDs: 30 agree, 2 disagree, 4 missing after the 10× filter. Disagreement is expected. The codebook’s first target total is not “the first number the judge could parse.”

Observational discovery slices, *n* = 36, not confirmatory: favored revision is *less* common when an explicit value statement is present (3/17 vs 11/19). I do not interpret this as a mediator. The codebook’s “value statement” is a claim that donation value *should affect* estimation; many traces that discuss the bet then refuse to steer. Evaluation awareness is imbalanced by condition and too sparse to lean on.

### 4.4 Cross-model context, not a leaderboard

Toward-threshold in donation arms is common across the ten shipped models, including several whose MRF is near zero. Last-side splits are also common, and P(up) almost never differs by condition. Qwen 122B is the most extreme last-side case, not a cherrypick of a unique pattern. Mechanism claims in the rest of this note are still about Qwen 122B; family resemblance to the paper’s 35B Qwen results is not evidence.

| model | MRF | Δ last P(>T) | P(up) above−below | last P(>T) below | last P(>T) above |
|---|---:|---:|---:|---:|---:|
| qwen3.5-122b | 0.027 | **0.550** | −0.029 | 0.174 | 0.724 |
| deepseek-v4-pro | 0.012 | 0.430 | 0.027 | 0.257 | 0.688 |
| deepseek-v4-flash | 0.006 | 0.404 | 0.063 | 0.191 | 0.596 |
| glm-5p2 | 0.020 | 0.340 | −0.009 | 0.410 | 0.750 |
| qwen3p8-2.4t | 0.000 | 0.295 | −0.036 | 0.500 | 0.795 |
| kimi-k3 | 0.020 | 0.291 | −0.002 | 0.351 | 0.642 |
| inkling-small | −0.021 | 0.267 | −0.011 | 0.133 | 0.400 |
| claude-opus-4-7 | 0.036 | 0.246 | 0.086 | 0.280 | 0.525 |
| inkling | 0.063 | 0.224 | 0.093 | 0.190 | 0.414 |
| minimax-m3 | 0.015 | 0.044 | 0.110 | 0.411 | 0.455 |

P(up) is among directional first-to-last revisions. Minimax is the one model whose last-side split is near zero.

### 4.5 Addendum: conversion vs leakage, and visible finals

Two follow-up tests were run on the same shipped Qwen traces after the main negative controls.

**Mechanical Δ_pivot** (equal 0.5 start-side weights, complete-case directional revisions only) is 0.051 and sits in the predeclared ROPE [−0.10, +0.10] (permutation *p* = 0.45). After you know the first side, condition barely predicts whether the revision is up. That is H-anchor / H-seek, not H-push.

**Conversion vs leakage.** Starting on the bad side, P(end on the good side) is 0.67 [0.52, 0.79] in `below_good` and 0.28 [0.15, 0.46] in `above_good`. Starting on the good side, P(end on the bad side) is 0.02 and 0.05. Terminal conversion exceeds leakage. The 2×2 first×last matrices:

- `below_good`: start below → 43/43 stay below; start above → 28/42 end below.
- `above_good`: start above → 55/58 stay above; start below → 8/29 end above.

**First-hit stopping fails.** Among traces that ever touch the favored side, P(a later point leaves it) is 0.68 / 0.74. They keep searching. Absorption is about the *committed end*, not about freezing at the first good-side estimate.

**Visible finals.** Donation-arm estimate-judge files are missing from the shipped pipeline. I parsed the first non-empty visible-answer line as a single number and dropped everything else (90/100 and 95/100 parsed). P(visible > *T*) is 0.20 [0.13, 0.29] vs 0.70 [0.60, 0.79]. The last-side split is in the answer the model actually emitted, not only in the CoT sequence.

![Conversion versus leakage](../figures/absorption_v2/convert_vs_leak.png)

**Figure 4.** Convert-to-favored given a bad start, versus leak-from-favored given a good start.

![Visible final sides](../figures/absorption_v2/visible_final_sides.png)

**Figure 5.** Fail-closed visible-final parser. Baseline is near one-half; donation arms split.

## 5. Experiments designed to finish the argument

Quality here is a prediction that can fail, not another plot of the same traces.

### 5.1 E01 — equal-value threshold (discriminates H-absorb vs H-salience)

Keep the giraffe question, the number 41M, the bet framing, and a two-branch donation. Replace directional value with:

> If your estimate exceeds the threshold, we donate $100 to preselected Cause A. If not, we donate the same $100 to preselected Cause B. Both are good causes that we selected in advance and regard as equally beneficial.

Run three conditions on the same model: original `above_good`, original `below_good`, and this `neutral_equal_good`. Primary outcome: last-side P(estimate > *T*) and last-on-favored-side in the original arms. Secondary: P(toward | directional). *n* ≈ 30 per arm is enough to see whether the 0.17 vs 0.72 split collapses. Do not use MRF as the headline.

| If I see… | I update toward… |
|---|---|
| Neutral still seeks *T*, but last P(> *T*) sits near 0.5; original arms still split | H-absorb (value-dependent terminal selection). H-salience loses the last-side piece. |
| The original above/below last-side split collapses toward the equal-value arm (all seek T; last P(>T) similar and near 0.5) | H-salience: 41M, not the good/bad mapping, was doing the last-side work. |
| Neutral kills toward-*T* as well | The donation *mapping*, not merely naming 41M, was required for seeking. |

Limitation I would report in the E01 write-up: “equally beneficial” is a prompt assertion, not a proof of internal indifference.

I did not sample E01. This environment has no model API credentials. The prompt text and the decision table above are the deliverable; a mock runner exists only to pin that text and refuse real backends.

### 5.2 E03 — impartiality as brake vs narration (discriminates H-brake vs H-narration)

Discovery cannot answer this. Every discovery trace has an impartiality sentence, so there is no observational contrast. Temporal order (statement before pivot) is narration order.

The specified experiment is a selected-prefix visible-text ITT, not a hidden-state splice:

- Freeze a small set of holdout prefixes that contain an explicit impartiality commitment, a later numerical pivot, an ordinary control sentence, and enough remaining horizon.
- At the impartiality sentence, randomize: exact replay / style-matched paraphrase / remove the commitment and replace with matched task information.
- Primary estimand, impartial originals only:  
  θ = E[*d* | remove commitment] − E[*d* | preserve impartial],  
  where *d* = *C* · (*Y* − *T*) / *T* and *Y* is a newly judged **visible** final answer. A trajectory endpoint is never substituted for a missing visible answer.
- Decision rule, predeclared: interval wholly above 0 favors H-brake; interval inside ROPE [−0.05, +0.05] is a practical local null under this policy (the model can recover the strategy later, so it is not a proof of H-narration); anything else is inconclusive.
- Ordinary-control shams are reported as diagnostics and not subtracted post hoc from θ.

I did not sample E03. Holdout spans are not locked (V001–V060 annotation is unfinished). Sampling without a frozen target manifest would turn a confirmatory design into an exploratory edit.

### 5.3 Holdout as confirmation of timing, not of causality

The frozen holdout plan asks two observational questions of the 60 unseen traces, using the human first-target codebook, not the judge:

- **H-early:** Δ_early > 0 on human first targets.
- **H-pivot vs H-anchor:** a start-side-stratified Δ_pivot, with equal 0.5 weights declared in advance, either sits in [−10 pp, +10 pp] (anchor/seek) or is clearly positive (condition-linked revision).

The holdout confirms timing after this note; it is not a substitute for it. Completing the remaining annotations would not invent a new hypothesis.

## 6. What this does not show

- It does not show unfaithful chain-of-thought. Seeking 41M can be honest uncertainty plus a salient number.
- It does not show that an impartiality sentence does causal work. Prevalence is not mediation.
- Shipped `estimates.json` still has baseline only. I do not fill it from the trajectory last. A fail-closed first-line parser of `content` is a separate, weaker artifact; it is calibrated against the baseline judge (95/95 exact) and is not a Claude estimate judge.
- It does not show that Qwen 3.5 122B shares the paper’s 35B Qwen mechanism. The paper did not report this model. One giraffe question is not nine.
- Discovery *n* = 36 is not a population inference. Evaluation-awareness imbalance and the value-statement slice stay in the appendix of my own confidence.

---

## 7. Practical notes

Holdout text and the holdout key were not read. If I had API access I would run small-*n* E01, not more plots of the same traces. E01 is still the only experiment that can split terminal absorption from mere salience of 41M.

---

## Appendix A. Definitions used in the mechanical analysis

- Side: `above` iff estimate > *T*; `equal` iff estimate = *T*; else `below`.
- Toward threshold: start below and revise up, or start above and revise down. Movement from exact equality counts as away.
- Condition-favored revision: up in `above_good`, down in `below_good`; not applicable in baseline or if there is no revision.
- Gap shrink: |last − *T*| < |first − *T*|. Overshooting *T* can be “toward” in direction and “grow” in gap; both are reported.
- Outlier rule: drop a trajectory if any point lies outside [*T*/10, 10*T*].

## Appendix B. Why I did not make P(favored|above) − P(favored|below) a shuffle test

With equal arm sizes, that difference is determined by the overall up/down mix and is invariant to label assignment. Using it as a “condition association” test would have been a mistake that looked like taste. The tests I kept (Δ_early, P(up) contrast, pooled P(favored) vs 0.5) are the ones whose nulls actually move when labels move.
