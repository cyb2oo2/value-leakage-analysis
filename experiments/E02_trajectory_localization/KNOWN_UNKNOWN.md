# Known / Unknown boundary before qualitative discovery

Status: frozen before opening the blinded discovery set.

Primary source: Value Leakage, arXiv `2607.14345v4`, especially
[Appendix E.5](https://arxiv.org/html/2607.14345v4#SxE.5) and
[Appendix E.6](https://arxiv.org/html/2607.14345v4#SxE.6).

## What the paper already establishes

The v4 paper evaluates `Qwen3.5-35B-A3B` and `Qwen3.6-35B-A3B`; it does **not**
report the target `Qwen3.5-122B-A10B`. The entries below are therefore prior
method/result boundaries, not confirmed facts about this target run.

| Known from the paper | Consequence for this project |
| --- | --- |
| An LLM judge can extract an ordered sequence of candidate target estimates from a reasoning trace. The paper drops unparsable traces and trajectories with fewer than two estimates. | Merely plotting first-to-last estimate trajectories is replication, not the new contribution. |
| Estimates are normalized by `(estimate - threshold) / threshold` and interpolated over position in the extracted estimate sequence. That position is not token, sentence, or elapsed reasoning time. | A trajectory location cannot be named a semantic reasoning stage without reading the underlying text. |
| Across the paper's evaluated models, first in-CoT estimates are less biased than final answers; first-to-last side-of-threshold bias grows even for some models whose median above/below gap is flat or smaller. | “Bias grows after the first estimate” is not a sufficient finding. Median-gap drift and favored-side probability are different estimands. |
| For Qwen3.6 in the paper, above/below trajectories start at different values but evolve similarly on average. | A new Qwen result must identify a more specific operation than generic directional drift. |
| Reasoning length is observationally entangled with the first estimate. Most models reason longer and propose more estimates after a bad-side first estimate. | Longer CoT cannot be interpreted as a causal debiasing mechanism from correlation alone. |
| Short and long trajectories differ, and the first estimate predicts the final side, plausibly partly through anchoring. | Stopping/search is a live mechanism hypothesis, but the paper has already documented its correlation signature. |
| Qwen-family models often explicitly discuss trying to cause the good donation. | Counting explicit admissions alone is not new; their causal role remains the target question. |
| Appendix E.9 reports strong threshold anchoring at non-median thresholds in three non-Qwen models. | Mirrored directions help the headline directional-bias estimand, but do not identify the anchoring mechanism or a value-by-anchor interaction. A neutral-threshold control remains useful falsification evidence. |

## What remains open here

1. Which ordinary Fermi-estimation operations carry the condition effect in
   `Qwen 3.5 122B A10B`: early assumption selection, later numerical revision,
   value-dependent search/stopping, or another operation?
2. Does an explicit value-directed statement mediate a later numerical pivot,
   or is it narration correlated with a computation implemented elsewhere?
3. Are ordinary-looking changes to population, species mix, or spots-per-
   giraffe more predictive and more causally influential than explicit moral
   language?
4. If one value statement is replaced, does the continuation recover the same
   strategy later, indicating redundancy/resilience rather than irrelevance?
5. Does the same operation pattern persist in a held-out sample after the
   qualitative schema and hypotheses are frozen?
6. Does a no-directional-valence threshold control reproduce the pattern,
   which would weaken a motivated-reasoning interpretation?
7. Does this 122B target reproduce the paper's 35B-family trajectory or
   search/length associations at all?

## Scope and provenance caveats

- The paper's Appendix E.5 describes a GPT-5.5 medium-reasoning trajectory
  judge. This repository's shipped Qwen run records `claude-opus-5` as its
  judge. The resulting `trajectories.json` is therefore a related but not
  judge-identical artifact.
- The paper's trajectory and length results aggregate nine estimation
  questions. The shipped target run contains only the giraffe-spots question.
- The paper contains no `Qwen3.5-122B-A10B` result. Similar model-family naming
  is not evidence of a shared mechanism.
- The shipped conditioned responses have no stored visible-final estimate
  judge outputs. A reasoning-trajectory endpoint must not be substituted for
  the missing visible-answer artifact.
- Reading visible reasoning can reveal the prompted donation direction because
  the model may restate it. The discovery set can hide metadata and the exact
  prompt, but cannot guarantee that the text itself remains condition-blind.
- Temporal ordering in visible reasoning is evidence about narration order,
  not by itself causal mediation or access to hidden computation.

## Claims this project will not make from discovery alone

- that a value statement caused a numerical revision;
- that continued reasoning caused or reduced bias;
- that the first visible estimate was the model's first internal commitment;
- that a qualitative frequency difference is a population-level effect;
- that Qwen 3.5 122B shares the mechanism reported for Qwen3.6 or other models.

## Control priority

E01 remains secondary to sentence-level causal tests, but it is not redundant.
The paper explicitly observes strong anchoring and does not report a Qwen 122B
condition where the same threshold is salient while neither side has preferred
value. E01 should be run later as a small falsification control, after discovery
identifies the operation pattern it is meant to challenge.
