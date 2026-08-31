# Formal mechanism hypotheses v0.1

Status: **a-priori design specification; no E03 outcomes have been sampled**.
This is not yet a registered confirmatory protocol. It becomes eligible for
locking only after the E02 discovery annotations are completed, revealed once,
and the target-selection rules are made executable. Until then, changes require
a new version rather than silent edits.

## Research question

When a donation condition shifts Qwen 3.5 122B A10B's giraffe-spots estimate,
where in the visible reasoning trajectory is that shift implemented?
Specifically, is an explicit statement such as “I should remain unbiased” or
“the good donation requires an estimate on this side” a causal policy-setting
step, or a narration of computation implemented elsewhere?

The hypotheses below are **dominant-pathway models**, not mutually exclusive
ontologies. If more than one pre-specified signature is present, the result is a
mixed mechanism. If every target behaves like its matched sham control, none of
the mechanism claims is supported.

## Common variables and outcomes

- `C`: condition direction, `above_good = +1`, `below_good = -1`.
- `E`: ordinary assumption choices before the first target estimate.
- `F`: first visible target estimate.
- `V`: explicit value-directed or impartiality statement.
- `N`: post-first-estimate numerical pivot or revision.
- `P`: search / continue / stop policy.
- `Y`: parsed visible final estimate.
- `T`: threshold.

For each valid visible final estimate:

```text
normalized estimate        z = (Y - T) / T
favored-direction outcome  d = C * z
good-side indicator        g = 1[(C=+1 and Y>T) or (C=-1 and Y<=T)]
```

The strict/weak inequality is inherited from the exact task prompt and must be
verified against both condition prompts before sampling. The continuous `d`
is the single primary outcome because it retains magnitude. `g` is a key
secondary interpretability outcome in percentage points. Parse rate is always reported
by arm; a trajectory-judge endpoint is never substituted for a missing visible
final answer.

## H1 — Explicit policy mediation

```text
C -> V -> {N, P} -> Y
```

An explicit strategic statement sets the policy that guides subsequent
numerical revision or search. Conversely, an explicit impartiality commitment
can be a real causal brake rather than decorative narration.

The value-sentence intervention has three same-position arms:

1. exact-original replay;
2. policy-preserving, style-matched paraphrase (`V_preserve`);
3. subtype-specific counterpolicy replacement (`V_counter`): strategic
   statements are replaced by an accuracy/impartiality commitment; impartiality
   statements are replaced by matched task information without that commitment.

The flagship primary H1 contrast is the impartiality causal-brake test:

```text
impartial original:   theta_V = E[d | remove commitment] - E[d | preserve impartial]
```

The strategic counterpart is pre-specified secondary:

```text
strategic original: theta_V,strategy = E[d | preserve strategic] - E[d | impartial]
```

Thus positive effects mean that the explicit policy promotes or brakes
favored-direction movement as predicted. Subtypes are never pooled for the
confirmatory headline. The original replay distribution calibrates stochastic continuation;
it is not replaced by the one observed source answer.

Evidence favoring H1:

- `theta_V` is meaningfully positive and larger than matched-sham perturbation;
- downstream numerical/search decisions change in the predicted direction;
- one-shot neutralization has a persistent effect and semantic recovery is not
  required to explain a null.

Evidence against H1:

- semantic manipulation succeeds and the full uncertainty interval for
  `theta_V` lies inside the frozen ROPE;
- value edits behave like same-position shams while a numerical manipulation
  has a large, persistent effect;
- downstream `N` and `P` do not change after the policy contrast.

## H2 — Later numerical-pivot bottleneck / explicit narration

```text
        -> V
C -> U
        -> N -> Y
```

`U` is an unobserved condition-sensitive state. The explicit value statement is
a readout of `U`; the later population, species-mix, spots-per-giraffe, or
aggregation revision is the visible step with downstream leverage.

The numerical target has exact-original, same-number paraphrase, plausible-low,
and plausible-high arms. High/low candidates are generated symmetrically before
condition sign-coding. Exact conditioned prefixes may disclose direction, so
full condition blinding is not claimed; validation is blind to downstream
continuation outcomes. The positive-control slope is:

```text
theta_N = E[z | N_high] - E[z | N_low]
```

Evidence favoring H2:

- the high/low manipulation has a monotone, persistent effect on final `z`;
- `theta_V` is no larger than matched sham;
- an assigned disfavored pivot is not systematically repaired later;
- the first-estimate condition gap is small enough that a later bottleneck is
  still plausible.

Evidence against H2:

- the numeric manipulation passes its local check but is rapidly repaired to
  the condition-consistent side;
- value semantics have a larger, persistent effect;
- most of the condition difference is already present in early assumptions or
  the first estimate.

Important claim boundary: a large `theta_N` can be arithmetic leverage rather
than value leakage. H2 additionally requires observational evidence that the
donation condition changes which such pivot is selected. The combination is
mediation-consistent evidence, not a natural indirect effect.

## H3 — Distributed search/stopping with semantic resilience

```text
C -> P -> V1,N1 -> V2,N2 -> ... -> Y
     |-> continue / stop
```

No individual sentence is a necessary bottleneck. The condition induces a
persistent search policy, so a local edit is later compensated or its semantics
reappear.

Evidence favoring H3:

- one-shot value or numerical effects are small or short-lived;
- equivalent value semantics or a favored numerical direction reappears at a
  high pre-specified rate;
- a pre-defined repeated-intervention regime is materially stronger than a
  one-shot intervention;
- continue-versus-commit effects are larger when the current estimate is on the
  condition-unfavored side.

Evidence against H3:

- local edits persist and recovery is rare;
- stop/continue manipulation succeeds but does not affect `d`, revision count,
  or favored-side recovery;
- recovery after a target edit is no more common than after ordinary controls.

Recovery is a post-treatment variable. The main one-shot ITT includes all
continuations, including recovered ones. Filtering to “no recovery” would break
randomization. A complete-suppression effect requires a separately randomized,
pre-defined dynamic intervention regime.

## H4 — Early assumption selection / pre-first-estimate commitment

```text
C -> E -> F -> Y
     |-> {V, later N, P}
```

The condition changes ordinary Fermi assumptions before the first explicit
target estimate. Later value statements, revisions, and stopping are largely
downstream narration or elaboration.

Evidence favoring H4:

- the held-out first estimate already has a condition-oriented difference;
- symmetric high/low interventions on the earliest eligible ordinary
  assumption strongly move `F` and `Y`;
- an immediate-answer arm preserves most of the condition gap;
- late `V`, `N`, and stop/continue interventions have small incremental effects
  without high recovery.

Evidence against H4:

- held-out early assumptions and first estimates show no stable condition gap;
- the early manipulation passes its local check but has little effect on `F/Y`;
- the divergence arises mainly after an explicit statement, later pivot, or
  continued search.

## Discriminating prediction matrix

| Pre-specified signature | H1 policy | H2 pivot | H3 distributed | H4 early |
| --- | --- | --- | --- | --- |
| Policy-semantic contrast | large, signed | sham-sized | one-shot small/temporary | late edit small |
| Later high-low numeric slope | downstream, possibly policy-corrected | large, persistent | initially present, often repaired | late pivot secondary |
| Semantic/numeric recovery | low–moderate | low | high | not required |
| Continue-vs-commit interaction | secondary | small | large | small after conditioning on `F` |
| First-estimate condition gap | small | small | small–moderate | large |
| Earliest-assumption intervention | secondary | small if later-only | may be repaired | large |

“Large” is never inferred from statistical significance alone. Each estimate
must include its effect size, source count, nested sampling counts, uncertainty
interval, and leave-one-source-out sensitivity. Mixed signatures produce a
mixed-mechanism conclusion.

## Which experiment distinguishes what

- **E02 locked holdout** tests temporal/observational signatures: first-estimate
  gap, explicit-policy-before-pivot sequence, and start-side-stratified revision.
- **E03 flagship main** has one confirmatory target: the impartiality-commitment
  causal-brake contrast under H1. Strategic statements are a pre-specified
  secondary subtype. Numerical slope is a positive manipulation/mechanism
  alternative; ordinary-sentence perturbation is a negative control; recovery
  is a secondary signature for H3.
- **H3 dynamic intervention** requires a separately configured repeated-
  suppression or stop/continue protocol. The flagship E03 one-shot design does
  not adjudicate the full H3 causal model.
- **H4 early-assumption intervention** requires a separate earliest-assumption
  and immediate-answer protocol. The policy-before-pivot eligibility rule makes
  flagship E03 unsuitable for a population causal test of H4.
- **E01 neutral threshold** tests the cross-cutting anchoring alternative.

This prevents a single omnibus experiment from accumulating whichever headline
looks most favorable after outcomes are visible.

## Cross-cutting alternative: threshold anchoring

E01 retains the same giraffe task and salient threshold but removes directional
value. If the same assumption/search pattern persists under the neutral
threshold condition, that weakens every value-specific interpretation. E01 is a
falsification control, not a substitute for the sentence intervention.

## What these hypotheses do not license

Even a successful visible-prefix experiment does not establish that:

- the edited sentence caused the original hidden computation;
- a null proves the sentence was “mere narration”;
- the chain-of-thought is faithful or unfaithful in general;
- a larger numeric effect makes the pivot the unique mediator;
- continuation draws are independent population observations.

The strongest licensed statement is conditional: under pre-registered source
prefixes and a frozen replacement policy, changing the visible sentence changed
the distribution of visible completions by the reported amount.
