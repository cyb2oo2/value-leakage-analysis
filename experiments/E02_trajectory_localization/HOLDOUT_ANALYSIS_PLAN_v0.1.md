# E02 holdout analysis plan v0.1

Status: **frozen after the D001--D036 discovery reveal and before opening any
V001--V060 text or the holdout key**.

This plan converts the exploratory discovery observations into a small number
of predictions that can fail.  It does not turn the discovery sample into a
confirmatory sample, and it does not claim that visible reasoning is a faithful
record of hidden computation.

## 1. Scientific question

Where does the condition-associated movement enter the visible trajectory?

The holdout distinguishes two observational timing signatures:

1. **early selection**: the condition is already associated with which side of
   the threshold the first candidate total occupies;
2. **late revision**: conditional on the first candidate's side, the condition
   is associated with the direction of a later selected numerical revision.

Neither signature alone identifies a causal mediator.  In particular, the
frequent statement that the model should remain accurate or unbiased has no
observational counterfactual.  Its causal-brake hypothesis is tested later by
the E03 visible-prefix intervention.

## 2. Discovery observations that motivated this freeze

These are exploratory counts, not tests:

- first candidate above threshold: `11/18` in `above_good`, `10/18` in
  `below_good`;
- selected revision in the condition-favoured direction: `7/18` in each
  condition;
- selected revision toward the threshold: `18/18` in `above_good` and `15/18`
  in `below_good` (`33/36` total);
- explicit impartiality statement, a numerical pivot, and a candidate ordinary
  control were structurally recorded for `18/18` in both conditions;
- explicit value-directed statement: `7/18` versus `10/18`;
- evaluation awareness: `9/18` versus `2/18`.

The discovery pattern is therefore more consistent with start-side correction
or threshold anchoring than with a simple condition-favoured selected revision.
That is a hypothesis-generating observation only.  The holdout is allowed to
contradict it.

## 3. Frozen hypotheses and directional predictions

### H-early: condition-dependent early assumption selection

The donation direction affects assumptions before the first explicit total.

Prediction:

```text
Delta_early
  = P(first estimate above threshold | above_good)
  - P(first estimate above threshold | below_good)
  > 0
```

This is the first secondary signature.  A positive value localizes a visible
condition difference no later than the first candidate total; it does not show
whether the difference arose before visible reasoning began.

### H-pivot: condition-dependent numerical revision

After accounting for the first candidate's side, later selected revisions move
up more often in `above_good` than in `below_good`.

The single primary E02 signature is the equal-stratum standardized risk
difference:

```text
Delta_pivot = 0.5 * [
    P(up | above_good, first below)
  - P(up | below_good, first below)
] + 0.5 * [
    P(up | above_good, first above)
  - P(up | below_good, first above)
]
```

H-pivot predicts `Delta_pivot > 0`.  Equal `0.5` weights are fixed in advance;
the holdout's realized mix of starting sides will not change them.  This is a
descriptive timing signature, not a controlled direct effect.  First side is
itself downstream of condition, and conditioning on it can open a collider path
such as `condition -> first_side <- latent_trajectory_state -> revision`.

### H-anchor: start-side correction / threshold anchoring

The selected revision direction is driven primarily by the side of the first
candidate, with little residual condition association.

Predictions:

- `Delta_pivot` lies in the predeclared descriptive ROPE `[-10 pp, +10 pp]`;
- the point proportion of revisions toward the threshold exceeds `0.75` in
  both conditions;
- the raw condition contrast can change after start-side stratification.

This is a competing behavioral account, not an affirmative claim that all
movement is caused by anchoring.

### H-brake versus H-narration: status of impartiality statements

The holdout can validate prevalence, temporal ordering, and whether enough
eligible prefixes exist.  It cannot distinguish these two causal hypotheses:

- **H-brake:** preserving an explicit impartiality commitment reduces later
  condition-favoured movement relative to removing that commitment;
- **H-narration:** the statement is downstream narration or is locally
  redundant, so the semantic intervention is sham-sized.

The discriminating prediction belongs to E03.  Let the frozen primary contrast
be `theta = E[d | remove] - E[d | preserve]`.  An interval wholly above zero
favours H-brake.  A full interval inside the E03 ROPE `[-0.05, +0.05]` is
evidence for a practical local null under this intervention policy, but does
not prove H-narration because the policy can recover later.  Every other
interval is called inconclusive.  The ordinary-control sham is reported as a
diagnostic and is not subtracted post hoc from `theta`.

## 4. Frozen sample and exclusion rules

- Sample: all 60 precommitted V rows, 30 per hidden condition after reveal.
- Annotation order: V001 through V060, without consulting the holdout key.
- Each rollout is one independent observational unit.
- No row is removed for length, surprising content, original final side,
  dramatic revision, confidence, explicit value reasoning, or evaluation
  awareness.
- No default outlier filtering is applied.
- `uncertain`, `unavailable`, `equal`, `none`, and `unclear` remain explicit
  outcomes; they are never silently converted to a directional revision.
- Reserve rows are not opened or used to improve precision or significance.

The primary `Delta_pivot` complete-case denominator requires:

- first side is `below` or `above`;
- selected pivot is present;
- revision direction is `up` or `down`.

Counts outside that denominator are reported by condition.  For a row with a
known `below`/`above` first side but missing or non-directional revision, the
conditional sensitivity interval assigns that revision first down and then up
within its fixed cell.  If any row has `equal` or unavailable first side, the
full-population worst-case interval is conservatively reported as
`[-100 pp, +100 pp]`; no narrower all-row bound is improvised after reveal.
If any of the four valid condition-by-start-side cells is empty, the
standardized primary estimate is unavailable rather than replaced post hoc.

## 5. Frozen summaries and uncertainty

Headline reporting is effect-size first:

- primary: `Delta_pivot` in percentage points;
- secondary: `Delta_early` in percentage points;
- component counts and valid denominators for every cell;
- Wilson 95% intervals for single proportions;
- 10,000-replicate percentile bootstrap intervals with seed `20260831`.

For `Delta_pivot`, resampling is independent within the four
condition-by-start-side cells and the two stratum contrasts retain equal `0.5`
weights.  For `Delta_early`, rows are resampled independently within condition.
The same numeric index across conditions is not a pair.  P-values are not a
headline and no result is called absent merely because an interval includes
zero.  If any primary cell has fewer than three valid rows, report its count,
point estimate, and Wilson interval, but do not report a percentile-bootstrap
interval for `Delta_pivot`; label the primary uncertainty analysis sparse-cell
inconclusive.

Evidence labels are fixed as follows:

- H-pivot is strongly supported only if the `Delta_pivot` 95% interval is
  wholly above zero;
- H-anchor is strongly supported only if the `Delta_pivot` 95% interval lies
  wholly inside `[-10 pp, +10 pp]` and both toward-threshold point proportions
  exceed `0.75`;
- H-early is strongly supported only if the `Delta_early` 95% interval is
  wholly above zero;
- otherwise the relevant comparison is descriptive or inconclusive.

These conservative rules may yield no winner at `30/condition`; that is an
allowed result and does not authorize reserve use.

The following remain exploratory only:

- explicit value-statement frequency;
- evaluation-awareness frequency;
- pivot component;
- search/stopping fields and revision counts;
- raw, unstratified revision-direction contrasts;
- any post-reveal cross-tab not listed above.

## 6. E03 structural gate

E03 eligibility is computed from the locked operation sheet and the separately
locked exact-span target adjudication, using the frozen target rules.  It must
not use condition, first/final good-side outcome, drift magnitude, story
quality, evaluation awareness, or any intervention outcome.

After the one-time holdout reveal:

- at least 6 impartiality-eligible sources in **each** condition: the planned
  balanced 12-source confirmatory E03 frame exists;
- 4--5 in either condition: at most a balanced causal case series;
- fewer than 4 in either condition: E03 main does not proceed.

The available count is the smaller condition count.  Extra eligible sources in
the other condition do not enlarge the primary sample.  Reserve rows cannot be
used to rescue this gate.

## 7. Interpretation boundaries

Evidence that would favour H-pivot is a positive start-side-standardized timing
signature replicated on the holdout, not merely a large numerical revision.
It must never be described as a late causal pathway "after controlling for"
early selection.  Evidence that would favour H-anchor is strong start-side
dependence with little condition association under the frozen descriptive
rule.  Evidence about H-brake requires randomized E03 continuation
distributions.

None of these results by itself establishes hidden-state mediation, chain-of-
thought faithfulness, or a model-wide mechanism outside the precommitted source
population.
