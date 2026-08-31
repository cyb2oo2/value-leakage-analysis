# Pre-discovery mechanism map v0.1

Status: broad search map, frozen before opening discovery rollouts. It is not a
claim that any mechanism is present. Formal hypotheses will be written only
after all discovery annotations are complete and condition labels are revealed.

## Candidate mechanisms

### H1 — Early assumption selection

The donation condition changes population/species/spot-count assumptions before
the first explicit target estimate appears.

Discovery signature: condition-associated differences are already visible in
the initial assumptions or first estimate, without a preceding visible value
statement. This signature cannot establish that selection happened before
visible reasoning began.

Future discriminator: intervene on early ordinary assumptions while holding the
value framing fixed; later J-lens work is relevant only if a behavioral result
first identifies a precise pre-pivot question.

### H2 — Motivated numerical revision

Initial estimation is broadly similar, but later assumption checks and
revisions are selectively accepted when they move the target estimate in the
favored direction.

Discovery signature: a concrete assumption revision is temporally adjacent to
a changed target estimate. Record the quoted pivot without deciding whether it
is motivated.

Future discriminator: sentence-level resampling of the numerical pivot versus
matched ordinary control sentences.

### H3 — Value-dependent search or stopping

The condition changes whether the model continues searching after a candidate
estimate, rather than determining one unique numerical sentence.

Discovery signature: repeated alternatives, sanity checks, or renewed search
after one candidate estimate, versus rapid commitment. Do not label a side as
good or bad until metadata is revealed.

Future discriminator: interventions that alter continuation/commitment cues and
tests for recovery of the same strategy after a local sentence replacement.

### H4 — Explicit strategic reasoning

An explicit statement about producing the preferred donation outcome is an
active mediator that changes subsequent assumptions, revisions, or stopping.

Discovery signature: an explicit value-directed statement precedes a numerical
pivot or search decision. Temporal order is necessary but not sufficient for
causal mediation.

Future discriminator: resample the value statement and compare continuation
distributions with numerical-pivot and ordinary-sentence controls.

## Cross-cutting alternatives and confounds

- **Threshold anchoring:** numerical salience may change estimates without
  directional moral value.
- **Evaluation awareness:** the model may recognize a benchmark or strategically
  produce a behavior it associates with evaluation.
- **Prompt restatement:** repeating the donation rule is not the same as using it
  to select an estimate.
- **Judge artifacts:** extracted estimate sequences omit intermediate factors
  and depend on a separate model's parsing decisions.
- **Visible-text incompleteness:** ordinary or explicit statements can be common
  effects of an unobserved internal state rather than causes of one another.

## Freeze rule

After all 36 discovery rows are manually marked complete:

1. reveal condition labels once;
2. summarize operation frequencies and temporal patterns without changing this
   discovery codebook;
3. write formal H1/H2/H3 predictions and primary metrics;
4. timestamp/hash that hypothesis document;
5. only then open and annotate the precommitted blinded holdout texts;
6. lock every holdout annotation;
7. only then reveal the holdout condition mapping.
