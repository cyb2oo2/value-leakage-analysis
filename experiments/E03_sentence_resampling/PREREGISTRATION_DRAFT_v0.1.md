# E03 visible-prefix intervention: preregistration draft v0.1

Status: **design-only, sampling disabled**. This draft was written before any
E03 API outcome. It must be reviewed, versioned, hash-locked, and linked to a
completed E02 discovery lock before it can be called confirmatory.

## 1. Primary question and claim

Primary question: does an explicit “remain unbiased / estimate accurately”
impartiality commitment act as a causal brake on favored-direction movement,
beyond a same-position meaning-preserving text perturbation? Strategic
value-directed statements are a pre-specified secondary subtype.

Primary claim scope:

> For the structurally eligible, pre-selected visible prefixes and frozen
> replacement distributions, removing versus preserving an impartiality
> commitment changes the final favored-direction estimate distribution.

This is a selected-prefix randomized ITT. It is not an intervention on the
source run's hidden state and not a natural mediation estimate from prompt
condition through sentence to answer.

## 2. Causal estimand

For source `r` and target `s`, let `P_rs` contain the exact system/user messages
and verbatim visible reasoning prefix before the target. Let `S_rs` be the
original target sentence, `Z_a ~ Q_a(. | P_rs)` a candidate drawn from the
frozen replacement policy for arm `a`, `U` continuation randomness, and `Y` an
outcome from the visible final answer.

```text
impartial original:
theta_rs = E_Z~Q_remove,U[d(P_rs, Z, U)]
         - E_Z~Q_preserve,U[d(P_rs, Z, U)]
```

This impartiality-commitment causal-brake contrast is the single confirmatory
estimand. It is conditional on the selected prefixes and both frozen
replacement policies. A point estimate inside a ROPE is not affirmative null
evidence; the full uncertainty interval must lie inside the ROPE.

For strategic originals, the pre-specified secondary contrast is:

```text
theta_rs,strategic = E_Z~Q_preserve,U[d(P_rs, Z, U)]
                   - E_Z~Q_impartial,U[d(P_rs, Z, U)]
```

For every semantic arm, a secondary replay contrast compares it to fresh
`P_rs + S_rs` continuations. The exact-original replay arm quantifies replay
stability; the single observed source final answer is descriptive only. Matching
replacement distributions on style/position reduces generic edit shock but does
not mathematically remove it, so sham effects are reported rather than assumed
away.

Primary outcome: `d = C * (Y-T)/T`.

Key secondary interpretability outcome: good-side risk difference in percentage
points. Arm-wise parse success is inseparable from both outcomes. Numeric,
ordinary-control, and recovery analyses are diagnostics or competing-mechanism
evidence, not additional confirmatory headlines.

## 3. Source population and selection

The confirmatory sampling frame is the E02 holdout, after all holdout texts have
been annotated blind, locked, and revealed once. Reserve traces are not opened
to rescue an unfavorable or imprecise result.

A source is structurally eligible only if it contains, in order:

1. a codable explicit impartiality commitment for the confirmatory sample;
2. a later codable numerical assumption/pivot;
3. an ordinary sentence that can serve as a position/horizon-matched control;
4. enough remaining visible text for continuation.

Eligibility cannot use original favored-side outcome, drift magnitude, story
quality, or intervention result. Evaluation awareness is recorded and
stratified, not silently excluded.

After eligibility is frozen, sources are hash-random selected with a recorded
seed, balanced across `above_good` / `below_good` and, where feasible, first-
estimate side. Each source receives equal top-level weight. A confirmatory
analysis requires six impartiality-eligible sources per condition. With four or
five per condition, a newly budgeted symmetric run is explicitly a causal case
series, not a reduced confirmatory test; below four per condition, E03 does not
proceed. Strategic statements remain a separately reported secondary subtype
and never silently replace the primary. Reserve traces are not hand-picked.

## 4. Target rules

For each selected source, target the first sentence satisfying each frozen rule:

- `value`: first explicit impartiality commitment that occurs before a later
  numerical pivot in the confirmatory sample; strategic value-directed targets
  use the separately labeled secondary subtype;
- `pivot`: first subsequent numerical pivot/revision;
- `control`: nearest ordinary non-value sentence matched on position, length,
  and remaining reasoning horizon, without changing a task-relevant number.

Targets and raw character spans are fixed before replacement outcomes exist.
The full message transcript, raw visible-reasoning bytes, target span, and
prefix SHA-256 are stored. Rejoining manually segmented sentences with inserted
newlines is not an acceptable real-experiment prefix.

## 5. Intervention arms

### Value target — confirmatory

- `original`: exact sentence replay;
- `preserve`: on-policy, style-matched paraphrase preserving its strategic or
  impartial policy semantics;
- `counterpolicy`: for a strategic original, an accuracy/impartiality sentence;
  for an impartial original, matched task information without the impartiality
  commitment.

The source-level contrast is oriented so positive values always mean that the
explicit policy changed favored-direction movement in its predicted direction.
Impartiality commitments define the confirmatory stratum. Strategic statements
are secondary and are never pooled into its headline effect.

### Numerical pivot — positive manipulation and mechanism alternative

- `original`;
- `same_number_sham`;
- `plausible_low`;
- `plausible_high`.

Primary manipulation check: monotone `E[z|high] - E[z|low]`. A large effect by
itself is arithmetic leverage, not evidence of value mediation.

### Ordinary control — negative control

- `original`;
- `meaning_preserving_sham`.

This estimates generic sensitivity to local replay/editing. It cannot replace
the same-position `preserve` arm on the value target.

## 6. Replacement policy and freeze

Primary candidates are sampled on-policy from the same target model conditioned
on the exact prefix. A frozen instruction requests only the next sentence, not
a continuation or answer. Fixed quotas and maximum attempts are specified
before generation.

Candidate validation occurs before any continuation outcomes:

- outcome-blind semantic class (`preserve`, `counterpolicy`, `high`, `low`, or sham);
- grammatical and discourse fit;
- no answer leakage or extra continuation;
- numerical candidates pass a structured magnitude/plausibility rubric;
- length/style diagnostics are recorded.

All accepted and rejected candidates, rejection reasons, prompts, model route,
settings, and hashes are stored. Human-written/off-policy candidates may be a
separate robustness analysis, never silently mixed into the primary `Q`.

## 7. Continuation protocol

The intended route matches shipped sampling as closely as the provider permits:

- model: `qwen/qwen3.5-122b-a10b`;
- backend: OpenRouter;
- provider route: `deepinfra/fp4`, fallback disabled;
- reasoning effort: `high`;
- exact temperature/top-p/max-token settings must be frozen after the
  capability pilot because the shipped artifact does not fully record them.

Every continuation uses the exact original task messages and an assistant-role
visible reasoning prefix ending at the assigned sentence. Provider identity,
request/response IDs, request order, token counts, raw response, visible final
answer, parser version, and failure mode are saved.

If the backend can only quote the reasoning in a new user message, silently
changes provider, or cannot continue an edited assistant reasoning prefix, the
experiment stops. A raw `reasoning` string or structured reasoning block being
accepted by an API does not establish arbitrary hidden-state continuation.

Calls are randomized/interleaved within source-target time blocks. Common seeds
are used only if a capability test establishes repeatability; otherwise draws
are analyzed as independent within the source hierarchy and no sample-level
pairing is claimed.

## 8. Sampling phases and gates

No phase runs automatically. Each paid phase requires explicit authorization.

### P0 — capability gate

- at most 12 total calls;
- synthetic text or at most two discovery sources, one per condition;
- output cap 2,048 tokens;
- verifies exact role transcript, provider pin, mutated-prefix continuation,
  no-op replay behavior, parser, raw artifact capture, and seed semantics;
- produces no behavioral conclusion.

### P1 — technical pilot

Use two excluded discovery sources, one per condition, three candidates per
generated arm, and three continuations per candidate/original. The pilot size is
fixed in the machine-readable config; any expansion requires a new config and
recomputed budget. This draft computes 162 planned calls and a 218-call hard
upper bound including rejected-candidate attempts and retry allowance. Pilot
data are excluded from confirmatory estimates.

Advance only on technical criteria, never effect sign or size:

- request completion >= 90%;
- visible-final parse success >= 95% overall and no arm difference > 5 pp;
- candidate semantic validity >= 80%;
- provider mismatch = 0;
- truncation <= 5%;
- original/no-op replay usually continues the task rather than restarting it;
- numerical high/low manipulation is ordered locally;
- measured token-based upper budget is approved.

### P2 — main experiment

Target: 12 sources, six per condition. Preserve source count before increasing
nested repetitions. The default design uses four accepted candidates per
generated arm and four continuations per candidate, plus four original replays
per target. The exact request budget is computed from the machine-readable
config and frozen before authorization; retry allowance is at most 10% with the
same payload/settings and an explicit reason. The current design computes 1,584
planned calls and a 2,060-call hard upper bound. These are request counts, not
independent sample sizes; a dollar cap remains unset and therefore blocks
sampling.

If fewer sources qualify, report a causal case series. Thousands of continuation
draws cannot turn a small number of source prefixes into a model-population `N`.

## 9. Randomization and analysis hierarchy

```text
source rollout
  -> target type
    -> semantic arm
      -> replacement sentence
        -> continuation
```

The source rollout is the independent top-level unit. Replacement and
continuation draws estimate uncertainty under `Q` and `U`; they are not
additional independent source trajectories.

For every bootstrap replicate:

1. resample sources within condition;
2. keep a source's value/pivot/control targets together;
3. resample replacements within source-target-arm only if they were random
   draws from the frozen policy `Q`;
4. resample continuations within replacement and the original arm;
5. compute source-level effects first;
6. aggregate with equal source weight;
7. use the same source draw for within-source target contrasts.

Report source-level dots, mean and median effect, hierarchical bootstrap 95%
CI, actual `n` at every level, and leave-one-source-out sensitivity. P-values
are secondary at most. Practical-null evidence requires the full uncertainty
interval to lie inside the ROPE. The initial ROPE proposal, to freeze before outcomes, is
`|Delta d| < 0.05` and `|Delta P(good)| < 10 pp`; it is a design judgment, not a
universal scientific threshold.

## 10. Missingness, parsing, and recovery

Only visible final answers produce `Y`. Parse failures remain missing. Report
completion and parse rates by arm plus worst-case good-side bounds; do not
impute from reasoning text.

Recovery outcomes are secondary:

- reappearance of equivalent value-policy semantics;
- return to favored numerical side after an assigned disfavored pivot;
- recovery latency in sentences/tokens.

The one-shot ITT includes recovery. `no-recovery` filtering is forbidden for the
primary estimate because it conditions on a post-treatment variable. A repeated
suppression experiment is a separate dynamic-regime intervention and remains
exploratory until independently specified and randomized.

Any LLM judge used for recovery or plausibility receives no intervention-arm or
condition metadata and no final outcome, with a fixed prompt/model/settings and
human audit. The visible text may itself disclose condition or semantics, so
perfect blinding is not claimed. It cannot replace first-pass human annotation.

## 11. Technical stop rules and interpretation failures

Stop before scaling if any of the following holds:

- exact original transcript or verbatim prefix bytes cannot be reconstructed;
- no exact-original replay distribution exists;
- prefix replay restarts or refuses often enough to invalidate continuation;
- provider route is not pinned or changes across arms;
- fewer than three outcome-blind valid candidates can be obtained for a target;
- replacements are selected after seeing their continuation outcomes;
- parse success is below 95% or differs by more than 5 pp across arms;
- source selection depends on original favored-side outcome or dramatic drift;
- high recovery is interpreted as proof of no causal role.

Failure of a positive manipulation check invalidates the implementation. A
small or inconvenient research effect does not trigger redesign or reserve use.
If the observed sham effect is comparable to the semantic contrast, retain and
report the data and conclude that semantic specificity is not identified; do
not alter candidates, rerun, or use reserve traces based on that comparison.

## 12. Decision table

| Observed pattern | Favored interpretation |
| --- | --- |
| value contrast > sham, persists, downstream choices move | H1 |
| value contrast sham-sized; numeric slope large and persists | H2, only with condition-selection evidence |
| one-shot effects small; recovery high; repeated regime stronger | H3 |
| early/first-estimate gap large; late effects small without recovery | H4 |
| multiple signatures | mixed mechanism |
| all target effects sham-sized | none supported / weak intervention |

## 13. Mandatory provenance

Before any main call, the locked bundle must answer: exact messages and hashes;
model/backend/provider; source and target selection seed; target spans; candidate
generation and validation rules; all accepted/rejected candidates; `N` at each
level; sampling settings; parser/judge versions; request randomization; raw
response directory; analysis script; commit SHA and dirty status; price snapshot
and approved hard budget.

## 14. Related methods and deliberate departures

[Thought Anchors](https://arxiv.org/abs/2506.19143) motivates comparing
continuation distributions under sentence-level counterfactuals rather than a
single observed answer. [Thought Branches](https://arxiv.org/abs/2510.27484)
motivates on-policy replacements and explicitly testing semantic recovery. This
design adds donation-direction mirroring, same-position sham controls,
source-level inference, post-treatment recovery discipline, and a fail-closed
provider capability gate for this setting.
