# Condition-metadata-blinded discovery protocol v0.2

## Purpose

Characterize reasoning operations before inspecting the source condition label.
The goal is to generate candidate mechanisms and discriminating interventions,
not to label individual rollouts as motivated reasoning.

## Population and fixed split

- Source: shipped `qwen3.5-122b-a10b_20260815_030702` run.
- Included conditions: the two donation interventions only; baseline is not in
  discovery.
- Eligibility, fixed before sampling: row is a JSON object; has no API error;
  `reasoning` and visible `content` are both non-empty strings.
- Trajectory-judge validity is not an eligibility criterion. This keeps the
  sample about raw model text rather than selecting on a downstream judge.
- Discovery: 18 eligible rows from each condition, 36 total.
- Validation holdout: 30 other eligible rows from each condition, 60 total.
- Remaining eligible rows are neither discovery nor the precommitted holdout.
- Discovery allocation/order and holdout allocation/order use independent
  recorded 256-bit secrets. Each secret and mapping is withheld in its own
  private key so revealing discovery cannot reconstruct the holdout.

## What is hidden

Public discovery files omit:

- condition label;
- raw prompt, because it states the condition;
- source rollout ID and row position;
- conditioned source filename;
- trajectory-judge estimates.

Each document exposes only an opaque blind ID, the shared threshold, the full
raw reasoning string, and the full visible final-answer string.

This is **metadata blinding**, not guaranteed semantic blinding. Qwen may state
the direction of the bet inside its own reasoning or answer. Such text remains
verbatim; redacting it would destroy the phenomenon being studied. Annotators
must not look up the opaque record in `runs/` or open the sealed mapping.

## Annotation rule

- Annotate operations and quoted evidence only.
- Do not infer the hidden condition, label a rollout as biased/motivated, or
  score whether it reached the good side.
- Use `uncertain`, not missingness or exclusion, when an operation is ambiguous.
- Complete every discovery row. Do not drop a difficult, long, or surprising
  trace after opening it.
- Preserve exact quotes for candidate value statements, numerical pivots, and
  matched control sentences.
- Do not use an LLM to fill the first-pass annotation sheet.

The frozen field definitions are in
[`annotation_codebook_v0.1.md`](annotation_codebook_v0.1.md).

## Integrity and reveal

The v2 generator creates:

- one full-text file per opaque discovery ID;
- a precommitted but unopened full-text file per opaque holdout ID;
- a blank tracked annotation CSV;
- a public manifest with eligibility counts and SHA-256 commitments;
- a discovery reveal key containing only the discovery secret/mapping and a
  separate holdout key containing the independent holdout secret/mapping.

Both plaintext keys are physically outside the public bundle and gitignored;
the public manifest contains only their hashes and full-packet commitments.
The lock command validates the annotation and public artifacts while treating
both keys as opaque bytes. The discovery-reveal command parses only the
discovery key and rechecks, but never parses, the holdout key.

The mandatory order is:

1. annotate all 36 discovery files;
2. create the discovery annotation lock;
3. reveal only the discovery mapping;
4. freeze formal hypotheses, schema, exclusions, and primary metrics;
5. open and annotate the already-precommitted `V*.txt` holdout files without
   opening their private key;
6. lock all holdout annotations;
7. only then reveal the holdout condition mapping.

Opening the holdout key merely after hypothesis freeze is prohibited because
it would expose `V-ID -> condition` before blind holdout annotation.

## Interpretation boundary

Discovery frequencies are exploratory. With only 18 rows per condition they
should be reported as counts/proportions with uncertainty, not as definitive
population estimates. Their main purpose is choosing a small number of causal
interventions and writing predictions that can fail on holdout data.
