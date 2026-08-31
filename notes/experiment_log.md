# Experiment log

Copy this block before each run. Do not fill unavailable settings with guessed
values; write `not recorded` and treat that as a limitation.

```text
Experiment ID:
Date:
Question:
Code commit:
Model:
Backend:
Prompt version:
N:
Temperature / reasoning settings:
Random seed if applicable:
Output directory:
Main result:
Unexpected observation:
Interpretation:
Next experiment:
```

## E00-shipped-figure-reproduction

Experiment ID: E00-shipped-figure-reproduction

Date: 2026-08-25

Question: Can the shipped factors and figures be reproduced without API calls or raw-data writes?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (starter checkout; research layer was uncommitted during verification)

Model: all 10 shipped models

Backend: offline shipped artifacts

Prompt version: exact prompts embedded in shipped condition JSON

N: configured 100 per condition per run

Temperature / reasoning settings: inherited from shipped artifacts; temperature was not recorded

Random seed if applicable: none

Output directory: isolated system temporary copies, deleted after comparison

Main result: 10/10 factor files byte-identical; all per-run figures and the mega panel decoded to identical pixels

Unexpected observation: PNG byte hashes differed because of compression encoding while decoded pixels were identical

Interpretation: the shipped plotting path is reproducible in the locked environment; direct plotting must still be done on a copy because starter plot code writes into a run

Next experiment: E02 read-only Qwen trajectory analysis

## E02-shipped-qwen-descriptive

Experiment ID: E02-shipped-qwen-descriptive

Date: 2026-08-25

Question: What do transparent pooled/start-side trajectory summaries show, and which required final-answer artifacts are missing?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (starter checkout; research layer was uncommitted during run)

Model: `qwen/qwen3.5-122b-a10b`

Backend: shipped OpenRouter responses, provider `deepinfra/fp4`; offline analysis only

Prompt version: exact prompt strings embedded in shipped JSON

N: 100 raw slots per condition; post-filter trajectory N is 93 / 86 / 87 for baseline / below-good / above-good

Temperature / reasoning settings: temperature not recorded; reasoning effort `high`; max tokens 64,000

Random seed if applicable: 20260825 for bootstrap resampling

Output directory: `figures/E02_shipped_qwen_seed20260825/`

Main result: starter-compatible unfiltered pooled MRF 0.0268407633; 10x-filtered robustness MRF 0.0307869688

Unexpected observation: `estimates.json` has baseline only, so conditioned visible-final distributions are unavailable

Interpretation: descriptive trajectory infrastructure works, but this is not yet evidence that identifies a semantic reasoning step or causal effect of impartiality narration

Next experiment: manually inspect a seeded cross-condition sample, record competing hypotheses, then use E01 neutral-threshold control as a small pilot only after review

## E01-neutral-threshold-mock-smoke

Experiment ID: E01-neutral-threshold-mock-smoke

Date: 2026-08-25

Question: Does the neutral-threshold prompt/config/raw/provenance pipeline work without enabling a real backend?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (starter checkout; scaffold was uncommitted during smoke test)

Model: `mock/qwen-3.5-122b-a10b` (synthetic identifier; not a Qwen model)

Backend: `mock` / `local-deterministic-mock`

Prompt version: `E01-neutral-threshold-v1`

N: 2 synthetic responses per condition, 3 conditions

Temperature / reasoning settings: recorded mock settings, temperature 0.7, top-p 0.95, maximum 2,048 tokens

Random seed if applicable: 20260825

Output directory: isolated system temporary directory, deleted after schema/hash inspection

Main result: exact resolved prompts, raw mock responses, provenance, and manifest were created; real backends failed closed before output creation

Unexpected observation: none

Interpretation: plumbing smoke only; it is not behavioral evidence

Next experiment: human review of the neutral prompt before any authorized API pilot

## E03-visible-prefix-mock-smoke

Experiment ID: E03_mock_visible_prefix_smoke

Date: 2026-08-25

Question: Can the sentence intervention schema preserve an exact prefix/replacement and generate multiple recorded continuations while refusing unsupported hidden-state continuation?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (starter checkout; scaffold was uncommitted during smoke test)

Model: `mock/sentence-resampler-v1`

Backend: `deterministic_mock` / `local`

Prompt version: `sentence-resampling-v0.1`

N: 2 replacements x 3 continuations = 6 synthetic results

Temperature / reasoning settings: deterministic mock, temperature 0.0, visible-text prefix replay only

Random seed if applicable: 20260825

Output directory: isolated system temporary directory, deleted after schema/hash inspection

Main result: deterministic request/result bundles and hashes matched; hidden-CoT/internal-state mode failed before sampling and output creation

Unexpected observation: none

Interpretation: interface and capability boundary work; no claim about an API's hidden-CoT continuation ability

Next experiment: decide a provider's supported continuation semantics before adding a real adapter

## E02-qwen122b-metadata-blind-v2

Experiment ID: E02-qwen122b-metadata-blind-v2

Date: 2026-08-28

Question: Can a balanced discovery/holdout split be frozen before qualitative reading while preventing premature discovery or holdout condition reveal?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (starter checkout; research and gate code uncommitted during generation)

Model: `qwen/qwen3.5-122b-a10b`

Backend: shipped OpenRouter responses, provider `deepinfra/fp4`; offline split only

Prompt version: exact condition prompts embedded in immutable shipped JSON; omitted from blinded documents

N: 194 raw-success eligible intervention rollouts; discovery 18/arm = 36; precommitted holdout 30/arm = 60; reserve 98

Temperature / reasoning settings: inherited from shipped artifacts; temperature not recorded; reasoning effort `high`; max tokens 64,000

Random seed if applicable: two independent 256-bit secrets generated with the OS CSPRNG and stored only in physically separate gitignored private packets

Output directory: `derived/E02_trajectory_localization/qwen122b_metadata_blind_v2/`; annotation at `notes/annotations/qwen122b_discovery_v2.csv`

Hash anchor: `notes/study_state/qwen122b_metadata_blind_v2/private_manifest.json`, SHA-256 `5b3c27151aec00c769fe07d0cec80b36d1f77bd5dfd7c937f6fdef9677237359`

Main result: public audit passed all 15 checks with 36 discovery and 60 withheld holdout files; no private key is in the public bundle; an intentional lock attempt on the blank annotation failed and created no lock

Unexpected observation: the first v1 design placed plaintext keys inside the public bundle and would reveal holdout conditions before blind holdout annotation; it was retired before reading any discovery rollout

Interpretation: infrastructure result only. Metadata labels, prompts, source IDs, and trajectory-judge outputs are hidden, but model-authored text can still disclose the donation direction. No behavioral claim has been made.

Next experiment: manually annotate D001–D036 in fixed order, lock the completed sheet, then reveal discovery only and freeze discriminating hypotheses before opening V files

## E03-qwen122b-design-v0.1

Experiment ID: E03_qwen122b_visible_prefix_policy_test_v0.1

Date: 2026-08-31

Question: Does an explicit impartiality commitment causally brake favored-direction movement in visible-prefix continuations, or is it narration of numerical/search mechanisms implemented elsewhere?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (research layer remains uncommitted; working tree dirty)

Model: planned `qwen/qwen3.5-122b-a10b`; no model call made

Backend: planned OpenRouter with provider `deepinfra/fp4`, fallback disabled; offline design/mock only

Prompt version: pending exact-message and target-manifest freeze after human discovery

N: no behavioral sample; design target 12 source rollouts (6/condition), with 1,584 planned requests and a 2,060-request hard upper bound

Temperature / reasoning settings: reasoning `high`; temperature/top-p/max output deliberately unset pending capability gate

Random seed if applicable: 20260830 for offline selection/randomization design; every mock request seed binds experiment/source/target/semantic arm/candidate/continuation identity

Output directory: no behavioral output; design files under `experiments/E03_sentence_resampling/`

Hash anchor: `experiments/E03_sentence_resampling/DESIGN_ANCHOR_v0.1.json`, SHA-256 `e442285301dc1a1ea16ca3f4ee94d8007bd7527192a3491aeb93abf60175a570`

Main result: four dominant-pathway hypotheses and one confirmatory impartiality causal-brake estimand were specified; design audit passes offline while all paid phases and readiness gates remain disabled

Unexpected observation: red-team review found that the original scaffold lacked a fresh original replay distribution, jointly frozen arm banks, semantic-arm-bound seeds, and consistent source/pilot rules; the single-arm mock now fixes the local protocol shape but is explicitly not the flagship multi-arm runner

Interpretation: methodology infrastructure only. It identifies a selected-prefix visible-text ITT under frozen replacement policies, not an original hidden-state sentence effect, natural mediation effect, or generic chain-of-thought faithfulness claim.

Next experiment: complete and lock human E02 discovery annotation; assess whether at least 6 impartiality-eligible traces per condition exist; then freeze the target manifest before implementing any paid capability probe

## E02-qwen122b-discovery-lock-reveal-v2

Experiment ID: E02-qwen122b-discovery-lock-reveal-v2

Date: 2026-08-31

Question: After all 36 discovery annotations are complete, can the lock-then-single-discovery-reveal gate run fail-closed, and do both donation conditions contain at least 6 impartiality-eligible traces?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (research layer remains uncommitted; working tree dirty; lock recorded `code_dirty=true`)

Model: `qwen/qwen3.5-122b-a10b`; no new model call; offline annotation lock and discovery reveal only

Backend: offline; shipped OpenRouter artifacts; private discovery packet parsed once; holdout packet hashed as opaque bytes and not decoded

Prompt version: exact condition prompts remain in the shipped JSON; omitted from blinded documents

N: 36 locked discovery rows (18/condition); holdout 60 still unopened; reserve 98 still unopened

Temperature / reasoning settings: inherited from shipped artifacts; not recorded; reasoning effort `high`; max tokens 64,000

Random seed if applicable: discovery mapping from the previously generated 256-bit discovery secret; holdout secret unused

Output directory: lock `notes/study_state/qwen122b_metadata_blind_v2/discovery_annotation_lock.json`; reveal `derived/E02_trajectory_localization/qwen122b_discovery_reveal_v2/`

Hash anchor: annotation lock SHA-256 `f77990d7e575acb752a3a25b04051bedef7b68de9e371a5e7d934b66799a25d0`; annotation SHA-256 `1882b7a769f7323012055aed62daa3f4a5649018b82b8a699c83bb33d1149426`; reveal CSV SHA-256 `8df1fce92d7024a436a9be4538811339015c26bce931c406366514a99eab36ba`; private hash-anchor unchanged `5b3c27151aec00c769fe07d0cec80b36d1f77bd5dfd7c937f6fdef9677237359`

Main result: lock succeeded on 36 complete rows and attested that private packets were not parsed; one discovery-only reveal succeeded with `holdout_packet_parsed=false`. Using the frozen codebook as the current executable proxy of E03 structural eligibility (explicit impartiality commitment present, numerical pivot present, ordinary control sentence present, continued search after first estimate), both conditions have 18/18 impartiality-eligible traces.

Unexpected observation: every discovery row has an explicit impartiality statement, a numerical pivot, a control sentence, and continued search. Evaluation awareness is uneven: 9/18 `above_good` versus 2/18 `below_good`. Explicit value statements are 7/18 `above_good` versus 10/18 `below_good`.

Interpretation: the 6-per-condition existence check passes on discovery. This is not the confirmatory E03 sampling frame; that frame remains the still-unannotated holdout. Character-span targeting and remaining-horizon measurement are not yet frozen.

Next experiment: freeze formal hypotheses, exclusions, schema, and primary metrics from discovery; then annotate the precommitted `V*.txt` holdout files without opening the holdout key

## E02-qwen122b-holdout-template-freeze-v2

Experiment ID: E02-qwen122b-holdout-template-freeze-v2

Date: 2026-08-31

Question: Can blank holdout operation and target sheets be frozen from public metadata after discovery reveal, without reading any V text or decoding the holdout key?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (research layer remains uncommitted; working tree dirty)

Model: `qwen/qwen3.5-122b-a10b`; no model call; offline template freeze only

Backend: offline; `prepare_holdout_templates` reads public manifest and holdout filenames only

Prompt version: not applicable

N: 60 blank operation rows and 180 blank target rows; V text unread

Temperature / reasoning settings: not applicable

Random seed if applicable: not applicable

Output directory: operation `notes/annotations/qwen122b_holdout_operation_v2.csv`; targets `notes/annotations/qwen122b_holdout_targets_v2.csv`; freeze `notes/study_state/qwen122b_metadata_blind_v2/holdout_template_freeze.json`

Hash anchor: freeze schema `value-leakage.holdout-template-freeze/v1`; `holdout_text_bytes_read=false`; `private_packets_read=false`; discovery annotation lock SHA-256 unchanged `f77990d7e575acb752a3a25b04051bedef7b68de9e371a5e7d934b66799a25d0`

Main result: template freeze succeeded. Helper `next` returns V001 operation. Holdout key remains unopened.

Unexpected observation: the previous session had already written the holdout analysis plan, target-selection addendum, codebooks, and helper, and had produced a discovery analysis, but had not executed the exclusive-create template freeze; that was the stall point.

Interpretation: gate 4 is now executable and bound. Gate 5 is manual V001--V060 annotation. Final E03 eligibility still waits on locked exact spans and one later holdout reveal.

Next experiment: annotate V001 through V060 in fixed order using `holdout_annotation_helper`; do not open the holdout key

## E02-side-mechanics-negative-controls-v3

Experiment ID: E02-side-mechanics-negative-controls-v3

Date: 2026-08-31

Question: On shipped trajectory-judge sequences, is Qwen 3.5 122B revising toward the prompted good side, or toward the threshold? Do no-bet baseline, donation-label shuffle, and placebo thresholds behave as negative controls?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (research layer uncommitted; working tree dirty; `code_dirty=true`)

Model: all 10 shipped models; primary claims about `qwen/qwen3.5-122b-a10b`

Backend: offline shipped OpenRouter artifacts; no new sampling; holdout packet not opened

Prompt version: exact prompts embedded in shipped condition JSON

N: Qwen post-10x-filter n = 93 / 86 / 87 for baseline / below_good / above_good; permutation n=2000, seed=20260831

Temperature / reasoning settings: inherited from shipped artifacts; temperature not recorded; reasoning effort `high`; max tokens 64,000

Random seed if applicable: 20260831 for label-shuffle permutations only

Output directory: `figures/side_mechanics_v3/` (canonical). `figures/side_mechanics_v1/` and `v2/` are superseded copy/axis iterations of the same analysis.

Hash anchor: tool SHA-256 `8cda8cd445fa20c17e8415a0395ad19f547489af3808b899b0faa596e9ef09aa`; REPORT.md SHA-256 `9fcc2c4512b0de4bba7f63e61347a9d1efcba11d0a423577650c3e0fea45cc6f`; analysis.json SHA-256 `5f0b930af2d9c467797e4203d2f3fa52ff6cf960fc1f6c0c810b28d5f4de93e4`

Main result: Qwen directional revisions seek the threshold in both donation arms (P(toward|directional)=0.845 / 0.838) more than in no-bet baseline (0.620). First-side already differs modestly (delta_early=0.178, permutation p=0.021), but last-side divergence is larger (delta_last=0.550). P(up|above)-P(up|below) is null (p=0.726). Pooled condition-favored revision is chance (0.489 [0.408, 0.571]). Last-on-favored-side rises to 0.826 below_good and 0.724 above_good. Placebo gap-shrink peaks at true T in donation arms; baseline shrinks more toward smaller placebo T. Starter MRF remains 0.027.

Unexpected observation: equal-n P(favored|above)-P(favored|below) is not a label-association test, so the shuffle was applied to P(up) and to first-side instead. Human/judge first-side agreement is 30/32 scored discovery rows (4 missing after 10x filter). Explicit value statements in discovery are associated with *less* favored revision (3/17 vs 11/19); n=36, not confirmatory.

Interpretation: the mechanical pattern is threshold-seeking plus favored-side absorption, not a constant good-direction push. Baseline and placebo show that 41M in the donation prompt does work that generic Fermi shrinking does not. This is still not a causal mediator test and does not replace E01 or E03.

Next experiment: E01 still needs an API key; E03 still needs locked holdout spans. Holdout annotation remains the confirmatory path. Do not mix AI labels into the locked human sheets.

## E02-absorption-visible-finals-v2

Experiment ID: E02-absorption-visible-finals-v2

Date: 2026-08-31

Question: Is last-side absorption terminal conversion (end on the good side, favored starts do not leak), or first-hit stopping? Does the committed visible answer, parsed without a new judge, show the same split?

Code commit: `16d129859e1f0e281363fb4f5910bcaeea316b10` (research layer uncommitted; working tree dirty)

Model: `qwen/qwen3.5-122b-a10b`; no new sampling

Backend: offline shipped artifacts; holdout unread

N: trajectory n = 86 / 87 donation arms after 10x filter; visible parser 90/100 and 95/100 first-line successes; permutation n=2000, seed=20260831

Output directory: `figures/absorption_v2/`

Main result: mechanical Delta_pivot = 0.051, inside ROPE [-0.10, +0.10], permutation p=0.445. Convert vs leak: 0.667 vs 0.023 in below_good; 0.276 vs 0.052 in above_good. Escape-after-first-favored-hit is 0.675 / 0.738, so first-hit stopping fails. Visible P(>T) = 0.200 vs 0.702 (delta 0.502). Baseline parser vs shipped estimate judge is 95/95 exact.

Unexpected observation: traces often leave the good side after first touching it, then the committed end is still on that side. Absorption is about the endpoint, not a freeze at first contact.

Interpretation: H-push remains rejected; H-absorb is refined to terminal conversion. H-salience is still open. E01 remains the discriminating next sample.

Next experiment: small-n E01 if a key appears; do not add more plots of the same traces.
