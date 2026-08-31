# E02 — trajectory localization

Current scope is transparent trajectory-event analysis plus human annotation.
Normalized trajectory position is position in the judge-extracted estimate
sequence; it is not token time, sentence time, or a semantic reasoning stage.

The qualitative workflow is governed by
[`KNOWN_UNKNOWN.md`](KNOWN_UNKNOWN.md),
[`hypothesis_map_v0.1.md`](hypothesis_map_v0.1.md), and
[`DISCOVERY_PROTOCOL.md`](DISCOVERY_PROTOCOL.md). The current formal reading
set is a metadata-blinded v2 bundle: private mappings are physically outside
the public bundle, and holdout texts are precommitted without exposing their
condition mapping.

Do not run `reveal_discovery.py` until every discovery annotation row is
complete and `lock_discovery_annotation.py` has succeeded. The reveal command
requires an explicit confirmation flag and never parses the holdout key.

Active offline artifacts:

- public reading bundle:
  `derived/E02_trajectory_localization/qwen122b_metadata_blind_v2/`;
- discovery annotation:
  `notes/annotations/qwen122b_discovery_v2.csv`;
- frozen codebook: `annotation_codebook_v0.1.md`;
- hash-only anchor:
  `notes/study_state/qwen122b_metadata_blind_v2/private_manifest.json`.

Safe public audit (does not locate or parse either private key):

```text
uv run python -m experiments.E02_trajectory_localization.audit_blind_bundle --bundle derived/E02_trajectory_localization/qwen122b_metadata_blind_v2 --annotation notes/annotations/qwen122b_discovery_v2.csv --runs-root runs
```

After all 36 rows are complete, create the lock. This command is expected to
fail while any required field is blank:

```text
uv run python -m experiments.E02_trajectory_localization.lock_discovery_annotation --bundle derived/E02_trajectory_localization/qwen122b_metadata_blind_v2 --annotation notes/annotations/qwen122b_discovery_v2.csv --codebook experiments/E02_trajectory_localization/annotation_codebook_v0.1.md --hash-anchor notes/study_state/qwen122b_metadata_blind_v2/private_manifest.json --private-discovery-key notes/study_state/qwen122b_metadata_blind_v2/sealed/discovery_reveal_key.json --private-holdout-key notes/study_state/qwen122b_metadata_blind_v2/sealed/HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json --output notes/study_state/qwen122b_metadata_blind_v2/discovery_annotation_lock.json --runs-root runs --repo-root .
```

Discovery reveal is intentionally documented only as the next gated command;
do not execute it before the lock exists:

```text
uv run python -m experiments.E02_trajectory_localization.reveal_discovery --bundle derived/E02_trajectory_localization/qwen122b_metadata_blind_v2 --annotation notes/annotations/qwen122b_discovery_v2.csv --codebook experiments/E02_trajectory_localization/annotation_codebook_v0.1.md --hash-anchor notes/study_state/qwen122b_metadata_blind_v2/private_manifest.json --annotation-lock notes/study_state/qwen122b_metadata_blind_v2/discovery_annotation_lock.json --private-discovery-key notes/study_state/qwen122b_metadata_blind_v2/sealed/discovery_reveal_key.json --private-holdout-key notes/study_state/qwen122b_metadata_blind_v2/sealed/HOLDOUT_DO_NOT_OPEN_UNTIL_ANNOTATIONS_LOCKED.json --output-dir derived/E02_trajectory_localization/qwen122b_discovery_reveal_v2 --runs-root runs --repo-root . --confirm-discovery-annotations-locked
```

Generate a new bundle from the shipped Qwen run:

```text
uv run python -m research.trajectory_analysis --config experiments/E02_trajectory_localization/configs/shipped_qwen.json --output-dir figures/E02_shipped_qwen_rerun
```

After discovery lock and reveal, freeze blank holdout sheets from public metadata
only. This command must not open any `V*.txt` bytes or a private packet:

```text
uv run python -m experiments.E02_trajectory_localization.prepare_holdout_templates --bundle derived/E02_trajectory_localization/qwen122b_metadata_blind_v2 --operation-output notes/annotations/qwen122b_holdout_operation_v2.csv --target-output notes/annotations/qwen122b_holdout_targets_v2.csv --template-manifest-output notes/study_state/qwen122b_metadata_blind_v2/holdout_template_freeze.json --operation-codebook experiments/E02_trajectory_localization/holdout_operation_codebook_v0.1.md --target-codebook experiments/E02_trajectory_localization/target_adjudication_codebook_v0.1.md --frozen-document experiments/E02_trajectory_localization/HOLDOUT_ANALYSIS_PLAN_v0.1.md --frozen-document experiments/E03_sentence_resampling/TARGET_SELECTION_ADDENDUM_v0.1.md --frozen-document experiments/E03_sentence_resampling/POST_DISCOVERY_STATUS_v0.1.md --frozen-document experiments/E03_sentence_resampling/FORMAL_HYPOTHESES_v0.1.md --frozen-document experiments/E03_sentence_resampling/PREREGISTRATION_DRAFT_v0.1.md --runs-root runs
```

Manual holdout annotation uses one explicit V ID at a time and never returns the
visible-final-answer block:

```text
uv run python -m experiments.E02_trajectory_localization.holdout_annotation_helper next --operation-csv notes/annotations/qwen122b_holdout_operation_v2.csv --target-csv notes/annotations/qwen122b_holdout_targets_v2.csv
uv run python -m experiments.E02_trajectory_localization.holdout_annotation_helper show-reasoning --bundle derived/E02_trajectory_localization/qwen122b_metadata_blind_v2 --blind-id V001
```

Do not lock or reveal holdout conditions until every V operation row and every
target row is complete.

The committed canonical bundle is `figures/E02_shipped_qwen_seed20260825/`.
Use a different `--output-dir` for a rerun because outputs are never overwritten.
