# Qwen 122B metadata-blinded study state v2

This is the active state directory. Plaintext private keys live only under the
gitignored `sealed/` subdirectory and are not part of the public reading
bundle. Never open them manually.

Workflow gates:

1. Fill every row in `notes/annotations/qwen122b_discovery_v2.csv`.
2. Run the fail-closed discovery annotation lock.
3. Run discovery-only reveal with explicit confirmation.
4. Freeze formal hypotheses, exclusions, schema, and primary metrics.
   Current freeze: `holdout_template_freeze.json`, plus the holdout analysis
   plan and target-selection addendum bound by that manifest.
5. Read and annotate the precommitted `V*.txt` files without opening the
   holdout key. Current blank sheets:
   `notes/annotations/qwen122b_holdout_operation_v2.csv` and
   `notes/annotations/qwen122b_holdout_targets_v2.csv`.
6. Lock all holdout annotations.
7. Only then reveal holdout conditions.

The active public bundle is
`derived/E02_trajectory_localization/qwen122b_metadata_blind_v2/`. The tracked
private manifest records hashes and commitments, never seeds or mappings.
