# Qwen 122B metadata-blinded study state — deprecated v1

Do not continue annotation or reveal from this v1 state. A security review
found that v1 placed plaintext reveal packets inside the regenerable public
bundle and would expose holdout conditions before holdout annotation. It is
retained only as an audit trail and has been superseded by v2.

This directory preserves the non-regenerable seed state for the discovery and
holdout allocation. The actual JSON keys under `sealed/` are locally ignored so
`git add -A` cannot accidentally reveal them before the appropriate gate.

Do not run the legacy aggregate auditor: it parsed both mappings even when it
printed only aggregate checks. The v2 public auditor treats private packets as
unavailable and verifies only public commitments before the relevant gate.

Historical workflow gates below are superseded and must not be executed:

1. Fill all 36 rows in `notes/annotations/qwen122b_discovery_v1.csv`.
2. Lock/hash the completed discovery annotation.
3. Open/reveal only the discovery key.
4. Freeze formal hypotheses, primary metrics, schema, and analysis plan.
5. Only then use the independent holdout key to materialize the holdout.

The regenerable reading bundle is at
`derived/E02_trajectory_localization/qwen122b_metadata_blind_v1/`.
`sealed_backup_manifest.json` contains the expected key and mapping-commitment
hashes without exposing either mapping.
