# Manual trajectory-localization annotation

Generate a CSV or JSON scaffold with `research.inspect_rollouts --annotation`.
The tool pre-fills only identifiers and mechanically available estimate fields.
It does not label motivated reasoning or infer mental states.

Fields:

- `rollout_id`, `model`, `condition`, `threshold`
- `first_estimate`, `trajectory_last_estimate`
- `parsed_visible_final_estimate`, `parsed_visible_final_estimate_status`
- `crossed_threshold` (mechanical strict-side crossing; blank if unavailable)
- `explicit_bias_statement` (human annotation)
- `explicit_impartiality_statement` (human annotation)
- `reasoning_component_notes` (human notes using population/species/spots/
  aggregation/sanity-check/threshold/impartiality components as useful)
- `notes`

Keep blank distinct from `false`. A blank human field means not yet annotated;
`false` means a reader checked the rollout and did not observe the feature.
