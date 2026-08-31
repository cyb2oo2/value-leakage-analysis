# Experiments

Each experiment directory owns its question, prompts/config, code or entrypoint,
and a pointer to raw and derived outputs. Experiment IDs are stable; reruns get a
new output directory rather than overwriting an older result.

- `E01_neutral_threshold/`: separates threshold salience/anchoring from outcome
  valence. Only mock/small-pilot scaffolding is allowed until explicitly enabled.
- `E02_trajectory_localization/`: read-only shipped-data analysis and manual
  component annotation.
- `E03_sentence_resampling/`: future prefix/replacement/continuation
  intervention scaffold; no assumption of hidden-CoT continuation support.

Before any run, copy the template in `notes/experiment_log.md`, fill exact model,
provider, prompt version, N, sampling settings, seed, output directory, judge,
and current commit, then preserve raw responses outside ignored scratch storage.
