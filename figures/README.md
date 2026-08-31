# Generated figure bundles

Each subdirectory is a self-contained, reproducible bundle. A trajectory bundle
contains `analysis.json`, `analysis_config.json`, `provenance.json`, and its PNG
figures. `provenance.json` records the analysis script hash, source commit,
settings, and SHA-256 for source artifacts.

The generator refuses to overwrite an existing bundle. Reproduce into a new
directory, compare semantic JSON values and decoded pixels, then decide whether
the old canonical bundle should be replaced in a separate reviewed change.
