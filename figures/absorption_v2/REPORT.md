# Absorption, visible finals, and mechanical Delta_pivot

Follow-up to the side-mechanics negative controls. No new sampling, no holdout text.

## Why this experiment

H-absorb says last-side splits because traces **convert onto the good side of T and then stay**. Symmetric seeking predicts similar crossing given start side, and would not require conversion to exceed leakage. Mechanical Delta_pivot is the frozen holdout estimand run on the judge sequence (descriptive, not confirmatory). Visible finals test whether the split is in the committed answer, without substituting the trajectory endpoint.

## Mechanical Delta_pivot (equal 0.5 weights, frozen in the holdout plan)

- P(up | above, first below) = 0.708 [0.508, 0.851]
- P(up | below, first below) = 0.697 [0.527, 0.826]
- P(up | above, first above) = 0.091 [0.036, 0.212]
- P(up | below, first above) = 0.000 [0.000, 0.094]
- Delta_pivot = 0.051 (ROPE [-0.10, +0.10]; inside_rope=True)
- Permutation p = 0.445 (n_perm=2000)

If Delta_pivot sits in the ROPE, condition-linked revision after start-side stratification is small. That is the H-anchor/H-seek prediction, not H-push.

## Conversion versus leakage

- `below_good` n=86: convert|start opposed = 0.667 [0.516, 0.790]; leak|start favored = 0.023 [0.004, 0.118]; escape after first favored hit = 0.675 [0.566, 0.768]
  first×last matrix: {'below': {'below': 43, 'equal': 0, 'above': 0}, 'equal': {'below': 0, 'equal': 0, 'above': 1}, 'above': {'below': 28, 'equal': 0, 'above': 14}}
- `above_good` n=87: convert|start opposed = 0.276 [0.147, 0.457]; leak|start favored = 0.052 [0.018, 0.141]; escape after first favored hit = 0.738 [0.632, 0.821]
  first×last matrix: {'below': {'below': 21, 'equal': 0, 'above': 8}, 'equal': {'below': 0, 'equal': 0, 'above': 0}, 'above': {'below': 3, 'equal': 0, 'above': 55}}

Terminal conversion exceeds leakage in both arms, especially below_good (0.67 vs 0.02). That is the last-side absorption signature. First-hit stopping is a different claim: escape-after-hit is high (~0.7), so traces keep moving after they first touch the good side. H-absorb should be read as **terminal** absorption (the committed end lands on the good side and favored starts rarely leak), not as **stop-on-first-hit**.

## Visible finals (fail-closed first-line parser)

- below_good P(visible > T) = 0.200 [0.130, 0.294] (parsed 90/100)
- above_good P(visible > T) = 0.702 [0.603, 0.785] (parsed 95/100)
- Delta_visible last = 0.502
- baseline parser vs shipped estimate judge: {'n': 95, 'exact': {'label': 'exact', 'k': 95, 'n': 95, 'p': 1.0, 'wilson95': {'k': 95, 'n': 95, 'p': 1.0, 'ci': [0.9611351464605291, 1.0]}}, 'rel_median_abs_err': 0.0}

Donation-arm estimates.json is missing by construction of the starter pipeline. This parser is not a Claude judge. UNKNOWN lines are dropped, not filled from the trajectory.

## What this still does not show

- It does not replace E01. Visible-final absorption can still be salience plus stopping, not value.
- Mechanical Delta_pivot is not the holdout confirmation. The holdout uses human first-target totals.
- A first-line parser can miss wrapped answers; parse-status counts are the coverage report.

