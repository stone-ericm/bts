# Gate B Raw Re-bin Measurement (2026-05-23)

**Status:** measurement-only. No production behavior change.

## Question

Gate B from `2026-05-23-calibration-resolve-gate.md` asks whether the MDP
policy should be re-binned and re-solved on the current raw probability
distribution, independent of any probability-calibration map.

This is the policy-scale counterpart to Gate A:

- Gate A asks whether to transform probabilities.
- Gate B asks whether the policy table should use boundaries that discriminate
  the current raw probability range.

## Current production point measurement

Read-only live production measurement, ending 2026-05-23:

- Source rows: resolved production days with both primary and double-down slots.
- `n=49` resolved pair rows, from 2026-04-02 through 2026-05-22.
- Deployed policy boundaries: `[0.795979, 0.811491, 0.825247, 0.840740]`.
- All candidate raw-bin representatives map to deployed policy Q0 for `n_bins`
  in `{2, 3, 4, 5}`.

Point results under the empirical current raw-bin manifold:

| n_bins | per-bin support | projected deployed baseline P(57) | raw re-solve P(57) | gap |
|---:|---:|---:|---:|---:|
| 2 | 24-25 | effectively 0 | `0.000003` | `+0.000003` |
| 3 | 16-17 | effectively 0 | effectively 0 | `+0.000000` |
| 4 | 12-13 | effectively 0 | `0.000010` | `+0.000010` |
| 5 | 9-10 | effectively 0 | `0.000008` | `+0.000008` |

The point gaps are too small to matter and the per-bin support is far below a
deployment-grade policy-estimation threshold.

## Backtest distribution mismatch

The full multi-season policy-file harness is the real Gate B evaluator, but it
has a probability-distribution compatibility question.

Current production primary probabilities in the point measurement:

- Range: `0.690-0.792`.
- All recent production picks sit below the deployed policy's lowest boundary.

Local 2021-2025 backtest rank-1 probabilities from `data/simulation`:

- Range: `0.732903-0.918326`.
- Median: `0.816654`.
- 20th percentile: `0.794352`.

That means the current production maximum (`0.792`) is below the historical
backtest 20th percentile (`0.794352`). A historical policy-file backtest may
therefore evaluate a candidate on a probability distribution that does not
match the current production collapse. This must be verified before treating a
backtest result as a clean Gate B answer.

## Decision now

Do not swap `data/models/mdp_policy.npz`.

Do not ship a raw re-bin/re-solve policy.

Current-era resolved pair data is too thin, the point P(57) gaps are negligible,
and the multi-season harness may not match the current production probability
range.

## Revisit gate

A future Gate B attempt should require all of the following before any policy
artifact is considered:

1. Current-era resolved pair support: at least `n >= 200` primary/DD pair rows.
2. Per-bin support: at least `min_per_bin >= 30` for the candidate bin count.
3. Distribution compatibility: the multi-season policy-file harness must be
   shown to evaluate probabilities in the same range as the candidate raw
   bins, or the mismatch must be explicitly corrected by a pre-registered
   transformation.
4. P(57) result: the candidate policy must meet or beat the deployed baseline
   in the off-host multi-season policy-file harness.
5. Operational safety: heavy bootstrap, multi-season backtests, or repeated MDP
   solves must run off the live production host.

The new `scripts/measure_raw_rebin_gate.py` implements the lightweight support
and point-measurement screen. It is intentionally not the full deployment gate.

## Operational note

An attempted 500-replicate bootstrap on the live host was stopped after it was
clearly too heavy for an interactive production check. Scheduler and dashboard
remained healthy afterward (`NRestarts=0`, dashboard health `200`), but future
Gate B compute should run off-host by default.
