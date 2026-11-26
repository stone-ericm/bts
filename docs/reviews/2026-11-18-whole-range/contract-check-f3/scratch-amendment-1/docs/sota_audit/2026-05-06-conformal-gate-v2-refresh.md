# Binary-Y Conformal Gate V2 Refresh

## Verdict

The old binary-y conformal validation anti-pattern is already addressed in code: `bts validate conformal-gate` uses the v2 lower-bound validity gate instead of per-row `(actual_hit >= bound).mean()` coverage.

A fresh real-data run on the current canonical profile surface produced no shippable conformal lower-bound cell:

- Schema: `conformal_validation_v2`
- Verdict: `NO_PRODUCTION_DEPLOY`
- Ship set: `[]`
- Methods: `bucket_wilson`, `weighted_mondrian_conformal`
- Alphas: `0.05`, `0.10`, `0.20`
- Lockbox: 2025-08-30 through 2025-09-28

This confirms #11 is infrastructure-shipped but not production-cleared.

The distinction matters: marginal conformal coverage can be mathematically valid while still uninformative for binary outcomes. The v2 gate evaluates group-conditional lower-bound validity on probability buckets and separates that from tightness. It does not claim row-level coverage.

## Artifacts

- Split manifest: `data/validation/split_manifest_conformal_2026-05-06.json`
- Gate output: `data/validation/conformal_gate_v2_2026-05-06.json`

Parked conformal-v1 artifacts currently present locally:

- `data/conformal/calibrator_2026-05-01.pkl`: weighted Mondrian conformal calibrator artifact
- `data/conformal/wilson_calibrator_2026-05-01.pkl`: bucket-Wilson lower-bound calibrator artifact
- `data/conformal/lr_classifier_2026-05-01.pkl`: LightGBM density-ratio classifier for covariate-shift weighting
- `data/conformal/validation_log.jsonl`: fit-time diagnostics; latest row has `n_calibration=2190`, effective weighted n `4854.387`, 9 populated conformal buckets, and 7 populated Wilson buckets

## Result Matrix

| Method | Alpha | Verdict | Dominant failure mode |
| --- | ---: | --- | --- |
| `bucket_wilson` | `0.05` | `FAIL` | validity failures in populated high-probability buckets |
| `bucket_wilson` | `0.10` | `FAIL` | validity failures in populated high-probability buckets |
| `bucket_wilson` | `0.20` | `FAIL` | validity failures in populated high-probability buckets |
| `weighted_mondrian_conformal` | `0.05` | `FAIL` | tightness median width around `0.78`, above `0.30` threshold |
| `weighted_mondrian_conformal` | `0.10` | `FAIL` | tightness median width around `0.78`, above `0.30` threshold |
| `weighted_mondrian_conformal` | `0.20` | `FAIL` | validity failures in populated high-probability buckets |

## Interpretation

- The v2 gate is the right shape for binary outcomes: bucket-level observed-rate Wilson lower bounds are compared against the claimed lower-bound level, with tightness checked separately.
- The current canonical surface does not clear that gate for either method family.
- The failure is useful: conformal lower-bound infrastructure should remain off for production until a non-empty `ship_set` appears under the v2 gate.
- The historical `data/validation/conformal_validation_2026-05-01.json` artifact uses the older decision-matrix schema. The new artifact is the durable current v2 result.
- #12 assets are already the foundation for the next conformal work: `src/bts/validate/proper_scoring.py` provides Brier/log-loss, reliability tables with Wilson bands, Murphy decomposition, and top-bin calibration. Those should be diagnostics or gates for any future conformal-v1 deployment decision, rather than reintroducing per-row binary coverage.

## Next Work

This slice does not fix conformal-v1. The next implementation slice should define the production gate contract at the metric level:

- keep v2 group-conditional lower-bound validity by probability bucket
- keep tightness as a separate sharpness gate
- add or require #12 calibration diagnostics on selectable rows, especially rank-1 / decision-bucket reliability and Brier decomposition
- consider Venn-Abers / PAV / beta calibration as binary-calibration baselines before enabling a conformal lower-bound path in production

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts validate split-manifest \
  --profiles-dir data/simulation \
  --output data/validation/split_manifest_conformal_2026-05-06.json

UV_CACHE_DIR=/tmp/uv-cache uv run bts validate conformal-gate \
  --profiles-dir data/simulation \
  --manifest data/validation/split_manifest_conformal_2026-05-06.json \
  --output data/validation/conformal_gate_v2_2026-05-06.json

UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/validate/test_conformal_gate.py \
  tests/validate/test_proper_scoring.py \
  tests/model/test_conformal.py
```
