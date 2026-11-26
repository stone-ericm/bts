# Audit-Verdict FDR Retrospective

- **Generated**: 2026-05-06
- **Artifact**: `data/validation/audit_verdict_fdr_2026-05-06.json`
- **Generator**: `scripts/run_audit_verdict_fdr.py`
- **Input family**: `experiments/results/phase1/*/diff.json`

## Verdict

The retrospective Phase 1 audit-verdict family has no positive discoveries after the p-value FDR baseline.

The script found `m=24` testable Phase 1 `diff.json` artifacts with per-season P@1 deltas. It computed exact paired sign-flip permutation p-values over those season deltas, then applied BH and BY across the family. At `q=0.05`, `0` positive audit candidates survive BH and `0` survive BY.

## Results

| Metric | Value |
| --- | ---: |
| Testable family size | `24` |
| Positive candidates surviving BH q<=0.05 | `0` |
| Positive candidates surviving BY q<=0.05 | `0` |
| Smallest two-sided p-value | `0.5000` |
| Smallest BH q-value | `1.0000` |
| Smallest BY q-value | `1.0000` |

The smallest p-value belongs to negative/failed candidates under this coarse two-season sign-flip test; the artifact's first row is `catboost` with mean P@1 delta `-0.010825`, `p_two_sided=0.5000`, and `q_BH=1.0000`.

## Interpretation

This is an honest p-value/randomization baseline, not e-BH or online FDR. It closes the immediate "run an audit-level FDR retrospective" action, but it does not close the full #7 SOTA target because no valid e-values or e-processes are constructed.

The test is intentionally conservative and coarse. Most Phase 1 artifacts only provide two season-level deltas, so exact sign-flip p-values cannot get smaller than `0.5` for a two-sided all-same-sign result. That coarseness is the point: the saved Phase 1 verdict artifacts do not contain enough independent paired evidence to support discovery claims after multiple-testing correction.

## Next Work

1. Use this artifact as the historical truth-up layer for Phase 1 verdicts: no positive candidate should be described as surviving audit-level FDR.
2. If future audit cycles need online/sequential control, design valid e-values or e-processes before running the cycle; do not retrofit `1/p` as an e-value.
3. For new candidate stacks, prefer the explicit selection/outer-evaluation split and record the tested family before looking at outer-evaluation outcomes.
