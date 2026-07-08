# park_drag_delta 2026 backtest screen — POWERED NULL (both folds)

**Decision-relevant summary:** adding park ball-drag regime state to
production FEATURE_COLS adds **no within-day ranking value** on 2026 at the
harness's ~+0.003 detection floor — in the post-selection window that
motivated the feature, and in a second fold with the regime fully inside
training. The same-recency rolling-OUTCOME park factor is equally null, so
this is not "drag vs outcomes"; park environment state of either kind does
not move candidate ranking at this model's scale. Do NOT promote
park_drag_delta to FEATURE_COLS on alpha grounds; it remains a shadow/
monitoring artifact whose value proposition is regime observability
(the frozen expanding park_factor cannot see mid-season ball changes),
not measurable pick quality.

## Design

- Harness: `bts/experiment/park_drag_screen.py` mirroring the validated
  swing screen (slate = actual starters, p_game = 1-(1-p_pa)^PA_EST,
  daily NDCG@10 / per-day rank-AUC / top1/top3; availability flags in EVERY
  arm; within-date venue-BLOCK permutation control; gross + calibrated soft
  oracle sentinels; M3-style same-day-leak canary). Driver
  `scripts/park_drag_screen_driver.py`, report
  `scripts/park_drag_screen_report.py`. 5 seeds (42/101/202/303/404),
  LightGBM deterministic, train 2019-2025 + 2026-to-split.
- Fold A: train extra through 2026-04-30, screen 2026-05-01..07-06
  (includes the May 24 league change-point; post-window broken out).
- Fold B (closes the OOD objection — training never saw regime-scale deltas
  in fold A): train extra through 2026-05-31, screen 2026-06-01..07-06.
- POST-SELECTION caveat: 2026 motivated the feature → supporting evidence
  by design (spec); values are physics-derived (labels untouched).

## Controls gate (passed, both folds)

| control | fold A Δday-pair-AUC | fold B | read |
|---|---|---|---|
| gross oracle | 1.0000 | 1.0000 | harness sees leaks |
| soft oracle (calibrated leak) | +0.0034 ±0.0002 | +0.0040 ±0.0012 | power: ~+0.003 resolvable |
| venue-block permuted | −0.0029 ±0.0011 | −0.0024 ±0.0008 | null band |
| mask-only | −0.0005 ±0.0008 | +0.0007 ±0.0008 | null band |
| same-day-leak canary | −0.0006 | n/a | too weak at game granularity — same lesson as the swing same-day sentinel; gross+soft carry leak detection |

## Arms (Δ vs baseline, day-pair-weighted AUC, mean ±sd over 5 seeds)

| arm | fold A full | fold A post-5/24 | fold B (June+) |
|---|---|---|---|
| pd_anchored (shipped shape) | −0.0003 ±0.0003 | **+0.0004 ±0.0005** | **−0.0001 ±0.0006** |
| pd_expanding (v1 shape) | −0.0018 | −0.0024 | −0.0037 |
| outcome_pf (rolling outcomes competitor) | −0.0009 | −0.0008 | −0.0006 |
| pd_plus_outcome | −0.0010 | −0.0003 | −0.0010 |

top1/top3 deltas are noise at n≈40-67 days (±~0.07); none coherent across folds.

## Interpretation

- Consistent with the physics arithmetic from the source analysis
  (~+0.4-0.6pp P(≥1 hit) at the MOST extreme park split, BABIP flat): the
  within-day ranking impact of a whole-game environment shift is far below
  this model's discrimination floor. The screen adds: no hidden
  amplification, and no OOD excuse (fold B).
- pd_expanding is mildly HARMFUL (at/below the permuted band in both folds) —
  empirical confirmation of the design-review critique of the decaying shape.
- The live shadow (same model class) should be expected to read ~neutral on
  pick quality; its remaining value is regime OBSERVABILITY + the park_drag
  table/health infrastructure itself. Promotion to FEATURE_COLS is not
  supported by this evidence and should not be argued from the shadow alone.

## Reproduce

    uv run python scripts/park_drag_screen_driver.py --out data/validation/park_drag_screen_2026
    BTS_PD_TRAIN_EXTRA=2026-05-31 BTS_PD_SCREEN_START=2026-06-01 \
      uv run python scripts/park_drag_screen_driver.py --out data/validation/park_drag_screen_2026_foldB
    uv run python scripts/park_drag_screen_report.py --dir <out dir>

Result JSONs (per arm × seed, with per-day payloads) are untracked at those
paths; needs data/processed/pa_{2019..2026}.parquet (2026 through ≥Jul 6) and
data/external/park_drag/park_drag_export.csv.
