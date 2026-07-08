# park_drag_delta 2026 backtest screen — NULL (v2, post Codex review)

*(v2 2026-07-08: Codex gpt-5.5 adversarially reviewed the harness, the numbers
— which it independently reproduced from the result JSONs — and this writeup;
six findings folded in. The NULL VERDICT STANDS; several claims are reworded
to what the evidence actually supports. Review notes at bottom.)*

**Decision-relevant summary:** in this 2026 ranking screen, adding park
ball-drag regime state to production FEATURE_COLS produced **no detectable
within-day ranking gain**, in either fold, against a soft-oracle positive
control that demonstrably registers a +0.003-0.004 label-leak at this
screen's scale. The same-recency rolling-OUTCOME park factor was equally
null. **Do not promote `park_drag_delta` to FEATURE_COLS on this evidence.**
The feature's standing value is regime observability (the frozen expanding
`park_factor` cannot see mid-season ball changes) plus the monitoring
infrastructure; the live shadow remains the pick-quality readout (this
screen sets a near-zero prior for ranking lift, but agreement/hit-rate/
pick-quality come from the shadow, not from here).

## Design

- Harness: `bts/experiment/park_drag_screen.py` mirroring the validated swing
  screen. Slate = **actual batter-game participants with lineup_position 1-9
  (includes substitutions — ~20.4 batters/game, not just starters)**;
  p_game = 1-(1-p_pa)^PA_EST with starter-slot PA estimates. Metrics:
  **day-pair-weighted within-day rank AUC** (primary; the report recomputes it
  from per-day payloads — distinct from the stored global AUC), NDCG@10,
  top1/top3. Availability flags in EVERY arm; venue-BLOCK within-date
  permutation; gross + hash-revealed soft-oracle sentinels. 5 seeds,
  deterministic LightGBM, train 2019-2025 + 2026-to-split.
- Fold A: train extra through 2026-04-30, screen 2026-05-01..07-06 (67 days;
  post-May-24 window broken out).
- Fold B: train extra through 2026-05-31, screen 2026-06-01..07-06. **Not an
  independent replication** — its 36 eval days are a subset of fold A's; it
  answers "does adding May 2026 (regime-scale deltas + outcomes) to training
  change the June+ read?" (it does not). "Regime in training" is partial:
  the rollout continued into June (CBP/Daikin switched ~Jun 6).
- POST-SELECTION caveat: 2026 motivated the feature → supporting evidence by
  design (spec); feature values are physics-derived (labels untouched).

## Controls gate (passed, both folds; paired per-seed deltas)

| control | fold A Δ | fold B Δ | read |
|---|---|---|---|
| gross oracle (label leak) | +0.4107 (raw 1.0000) | +0.4051 | saturates — harness sees leaks |
| soft oracle (hash-revealed label leak) | +0.0034 ±0.0007 | +0.0040 ±0.0015 | positive control at the +0.003-0.004 scale |
| venue-block permuted | −0.0029 ±0.0012 | −0.0024 | null band |
| mask-only | −0.0005 ±0.0011 | +0.0007 | null band |
| date+1 canary | −0.0006 | (not run) | measured-environment value, NOT a label leak — its null is **inconclusive** about leak visibility; gross+soft carry that evidence |

Note the soft oracle proves detection of *this specific label leak* at that
scale; it is not a universal detection floor for a collinear venue-level
physics feature. "Powered null" (v1 wording) overstated this — the accurate
claim is: the candidate deltas are an order of magnitude below the
positive-control scale and within the seed-noise band.

## Arms (paired per-seed Δ vs baseline, day-pair-weighted AUC, mean ±sd, n=5)

| arm | fold A full | fold A post-5/24 | fold B (June+) |
|---|---|---|---|
| pd_anchored (shipped shape) | −0.0003 ±0.0005 | **+0.0004 ±0.0004** | **−0.0001 ±0.0009** |
| pd_expanding (v1 shape) | −0.0018 | −0.0024 | −0.0037 |
| outcome_pf (rolling outcomes) | −0.0009 | −0.0008 | −0.0006 |
| pd_plus_outcome | −0.0010 | −0.0003 | −0.0010 |

top1/top3 deltas are noise at n≈36-67 days; none coherent across folds.
Codex independently recomputed all deltas from the JSONs and matched.

## Interpretation

- Consistent with the physics arithmetic from the source analysis
  (~+0.4-0.6pp P(≥1 hit) at the MOST extreme park split, BABIP flat): a
  whole-game environment shift of that size is far below this screen's
  demonstrated detection scale for within-day ranking.
- Neither `park_drag_delta` nor the rolling-15 outcome park factor moved
  THIS screen's ranking metric — so this is not "drag lost to outcomes."
- `pd_expanding` is mildly harmful in both folds (at/below the permuted
  band) — empirical support for the design-review critique of the decaying
  shape.
- Guidance: **do not promote on this evidence; keep the shadow/observability
  value; let live-forward results carry the pick-quality question.**

## Codex review disposition (v1 → v2)

#1 slate is participants-not-starters (inherited from swing_screen, whose
docstring says "actual starters" — same property in prior screens) → reworded,
noted here; #2 "powered null" overstated + report ± was arm-score sd not
paired-delta sd → report fixed (`paired_delta`), claims reworded; #3 fold B
not independent → reframed as the add-May-to-training probe; #4 (Codex
disagreement, accepted) the date+1 canary null was rationalized as a finding —
it is inconclusive by construction; #5 gross-row units mixed raw/delta +
fold B lacks the leaky arm → table fixed, noted; #6 metric naming
(day-pair-weighted vs stored global AUC) → named precisely throughout.
Checks that passed review: baseline does not carry the candidate; train/screen
dates disjoint; export as-of semantics hold; the per-arm `notna().any` train
mask was verified identical across arms (all 1,184,069 training rows pass).

## Reproduce

    uv run python scripts/park_drag_screen_driver.py --out data/validation/park_drag_screen_2026
    BTS_PD_TRAIN_EXTRA=2026-05-31 BTS_PD_SCREEN_START=2026-06-01 \
      uv run python scripts/park_drag_screen_driver.py --out data/validation/park_drag_screen_2026_foldB
    uv run python scripts/park_drag_screen_report.py --dir <out dir>

Result JSONs (per arm × seed, per-day payloads) untracked at those paths;
needs data/processed/pa_{2019..2026}.parquet (2026 through ≥Jul 6) and
data/external/park_drag/park_drag_export.csv.
