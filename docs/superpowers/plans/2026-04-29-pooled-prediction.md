# Pooled-Prediction Production Architecture (item #6)

**Status**: scoping; not yet implemented.
**Created**: 2026-04-29
**Decision driver**: F's calibration finding (n=57 picks, +14pp overconfidence in 75-80% bucket; +7pp overall). Production seed=42 is at the 95th pctile of the n=100 baseline P(57) distribution, producing systematically optimistic predictions.
**Verdict**: ship daily multi-seed pooled prediction (option 4 from item #6 options table) in two phases:
- **Phase 1 (Vultr-only, ~3 weeks bridge)**: Hetzner account is <1mo old, so the limit-increase ticket can't be filed yet. Run on Vultr in Frankfurt (`voc-c-16c-32gb-300s-amd`, 10 parallel boxes, ~$75/mo).
- **Phase 2 (Hetzner-only, ongoing)**: when account turns 1mo, file Hetzner ticket → 10 CPX62 in fsn1 (~$95/yr). One config-flag change to migrate; predictions byte-equivalent under `BTS_LGBM_DETERMINISTIC=1`.

Bridge cost: ~$60-90 over the 3-4 week Phase 1 window.

## Architecture overview

```
                                  ┌─────────────────────────────────────┐
                                  │ 02:00 ET cloud burst                │
                                  │ (10 fresh boxes, ~1h elapsed)       │
                                  │                                     │
                                  │ Phase 1: Vultr voc-c-16c-32gb in fra│
                                  │ Phase 2: Hetzner CPX62 in fsn1      │
                                  │                                     │
                                  │ each box: bts run --train-only      │
                                  │   --seed <N> --date <TOMORROW>      │
                                  │   --models-out data/models_pooled/  │
                                  │           seed_<N>/                 │
                                  └────────────┬────────────────────────┘
                                               │ rsync 12 .pkl files per seed
                                               ▼
                              data/models_pooled/                   (on bts-mlb)
                                seed_<N>/blend_<DATE>.pkl × 10 seeds
                                <DATE>_status.json
                                               │
                                               ▼
                              ┌──────────────────────────────────────┐
                              │ bts-mlb (always-on)                  │
                              │   bts run --date <TODAY>             │
                              │     loads data/models_pooled/ if     │
                              │     present, falls back to single    │
                              │     seed=42 from data/models/ if not │
                              │   averages p_game_hit across seeds   │
                              │   produces top-1 + DD pick           │
                              │                                      │
                              │   game_time-45min lineup re-checks   │
                              │     (same model state, re-rank only) │
                              └──────────────────────────────────────┘
```

## Key decisions (with reasoning)

### D1: Pool at the prediction level, not feature level
Each seed produces a 12-model blend → per-(game, batter) `p_game_hit` value. Pooling = mean of 10 seed-level `p_game_hit` per (game, batter), then rank by pooled value.

**Why not pool individual model predictions across seeds?** Hierarchical pooling (mean per (model, seed)) doesn't add information vs flat-mean of per-seed-blend outputs, since the within-blend ensemble already aggregates 12 models. Flat mean is simpler.

**Why not pool at the rank level (ensemble of rank-1 picks)?** Rank pooling discards the magnitude information. We'd lose the calibration win that's the whole point of this exercise.

### D2: 10 seeds, drawn from canonical-n10
Use the same canonical-n10 manifest we shipped today (`data/seed_sets/canonical-n10.json`). Stratified across the n=100 baseline distribution → an unbiased estimator of the underlying effect.

**Not seed=42**: the seed everyone "knows" is in this set (it's the one near the 95th pctile, position ~95). Including or excluding it doesn't change the pooled mean meaningfully but reveals an awkward asymmetry. Going with the canonical 10 unchanged.

**Why not n=20**: marginal calibration improvement vs doubled cost. n=10 gives stable means and matches our screening protocol.

### D3: Cloud-burst training, prod-host inference
Training runs on disposable cloud boxes once daily; predictions and lineup re-checks use cached models on bts-mlb. **Inference cost = 0 additional vs status quo.**

**Why not train on bts-mlb?** 10× single-seed training time = ~10-15h/day on a single 16-vCPU box. Conflicts with prod scheduler responsibilities (lineup checks, dashboard serving, evening cron).

**Provider sequence — Vultr Phase 1, Hetzner Phase 2.** Hetzner is the long-term home: $0.026/hr CPX62 in fsn1 vs Vultr's $0.247/hr voc-c-16c-32gb in fra (~10× cost difference). But Hetzner's 5-server account-default cap blocks 10-parallel until the account hits 1 month old (then a limit-increase ticket can be filed). Vultr quota is unconstrained on this account (101 instances + 3,512 instance-hours used this billing period), so we use it for the bridge window. The orchestrator's `--provider` flag flips between them with no data-format change.

**Why CPX62 (Phase 2) and voc-c-16c-32gb-300s-amd (Phase 1) specifically?** Both are 16 vCPU + 32GB. CPX62 is our determinism-validated Hetzner workhorse. The Vultr CPU-Optimized AMD plan was used in canonical-n10 audits, so its byte-equivalence vs Hetzner is already in our test record.

### D4: Determinism gate
All 10 boxes set `BTS_LGBM_DETERMINISTIC=1` (already validated cross-provider, atol=1e-10). Without this, byte-equivalence is lost and cross-seed averaging would conflate noise + signal.

### D5: Fall back to single-seed if pool incomplete
If <8 of 10 seeds successfully train by 08:00 ET (1h before pick generation deadline), prod uses the existing `data/models/blend_<DATE>.pkl` (seed=42) → degrades gracefully, picks still ship. New health check fires INFO/WARN/CRITICAL based on missing seed count.

### D6: Storage retention 5 days
Keep last 5 days of pooled models for rollback safety. ~5 GB steady-state on bts-mlb's 80 GB volume. Older seeds pruned by daily cleanup cron.

## Components

### Component A — Daily pooled-training orchestrator
**File**: `scripts/pooled_train_daily.py` (new, ~250 lines)
**Role**: provisions 10 fresh boxes (Phase 1: Vultr; Phase 2: Hetzner) in parallel, runs `bts run --train-only --seed <N>` on each, rsyncs models back to bts-mlb, tears down boxes, writes status JSON.
**Pattern**: extends `scripts/audit_driver.py`. New `--train-only` mode; no Phase 1 screening. Existing `--provider` flag (vultr|hetzner|oci) flips clouds with no data-format change.
**Phase 1 invocation**: `pooled_train_daily.py --provider vultr --plan voc-c-16c-32gb-300s-amd --region fra --seed-set canonical-n10`
**Phase 2 invocation** (post-Hetzner-ticket): `pooled_train_daily.py --provider hetzner --seed-set canonical-n10`
**Trigger**: bts-mlb `systemd --user` timer firing daily at 02:00 ET (cron `0 6 * * *` UTC equivalent). Self-contained — no GHA dependency.
**Failure mode**: if cron fails for the morning, fallback (D5) uses yesterday's seed=42 model from `data/models/`. Manual recovery: `systemctl --user start bts-pooled-train.service`.

### Component B — `bts run` pooled inference path
**File**: `src/bts/model/predict.py` (modify, ~50 lines)
**Change**: new `load_pooled_models(date, models_dir, n_seeds=10) -> list[Blend]` and `predict_pooled(...)`. Falls back to single-seed `load_models()` if pool missing/incomplete. Pools at p_game_hit level (D1).
**Flag**: `--pooled` CLI arg + env var `BTS_POOLED_PREDICTION` (default: auto-detect from filesystem).

### Component C — Storage layout
**Path**: `data/models_pooled/seed_<N>/blend_<DATE>.pkl`
**Status file**: `data/models_pooled/<DATE>_status.json` with `{seed: {status: ok|failed, hash: <sha256>, train_time_s: <int>}, n_complete: <int>}`.
**Pruning**: daily cron deletes folders older than 5 days.

### Component D — Health check
**File**: `src/bts/health/pooled_training.py` (new, ~80 lines)
**Logic**: read `<TOMORROW>_status.json`. INFO if <10 complete; WARN if <8; CRITICAL if <5 (would force fallback to single).
**Alert path**: existing `dispatch_dm_for_critical` Bluesky DM.

### Component E — Validation harness (pre-cutover)
**File**: `scripts/validate_pooled_calibration.py` (new, ~150 lines)
**What**: backtests pooled (canonical-n10) vs single-seed=42 on 2025 season. Reports:
- Per-bucket calibration tables (matches F's analysis format)
- Brier score comparison
- P@1 and P(57) deltas
**Cost**: 1 box × ~12h (re-running full backtest at 10 seeds + averaging).
**Gate**: pooled must show ≥ 5pp better calibration in the 75-80% bucket on 2025 forward-realized data, OR Brier ≤ single-seed Brier. If neither: don't cut over.

### Component F — Cutover protocol (post-validation)
**Phased rollout**:
1. **Shadow** for 14 days: pooled trains daily, prod still uses seed=42; compare realized calibration of pooled-shadow-pick vs prod-pick day-by-day.
2. **Soft cut** for 14 days: prod uses pooled, but seed=42 single-seed pick logged as shadow for comparison. Calibration alert fires on divergence.
3. **Full cut**: pooled is the source of truth; single-seed retained as fallback only.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Cloud burst fails on a Hetzner outage morning | D5 fallback: prod uses yesterday's seed=42 model |
| Determinism breaks (e.g., libomp version drift across boxes) | D4 gate; bit-exact validation harness (Tasks 6 / Stage 2) catches before deploy |
| Pool calibration is *no better* than seed=42 (counter-intuitive but possible if seed=42's "luck" is real signal) | Component E gate; abort cutover if validation criteria not met |
| Per-day storage growth fills bts-mlb disk | D6 5-day retention + alert at 80% disk |
| Pool training conflicts with daily ingest job | Schedule training at 02:00 ET, after ingest's 01:00 ET completion window |

## Open questions — RESOLVED 2026-04-29

1. **Trigger mechanism**: bts-mlb systemd timer (Q1). Self-contained, no GHA cross-system trust dependency.
2. **Validation gate**: 3-day operational burn-in pilot (no separate shadow window) gated by Component E backtest validation. If E shows ambiguous results, fall back to a 7-day forward-realized comparison.
3. **Default**: `--pooled` on, `--single-seed` off (Q3 = symmetric to `--use-factored` flip pattern).
4. **Canonical-n10 sanity check**: Component E now backtests both canonical-n10 AND a fresh random-10. If canonical does materially better than random → investigate cherry-pick risk. Otherwise, ship.

## Effort estimate

- Component A: 1 day (extends audit_driver pattern)
- Component B: 0.5 day (small predict.py change)
- Component C: trivial (storage convention + folder creation)
- Component D: 0.5 day (TDD pattern matches our 9 existing health checks)
- Component E: 1 day (validation backtest + analysis)
- Component F shadow: 14 days passive (just data collection)
- Component F cutover: 0.5 day (flip default, monitor)

**Total active code time**: ~3-4 days. Plus 14d shadow window.

## Definition of done

1. Daily pooled-training cron runs reliably for 7 consecutive days without manual intervention.
2. `<DATE>_status.json` shows ≥9 of 10 successful seeds for ≥6 of those 7 days.
3. Component E validation: pooled Brier ≤ single-seed Brier on 2025 backtest.
4. Component F shadow: pooled-shadow calibration shows ≥3pp improvement in 75-80% bucket on ≥10 days of realized data.
5. Health alert fires correctly when 1 seed manually killed (smoke test).
