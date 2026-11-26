# M3: serving-lookup staleness — verification, sizing, and fix design

**Status: CLOSED — HOLD, do not change serving (2026-06-11).** The staleness is
real and feature-level large (bpm 0.73 std). A serving-parity replay across
446 days (2024/2025/2026; the harness's reproduction of *current* serving is
validated to machine precision against the real lookups) **did not detect a
decision-level benefit from fixing it**: pooled Δtop-1 = −0.67pp, 95% CI
[−3.4pp, +2.0pp], sign-test p = 0.75. Important power caveat: the top-1 pick
identity changes on ~20–29% of days, but only **39 days have discordant
outcomes** (18 fresh-only-hit vs 21 stale-only-hit) — that is the effective
evidence, and the CI still permits effects in either direction that could
matter. The verdict is therefore "inconclusive with a near-zero point
estimate," NOT "proven harmless": HOLD because there is no demonstrated
benefit to justify touching the live prediction path (a judgment call given 5
prior bugs in adjacent code classes), with an explicit revisit trigger below.
Follow-up to the 2026-06-09 Fable-5
audit finding "[P1] Inference lookups stale by one played day; bpm collapses to
the prior for single-meeting pairs" (`docs/audit/2026-06-09-fable5-full-audit.md`).
The audit recommended a human spot-check before acting; this doc records it.

## The finding, verified in code

Training contract (`compute.py`): every rolling/expanding feature groups date-level
aggregates by entity and applies `shift(1)` — the value stored at date D reflects
data **strictly before D**. Correct and leak-free for training.

Serving (`predict.py` `_build_feature_lookups`): every lookup is
`df.dropna(subset=[col]).groupby(entity)[col].last()` — the value at the entity's
most recent **played** date D_last. That value excludes the game played on D_last.
Predicting D_next should use data through D_last; serving uses data through
D_last − 1 played date. **Every entity, every feature, every day, exactly one
played date stale.**

Worst case, `batter_pitcher_shrunk_hr` (bpm): `_cum_hits_prior = cumsum − current_day`
(`compute.py:652`) means a pair's only-meeting date stores exactly the league prior
0.2195 — identical to the no-history fallback at `predict.py:622`. A batter who went
3-for-4 against tonight's pitcher in their single prior meeting is served as if
they had never met.

Scope note: the backtest (`simulate/backtest_blend.py`) scores feature **rows**
directly and never calls `_build_feature_lookups` — no backtest number has ever
seen this gap. It is pure serving-path degradation. (An early hypothesis — that
fixing it would recover already-validated backtest performance, e.g. bpm's
+2.77 to +3.49pp promotion case measured on fresh values — was NOT borne out by
the decision-level replay below; the replay's point estimate is slightly
negative.)

## Spot-check: feature-level gap on real data

`scripts/spotcheck_m3_staleness.py` — appends one synthetic next-day row per
active entity (442 batters, 478 pitchers, 2023–2026 history through 2026-06-09)
and reruns the real `compute_all_features`; the shift(1) machinery then emits the
true as-of value at the synthetic date. No transform is duplicated. Sanity check:
the augmented run leaves all historical lookup values bit-identical (0 mismatches).

| feature | % entities changed | mean \|Δ\| / feature std |
|---|---|---|
| **batter_pitcher_shrunk_hr** | 100% | **0.73** |
| **batter_hr_7g** | 71% | **0.38** |
| batter_sweet_spot_rate_30g | 81% | 0.20 |
| batter_hr_30g | 77% | 0.18 |
| batter_avg_velo_faced_30g | 100% | 0.16 |
| pitcher_hr_30g | 87% | 0.15 |
| batter_count_tendency_30g | 93% | 0.15 |
| batter_hard_hit_rate_30g | 84% | 0.14 |
| batter_hr_60g / 120g, barrel, avg_ev | ~80–98% | 0.09–0.13 |
| pitcher Statcast (velo/spin/extension/break) | 100% | 0.01–0.03 |

bpm breakdown: of 442 active batter × last-faced-pitcher pairs, **258 (58%) have
exactly one prior meeting and 100% of them are served exactly 0.2195** (true as-of
values range 0.169–0.350). Multi-meeting pairs still move mean |Δ| 0.025.

## Decision-level replay (running)

`scripts/replay_m3_serving_parity.py` — trains the production single+12-blend on
pre-fold seasons, scores each fold day's actual starting lineups twice (FRESH =
as-of values = post-fix serving; STALE = previous-played-date values = current
serving; identical machinery, ffill fallback both arms), compares daily top-1/
top-3 hit rate, game-level AUC, pick divergence, paired day-bootstrap CI.

Folds: 2024, 2025, 2026-to-date.

### Results (v2 harness — joint candidate mask, pooled stats)

| fold | FRESH top1 | STALE top1 | Δ top1 | 95% CI | AUC Δ | pick differs |
|---|---|---|---|---|---|---|
| 2024 (185d) | 0.7784 | 0.7838 | −0.0054 | [−.038, +.027] | +0.0005 | 24.3% of days |
| 2025 (184d) | 0.6902 | 0.7011 | −0.0109 | [−.060, +.038] | +0.0004 | 28.8% |
| 2026 (77d) | 0.7013 | 0.6883 | +0.0130 | [−.052, +.078] | +0.0007 | 20.8% |
| **POOLED (446d)** | | | **−0.0045** | **[−.031, +.022]** | | |

Discordant days pooled: fresh-only-hit = 18, stale-only-hit = 20 (sign-test
p = 0.87). **The staleness changes WHICH batter is picked on ~1 in 4 days; the
replay did not show that those pick changes alter the hit rate.** AUC
differences ≤ 0.0007. Consistent with the 2026-06-10 finding that the model is
near its achievable ceiling: adjacent top candidates are separated by less
than the feature noise, so swapping among them moves outcomes within noise.

### Codex (gpt-5.5) methodology review — incorporated

Real findings, fixed in v2/v3: per-arm NaN masks could rank different candidate
pools (→ joint mask; in practice dropped 0 rows); per-fold CIs are weak since
most days both arms pick the same batter (→ pooled fold-stratified bootstrap +
sign test on discordant days); demanded a golden parity test of the STALE arm
against the real `_build_feature_lookups` (→ added, and it caught a real bug,
below); flagged that exact-(key,date) merges flatten sparse composite keys to
the prior in both arms (→ measured: bpm merge coverage 89–90%, and fixed, below).

False flags (lacked repo context): the full-history pitcher-hand map is exactly
production's `_pitcher_hand_lookup`; per-column ffill on level frames does
reproduce per-feature `dropna().last()`; one row per (batter, game) matches
production slot ranking.

### Parity bug found by the golden test (v2 → v3)

v2 parity check: max|diff| up to 0.145 vs production lookups. Debug
(`scripts/debug_m3_parity.py`) isolated it to **bpm only — 4 of 7101 sampled
values**; every other feature matched to 1e-9. Cause: for slate rows where the
batter did not actually face the modal-starter proxy that day (~10% of rows,
~1.3% with real pair history), the exact-date merge missed and BOTH arms fell
back to the prior — but production serving would return the pair's last-meeting
value (e.g. served 0.346, harness said 0.2195). Both arms flattened equally, so
it removed those rows' contribution rather than biasing direction — but it
understated the bpm staleness contrast.

v3 fix: bpm arms rebuilt from first principles (golden-asserted against the
pipeline's own column) + `merge_asof` to the last meeting strictly before the
slate date — STALE = that meeting's stored value (= production `.last()`),
FRESH = the value including that meeting's outcomes (= fixed serving).

### Results (v3 — final; STALE-arm parity vs production lookups = 5.55e-17, all folds)

| fold | FRESH top1 | STALE top1 | Δ top1 | 95% CI | AUC F/S | pick differs | discordant (F/S) |
|---|---|---|---|---|---|---|---|
| 2024 (185d) | 0.7730 | 0.7838 | −0.0108 | [−.043, +.022] | .5923/.5918 | 24.3% | 4 / 6 |
| 2025 (184d) | 0.6902 | 0.7011 | −0.0109 | [−.060, +.038] | .5906/.5903 | 28.8% | 10 / 12 |
| 2026 (77d) | 0.7013 | 0.6883 | +0.0130 | [−.052, +.078] | .5836/.5829 | 19.5% | 4 / 3 |
| **POOLED (446d)** | | | **−0.0067** | **[−.034, +.020]** | | | **18 / 21, p=0.749** |

## Verdict: HOLD — close M3 without changing serving

What the evidence supports, stated precisely:

- The contract violation is real, verified, and feature-level substantial.
- The replay **did not detect** a top-1 hit-rate benefit from fixing it. The
  effective paired evidence is **39 discordant-outcome days** (18 fresh-only
  hits vs 21 stale-only), point estimate −0.67pp, CI [−3.4pp, +2.0pp]. This is
  an *inconclusive result with a near-zero point estimate*, not proof of
  harmlessness — the CI permits a benefit (or harm) of a size that could
  matter over a season.
- HOLD is a risk judgment, not a measured cost-benefit: the fix (synthetic
  prediction-date rows — design above; the *mechanism* was validated by the
  spot-check, but its live behavior, slate timing, and probable-pitcher
  availability were not exercised here) touches the live prediction path,
  where 5 prior bugs have occurred in adjacent classes, and there is no
  demonstrated benefit to buy with that risk. Same shape as the MDP estpa HOLD.
- **Secondary finding (bounded)**: bpm has been served in materially degraded
  form since its 2026-04-29 promotion (58% of matchup pairs collapse to the
  prior), and top-1 decisions barely differ. This does not quantify bpm's
  realized live contribution (it may act through calibration, sub-top-1
  ranking, or interactions), but it is a reason to discount fresh-value
  backtest deltas when weighing future feature promotions.
- Revisit triggers: (a) the model materially improves top-of-slate
  discrimination (starter AUC well above ~0.59) — re-run
  `scripts/replay_m3_serving_parity.py`; (b) if settling the question becomes
  worth it, the higher-power instrument is a live shadow A/B (score picks both
  ways in production, compare over a season), which also exercises the fix's
  real serving-time behavior.

## Artifacts

- `scripts/spotcheck_m3_staleness.py` — feature-level gap sizing (synthetic
  next-day rows through the real pipeline).
- `scripts/replay_m3_serving_parity.py` — decision-level replay; STALE arm
  golden-tested against `_build_feature_lookups` (max|diff| 5.55e-17).
- `scripts/debug_m3_parity.py` — the drill-down that isolated the v2 bpm
  exact-date-merge flattening.
- Codex (gpt-5.5) methodology review incorporated; its sparse-composite-key
  concern was confirmed by the parity test and fixed in v3.

Methodology caveats (deliberate):
- Slate = actual starters (lineup 1–9); opposing starter proxied by the modal
  pitcher faced by that game-side (PA row order within a game isn't reliable).
- PA aggregation uses the production lineup-slot map (4.5→3.6) for both arms;
  estimated-vs-actual-PA basis is a separate known issue (MDP hold) and is held
  constant here.
- Openers not modeled (same both arms).
- One train per fold (not walk-forward daily retrain like production). The
  comparison is paired — same models, same slates, only freshness differs — so
  the delta isolates staleness; absolute hit-rate levels will differ from the
  walk-forward backtest.

## Fix design (NOT SHIPPED — retained for the revisit triggers above)

**Synthetic prediction-date rows through the real pipeline** (the mechanism the
spot-check validated):

1. In `run_and_pick`, after loading parquets and before `compute_all_features`,
   append one synthetic row per active entity (or per tonight's slate slot) at
   the prediction date: copy of the entity's last PA row with `date` set to the
   prediction date, a sentinel `game_pk`, and `is_hit = NaN`.
2. `compute_all_features` then emits the correct as-of value at the prediction
   date via the identical training transform — no duplicated logic, contract
   preserved by construction.
3. `_build_feature_lookups(df)` `.last()` picks up the synthetic-date rows
   unchanged.
4. **Guards required:**
   - Training must exclude synthetic rows — `train_model`/`train_blend` mask on
     feature-NaN, not target-NaN, so they would currently ingest them. Filter
     `is_hit.notna()` (good hardening regardless) or pass the pre-augmentation
     frame to training.
   - `_check_opener` consumes the PA frame — synthetic rows would register as
     fake 1-PA appearances and skew the opener heuristic. Pass it the
     un-augmented frame.
   - bpm pair coverage: synthetic rows must pair tonight's batter with
     **tonight's probable pitcher** (not last-faced) for the pair lookup to be
     fresh for the actual matchup; requires building synthetic rows from the
     slate (after `_fetch_game_slots`), not just from history.

Alternative considered: a parallel "as-of" computation inside
`_build_feature_lookups` — rejected; duplicates 30+ transforms, the exact drift
risk the audit warned about.

(The final verdict is the **HOLD** section above. This writeup was reviewed
adversarially by Codex gpt-5.5 — "what's overstated?" — and the verdict
language was tightened accordingly: the replay result is recorded as
inconclusive-with-near-zero-point-estimate, not as proof of harmlessness.)
