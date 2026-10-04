# K / contact / expected-AB screen — RESULT: powered NULL

**Date:** 2026-06-15 · **Branch:** `kcontact-screen` · **Verdict:** no ship candidate; model at ceiling.

## Run
12 seeds × 6 Hetzner boxes, walk-forward monthly refit through 2024-09 (2025 held out, never touched), 17 arms, `BTS_LGBM_DETERMINISTIC=1`. All 12 seeds `rc=0`. Fleet torn down + verified (only bts-mlb + miniflux remain). Primary metric = within-day **pair-count-weighted AUC** (the swing campaign's powered statistic; per-day NDCG@10 single-split was underpowered — permuted-null sat *above* baseline).

## Controls gate — PASS (verdict is trustworthy)
| control | pair-AUC Δ vs baseline | reading |
|---|---|---|
| `ctl_sentinel_gross` (label leak) | +0.409 → 1.0000 | harness sound |
| `ctl_sentinel_soft` (calibrated weak signal) | **+0.0502** | **POWER gate clears null ~25×** |
| `ctl_sentinel_leaky` (1-game same-day) | +0.0049 | leak canary fires |
| `ctl_permuted` / `ctl_mask_only` | +0.0011 / +0.0002 | clean nulls |

Null band (permuted+mask paired |Δ|): mean 0.0007, max 0.0020, p95 0.0015.

## Families — NULL
| arm | pair-AUC Δ | P@1 |
|---|---|---|
| `omni_ALL` (best) | **+0.0024** | 0.755 |
| `omni_pitcher` / `omni_batter` | +0.0013 / +0.0012 | 0.749 / 0.761 |
| `p_csw` / `b_chase_rate` (best singles) | +0.0013 / +0.0012 | 0.764 / 0.774 |
| all other singles | +0.000 … +0.0009 | ~noise |
| `p_zone_rate` | −0.0001 | — |
| `ctl_uncompression` (the bar) | +0.0004 | 0.778 |

**Why this is a null, not a find:** the best arm (`omni_ALL`, +0.0024) clears the now-tight null band but is **below the ~0.003 practical ship threshold** (smaller than the swing campaign's already-shelved +0.0030), is **smeared across all features** (no single feature clears the band), and **does not move P@1** — the contest metric — where every arm sits within noise and re-expressing the *existing* `pitcher_hr_30g` (un-compression, P@1 0.778) does as well or better. Real-but-useless: a whisper of overall-slate-ordering signal that never reaches the top pick. No winning bundle → no 2025 confirmation run.

## Conclusion
The one genuinely-new, leak-verified signal remaining for this model — pitcher contact/command (K/whiff/CSW/zone/BB) + the expected-AB walk-adjustment — adds nothing usable to the top pick. Combined with three prior deviation nulls, ~30 rejected features (`docs/validation/final-report.md`), the rejected market signal, and the MDP audit already ruling out a better-starter feature, this is the definitive **model-at-ceiling** result. Feature code is clean and leak-verified (33 unit tests) but **not promoted** (null). Branch retained as a record.
