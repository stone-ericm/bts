# Result: catcher-grouped framing, 2026 out-of-sample test (X-37)

**Disposition: NEGATIVE** by the frozen rule (design `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` §6, frozen at `9b344d9`). Variant A's top pick hit exactly as often as the baseline's over the 2026 season, averaged over the ten registered seeds. The design calls a tie negative and says that does not imply measured harm.

**What this means.** The screen's positive result for A on 2024–2025 (+1.95pp, six of ten seeds, the minimum majority) did not carry over to the one season it had never seen. Nothing changes in production. There is no case for a 2027 prospective test of variant A on this evidence.

## The numbers (ten seeds, 184 game days, 2026-03-25 to 2026-09-27)

| | A-posted (decides) | A-projected (described only) |
|---|---|---|
| Mean difference in top-pick hit rate, m | 0.00pp | 0.00pp |
| Lower bound L (10th percentile, paired day resampling) | −1.30pp | −1.41pp |
| Same, seven-day blocks | −1.20pp | −1.25pp |
| Seeds where the arm beat the baseline | 3 of 10 | 3 of 10 |
| Per-seed difference in hits over 184 days | −1, 0, −3, +8, 0, 0, −5, +4, +2, −5 | −3, 0, −3, +7, −2, 0, −4, +4, +3, −2 |

- Positive needed all three of m ≥ +0.3pp, L > 0 and at least 6 of 10 seeds. None held.
- Top-pick hit rate, mean over seeds: baseline 70.16%, A-posted 70.16%, A-projected 70.16% (per-seed range 66.8% to 73.4%).
- Days on which the ten-seed mean differed: 14 up, 21 down, 149 equal. The two resamplings agree in sign.
- Catcher value available in the deciding arm: 4,771 of 4,854 side-games (83 had fewer than five earlier daily rates; none lacked an identified catcher). A-projected: 4,843 of 4,854.
- No rank ties at rank 1 or elsewhere in any retained profile.

**Streak measures (no weight in the disposition, design §6).** They moved the other way from the top-pick rate and from the screen: simulated mean longest streak rose on 7 of 10 seeds under A-posted (mean +1.49) and the replayed longest streak rose on 7 of 10 (mean +3.6 days); A-projected was similar. In the screen they fell on 8 of 10 seeds. With a zero top-pick difference these come from changes in the order of hits and in lower-ranked picks, and one season of ten correlated seeds cannot separate that from noise. Reported as required; not a finding.

## How it was run, and its limits

- **Where:** the Mac Studio, 2026-11-12 23:18 to 2026-11-13 11:50 ET, ten seeds one at a time, 75 minutes each, 108.9 CPU-hours. One one-minute pause for an EoR engine.
- **Deviations from the frozen design** are those declared in X-37 before any 2026 feed was parsed: `pa_2026` rebuilt from re-downloaded feeds (183,174 plate appearances, 2,427 games; identical row and game counts to the Mac's partial copy of the box build through 2026-07-06); Apple silicon, so not comparable byte for byte with the October box runs; no C1 launcher.
- **Code:** `scripts/audit/c2_framing/f26_studio.py` around `f26.py`'s unchanged science, at `e8dcbc6`. Its one review ran before any outcome was opened: Codex r1 BLOCK (the aggregate could accept incomplete or invalid evidence; prediction path confirmed faithful), fixed, r2 SIGN (`docs/audit/2026-11-13-c2-framing-2026-studio-codex-r{1,2}.md`). The aggregate ran once, at the signed commit.
- **A first start of seed 1 refused before computing anything** (the calendar was read from the wrong frame); fixed with a regression test and restarted. The refused attempt is kept beside the run.
- **What the test does not show** (design §§1, 9, 10): which catcher production would have known at pick time (A-posted uses the final-boxscore starter as a proxy); anything about variant B, which was not carried; anything beyond one season.
- **Evidence:** `~/projects/bts-c2-framing-results/2026-test/studio-run/` (run directories, inputs with `PREPARED.json`, logs, `files.sha256`, `f26-out/aggregate.json` sha256 `bfa0beadb21de0e4…`). Not in git; in the Mac backup set only if that directory is.

## Side observation, outside the test

The baseline's 2026 top-pick hit rate here, 70.2%, is a walk-forward of the screen recipe on 2026. It is in line with the live season (68 to 73 percent) and below the same recipe's 2024–2025 walk-forwards. This reads 2026 baseline outcomes, which X-37 covers as part of the contrast; it is noted for the calibration thread and is not a new analysis.
