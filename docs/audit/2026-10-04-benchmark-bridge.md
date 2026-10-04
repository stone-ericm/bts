# W1.2 benchmark bridge: results memo

**Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated). Numbers come only from the accepted run's `summary.json`.
**Design:** `docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md` rev 3, FROZEN. **Exposure:** X-21 (`e09a7b7`), predeclared before any outcome-bearing execution.
**Use:** descriptive only, under D3 = RESERVE. Nothing here nominates or tests a candidate.

## 1. The run
- **Accepted run:** `data/validation/w12_bridge/4e4bf5a-20261004T060725Z/` (copied to the restic-backed `data/hetzner_results/season_wrap_outputs/w12/`), code `4e4bf5a`, finished 2026-10-04 05:07:45 ET. It was accepted on technical grounds before any result was read for W1.3/W2.3: `ACCEPTED.json` lists the criteria and the sha256 of `table.parquet` (`accb6d48…`), `summary.json` (`2373a771…`) and `manifest.json` (`dae86f08…`). It is a full run (no smoke flags) over 105 registered dates of the season's 184 game dates (missing fraction 0.429).
- **Pre-output fixes.** Neither attempt below wrote any output, and none was read.
  1. **Memory (`24129fd`).** The first full attempt was OOM-killed under its 11G cap while training C-frozen. The runner held every slate game's full v1.1 feed in memory: about 4.2 MB each, measured on the box, across about 1,575 games. It now keeps `gameData` without its `players` map. The bridge reads only teams, weather, officials, venue, datetime and status. A test compares slots, start times and final status on full vs slim feeds.
  2. **Gzipped unit captures (`4e4bf5a`).** The static capture writer switched to `.json.gz` on 2026-07-10, but verified eligibility globbed only `units/*.json`. It therefore read 187 of 2,335 unit captures, and verified eligibility came out unknown for later dates: 1,854 of 26,399 candidate rows were verified-eligible in the stopped attempt. Both forms are now read. The fix is confirmed in the failure direction: the restarted run reports 14,094 verified-eligible rows.
  - **Operational changes, results unaffected:** the memory cap was raised from 11G to 13G at runtime (per-date scoring peaks hit the cap with no OOM kill), with `OOMScoreAdjust=1000` so the kernel kills this job before any production process.
- **The 03:00 data rebuild.** The production cron rebuilds `data/processed/pa_2026.parquet` at 03:00 ET. The run read that file at start (features) and reads it again at the end (outcome labels), and its manifest hashes the file at the end. Pre-03:00 sha256: `06264f2f…`. The 03:00 build rewrote the file at 03:02:53 ET with **byte-identical** content (sha256 `06264f2f…` again), so both reads saw the same data.
- **The 03:00 pause.** The run was frozen (`systemctl --user freeze`) from 02:58:41 to 03:07:51 ET, across the production data chain, at a 6.1 GB low point, then thawed after the sync completed. Its log shows the gap; the computation is unaffected.

## 2. Reproduction (C-served vs D), first
On the 77 selection-consistent dates, 19,050 rows have both a C-served and a D score:

| Class | Rows | Share |
|---|---|---|
| `exact_final_feed` (within 1e-9) | 7,532 | 39.5% |
| `inferred_weather_absent_at_serve` | 2,808 | 14.7% |
| `unexplained` | 8,710 | 45.7% |

- **The unexplained residuals are small:** median 0.0017, 90th percentile 0.0069, maximum 0.075. Over all reproducible rows the absolute residual's 99th percentile is 0.049.
- **Pre-game rows reproduce less often.** In the verified pool (games verified pre-game at written_at) 7,101 of 12,726 rows (55.8%) are unexplained, against 45.7% overall. This is the pattern the 6/11 smoke run showed: games already under way when the slate was written reproduce exactly from final feeds, while future games carry small differences. The likely sources are final-feed game fields that differ from what serving saw; the weather class shows one such source as a numerical sensitivity. No cause is verified (design §3).
- **Served artifacts:** C-served scored 81 of 105 dates; 24 are `sha_unbound` (no pick file binding an archived model, mostly skip days).

## 3. Coverage and denominators
- **Registered dates:** 105 served slates (6/11 → 9/27) of 184 season game dates.
- **Selection states:** 77 selection-consistent, 4 inconsistent, 24 no selection.
- **Candidate rows:** 26,399; pools: verified 14,094, surrogate 16,956, all 26,399.
- **Primary stratum** (selection-consistent × verified): 63 dates with a rank-1 in every surface.
- The A26/B26 walk-forwards ran over all 184 season test days; only registered dates and candidates are disclosed (X-21).

## 4. Surfaces and the chain
**Primary stratum: selection-consistent × verified, 63 dates.**

| Surface | Top-1 hits | AUC (equal-date) | Brier | Log loss | Stated − realized |
|---|---|---|---|---|---|
| A26: actual-PA walk-forward (oracle) | 53/63 = 84.1% | 0.651 | 0.221 | 0.633 | +0.013 |
| A26_count: count-normalized | 41/63 = 65.1% | 0.572 | 0.236 | 0.664 | +0.025 |
| B26: estimated-PA walk-forward | 42/63 = 66.7% | 0.568 | 0.234 | 0.661 | +0.024 |
| C_frozen: trained through 2025 | 44/63 = 69.8% | 0.567 | 0.237 | 0.666 | +0.031 |
| C_served: archived serving model | 43/63 = 68.3% | 0.567 | 0.237 | 0.666 | +0.031 |
| D: the served slate | 45/63 = 71.4% | 0.566 | 0.237 | 0.667 | +0.031 |

No rank-1 was no_pa or unknown in this stratum. **The chain** (W1.2's sign: later minus earlier; paired on each pair's common pool; date-block 95% intervals):

| Link | Top-1 change | Discordant (a hit only / b hit only) | Rank-1 changed dates | Brier change |
|---|---|---|---|---|
| A26 → A26_count | −19.0 pp [−30.2, −9.5] | 13 / 1 | 41 | +0.0142 [+0.0128, +0.0155] |
| A26_count → B26 | +1.6 pp [−3.2, +6.3] | 1 / 2 | 12 | +0.0002 [−0.0000, +0.0004] |
| B26 → C_frozen | +3.2 pp [−4.8, +11.1] | 2 / 4 | 19 | +0.0006 [+0.0001, +0.0012] |
| C_frozen → C_served | −1.6 pp [−6.3, +3.2] | 2 / 1 | 13 | +0.0001 [−0.0001, +0.0002] |
| C_served → D | +3.2 pp [0.0, +7.9] | 0 / 2 | 5 | +0.0001 [−0.0001, +0.0003] |

**Reading it (descriptive, under D3):**
- **The headline's optimism sits almost entirely in the first link.** A26 is the README-style actual-PA walk-forward: 84.1% here, against the README's 86% on 2021–2025. Normalizing each candidate's probability to the expected rather than the realized PA count (A26_count) drops top-1 by 19 points and AUC from 0.651 to 0.572. The realized count is outcome-adjacent hindsight: a batter who batted more often had more chances to hit. It is an oracle, not an achievable improvement.
- **Every serving-realistic surface sits together.** B26, C_frozen, C_served and D all land at 66.7–71.4% top-1 and 0.566–0.568 AUC on these dates. No later link moves top-1 by more than about 3 points, and each interval includes zero (C_served → D touches it: 0 / 2 discordant).
- **The serving path stays close numerically** even where it is not exact: C_served → D changes the rank-1 on 5 of 63 dates.
- **All serving-realistic surfaces overpredict by about 3 points** (stated minus realized +0.024 to +0.031); A26's oracle count brings it to +0.013.
- **Other strata agree.** Selection-consistent × all pools, 77 dates: A26 84.4%, A26_count 67.5%, B26 71.4%, C_frozen 72.7%, D 72.7%; A26 → A26_count −16.9 pp [−27.3, −7.8]. All dates × all pools, 105 dates: A26 81.0% against D 69.5%; A26 → A26_count −11.4 pp [−20.0, −3.8].
- **Nothing here** separates drift, calibration, ball regime, selection or sampling. Those are W1.3's predeclared component diagnostics (X-29), which read this accepted run.

## 5. Prior results carried (not re-derived)
From the plan: the May PA-basis memo; M3's null; the June 29 PA-tilt; the May Gate B headline.

## 6. Limits
- **A window, not the season.** These are 105 served dates from 6/11, so 43% of the season's game dates have no served slate. 63 dates make the primary stratum: top-1 intervals are about ±10 points wide.
- **Final-feed game fields.** C uses archived final feeds for game-level fields, not what serving saw. The weather class is a numerical sensitivity, not a verified cause; nothing here proves attempt identity or historical availability (design §3).
- **One learning recipe at seed 42**, with `BTS_LGBM_DETERMINISTIC=1`. A26/B26 used all 2026 PA before each retrain date (the original calendar), so they are not a pristine 2026 holdout (plan rule 1.2).
- **Date-block intervals assume exchangeable dates.** They exclude cross-date dependence and selection effects.
- **Descriptive only (D3 = RESERVE).** No candidate is nominated or tested; any idea motivated here needs its own registration and 2027 validation.

