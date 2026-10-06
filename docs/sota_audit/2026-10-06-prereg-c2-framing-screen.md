# C2 side item (e): catcher-grouped framing screen, stage one — pre-registration (revision 3)

**Status:**
- Revision 3, for the third and final review round (Eric, row **C2-framing-review-r3**).
  - Revision 2 answered review r1 (`docs/audit/2026-10-06-c2-framing-codex-r1.md`, BLOCK B1–B7).
  - Revision 3 answers review r2 (`docs/audit/2026-10-06-c2-framing-codex-r2.md`, BLOCK R2-1 to R2-4).
- The note is frozen at the reviewed commit once signed.
- Nothing runs before three things: a plain SIGN, the exposure row X-35, and the admission record.

**Authority:** Eric, 2026-10-06, in his own words relayed by job-search-52.
- Register row **C2-framing-side-item**: add this screen to C2 as a scoped side item, with no production change and no deploy. Stage one is 3 paired seeds; stage two (to 10 seeds) needs his second go-ahead. If the first walk-forward's measured cost is well above the estimate, stop and report.
- Register row **C2-framing-seed-gate**: seed 1 runs first, then a cost report. He decides seeds 2–3 and the 50 CPU-hour checkpoint then. He gave no pre-approval of that checkpoint.
- C2 step 2a keeps priority.

**What it can support:** the 2021–2025 seasons are already consumed for deployment-grade evidence (`docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md`).
- The screen can conclude at most "worth testing on untouched 2027 data". It can never conclude "ship it" (decision memo D2/D4).
- The three seeds measure sensitivity to the algorithm's seed on two consumed seasons. They are not three new season samples (§5).

## 1. The question
Production computes its framing proxy `pitcher_catcher_framing` in `bts.features.compute`:
- **Measure:** each PA's borderline called-strike rate, `pa_borderline_csr`.
- **Borderline pitch:** |pX| between 0.5 and 1.2, or within 0.3 ft of `sz_top` or `sz_bottom`.
- **History:** grouped by **pitcher** and date: the per-pitcher expanding mean of the daily means, with shift(1) and min_periods 5.

The pitcher grouping predates the catcher id in the PA schema (`31ead55`, then `38bc71c` and `0bfbb93`). A catcher-grouped version of the same measure has never been tested. The 4/09 `catcher_hit_rate` experiment measured hits allowed per catcher, a different quantity.

## 2. The variants
- **The feature:** `catcher_framing` is the same computation with `fielding_catcher_id` as the grouping key. It runs on the same `pa_borderline_csr` column, through `framing_by` in `scripts/audit/c2_framing/screen.py`.
- **Variant A:** the base feature set with `pitcher_catcher_framing` replaced by `catcher_framing`, at the same position.
- **Variant B:** the base feature set with `catcher_framing` appended.
- **Baseline:** the current production base, `FEATURE_COLS` (16 features), at the registered feature settings (§3).
- **Blend configs:** each of the 12 blend configs keeps its Statcast extras and any parameter dictionary. Only its base is rewritten, by `bts.experiment.runner`'s rule.

**Self-check before any walk-forward:** on the run's real feature frame, `framing_by(df, "pitcher_id")` must reproduce production's `pitcher_catcher_framing` exactly: same rows, same order, NaN equal. Otherwise the run stops (`STOPPED.json`, exit 4).
- The check certifies that the copy matches production's computation.
- It does not certify the input closure (§3) or event availability (§8).

## 3. Inputs: the complete read inventory (r1 B1, B6)
The run reads exactly ten pinned files and nothing else that affects a result.

**1. The nine PA parquets `pa_2017.parquet` … `pa_2025.parquet`:**
- They are in the production data directory on the box, and each is read once from its hashed bytes.
- 2026 is never read.

**2. The frozen probable-pitcher lookup `probable_pitcher_lookup.2017-2025.json`:**
- **Why:** `compute_all_features` builds `opp_bullpen_hr_30g`, a baseline feature, from a lookup. In production that lookup is the cache `data/models/probable_pitcher_lookup.json`, plus a scan of `data/raw`. On the box `data/raw` holds only 2026 (checked 2026-10-06), so historical games come from the cache.
- **How it is frozen:** `freeze-lookup`, run once on the box before admission, restricts that cache to the game ids of the nine pinned parquets, as canonical JSON.
- **Inside the run:** it replaces `_build_probable_pitcher_lookup` for the process. There is no raw scan and no cache write.
- **Recorded:** the lookup's coverage of the pinned games is in the manifest.

**Pins:** all ten files are pinned by sha256 in `scripts/audit/c2_framing/admission.json` (`input_pins`). The pins are hashed on the box before the exposure row, and X-35 cites their digest.

**Park drag:** `compute_all_features` also attaches `park_drag_delta` from an external table. That column is in no registered blend (it is a context column). The run replaces the attach with its table-absent result, so it is all NaN and nothing is read.

**Feature settings, read at import:**
- `ROOKIE_GATE_K = 20` and `PITCHER_HR_30G_MIN_PERIODS = 7`. Production's box environment sets neither, so production runs the source defaults (checked 2026-10-06).
- The run refuses any other effective value and records them in the manifest.
- LightGBM's params must carry `deterministic` and `force_row_wise`; the box environment sets `BTS_LGBM_DETERMINISTIC=1`. The run refuses otherwise and records the params.
- The seed is read from `BTS_LGBM_RANDOM_STATE` at each model's creation, and the run sets it.

**Coverage of `fielding_catcher_id`** (parquet metadata, 2026-10-06):
- 2019–2025: no nulls.
- 2017–2018: no such column, so the catcher feature is NaN there.
- Training starts in 2019. The run records test-season coverage.

## 4. Procedure
**Prep, once, before admission** (on the box):
1. Hash the nine parquets.
2. Run `freeze-lookup` (`--cache data/models/probable_pitcher_lookup.json`) to write the frozen lookup to `data/hetzner_results/c2/framing_screen/inputs/`, and hash it.
3. Write `admission.json`: the reviewed commit, the archived review, the ten pins, and the exposure commit of X-35.

**Per seed (one claimed run each), through `screen.py launch --seed S`.** The launch wrapper refuses unless the same admission and seed-order checks pass that `run` makes. Then:
1. It builds the C1 launcher command:
   - `--name c2-framing-seed<k>`;
   - `--cpu-hours` from §6;
   - `--max-hours 16`;
   - `env BTS_LGBM_DETERMINISTIC=1 TZ=America/New_York … screen run --seed S`.
2. The admission gate (shared `scripts/audit/c1/admission.py`) checks:
   - the reviewed commit, a plain SIGN of this unit, and X-35 published unchanged;
   - the closure unchanged, with no foreign modules.
3. Seed order:
   - Seed 1 (2273360) may run first.
   - Seeds 2–3 run only after Eric's release (r2 R2-2). It is register row `C2-framing-release-seeds-2-3`, whose ruling cell reads exactly "**RULED <date> (Eric): RELEASE seeds 2–3 of the framing screen after seed-1 run `<run name>`; declared budget <N> CPU-hours per seed**":
     - the source cell's first token must be exactly `Eric`, the shared gate's rule;
     - N must be finite and positive;
     - the named run must be the only run under seed 1's claim root, and it must validate as a complete run of the same admitted identity and pins (§5, `validate_run`).
     - An empty or partial results file is not a completion.
4. Claim: one durable claim per seed. It lives only under the canonical `~/projects/bts/data/hetzner_results/c2/framing_screen/seed_<seed>/`. The command line has no output-root option.
5. Inputs: load the ten pinned inputs, install the frozen lookup and the park-drag bypass, then run `compute_all_features`, then the self-check, then add `catcher_framing`.
6. Labels (the BTS scoring rule, r1 B4):
   - Each profile row's `actual_hit` is rebuilt from the pinned PA rows with the resumed portion of suspended games excluded (`filter_out_resumed_portion`), by `(batter_id, game_pk)`.
   - A row with no original-portion PA is **void**: it is dropped, and that day's ranks are renumbered in their original order.
   - Training and features keep every PA, as production does.
   - Per walk-forward, the run records how many labels changed and how many rows were void.
7. Walk-forwards: for baseline, then A, then B, and each of 2024 and 2025:
   - `blend_walk_forward(..., retrain_every=7, blend_configs=…, game_probability_mode="estimated_pa")`, top-10 profiles. This is the serving-realistic probability basis, not actual_pa hindsight.
   - Profiles, CPU time and label counts are saved.
8. Scoring:
   - `compute_full_scorecard` per variant;
   - `diff_scorecards` of A and B against the same seed's baseline;
   - the repo's per-seed screening rule `evaluate_pass_fail`.
   - Scorecards and diffs are retained.

**Seeds:** positions 0–2 of `data/seed_sets/canonical-n10.json`: 2273360, 260991262, 1746737973.
- That file's positions come from an outcome-ranked, stratified historical baseline distribution. Taking the first three is fixed here.
- They are neither an outcome-independent random sample nor full-range coverage of canonical-n10.

## 5. Dispositions (fixed now), per variant: only for exactly the three registered seeds
**Quantities:**
- For each seed, d = the mean of that seed's 2024 and 2025 P@1 deltas.
- m = the mean of the three d.
- t = m / (sd / √3). When sd = 0, t is +∞ if m > 0, −∞ if m < 0, and 0 if m = 0.

**positive** ("worth testing on untouched 2027 data") requires all of:
- the mean 2024 delta and the mean 2025 delta are both above 0;
- m ≥ +0.003 (+0.3pp, the June screens' practical threshold);
- t ≥ 1.5 (the repo's multi-seed keep rule);
- the per-seed screening rule holds on at least 2 of 3 seeds.

**negative:** m ≤ 0, and the per-seed rule holds on fewer than 2 seeds.

**inconclusive:** anything else. Only an inconclusive variant could justify stage two, and stage two needs Eric's second go-ahead.

**incomplete:** any input other than exactly the three registered seeds.

**What decides:**
- Per-season P@1 decides §5 directly.
- The exact P(57) and `mean_max_streak` deltas decide it **indirectly**, through the per-seed rule's neutral fallback: P@1 within 0.3pp in both seasons, plus `mean_max_streak` ≥ 0 and exact P(57) > 0.
- The CPU cost controls the stops (§6).
- Coverage and label counts are reported without decision weight.

**`aggregate` (r1 B2; r2 R2-1, R2-3):** it gives dispositions only when all of these hold:
- exactly three distinct run directories, whose claim roots name exactly the three registered seeds;
- each passes `validate_run`, which checks:
  - **namespace and state:** the canonical claim namespace (`…/framing_screen/seed_<seed>/<run>`), and no stop;
  - **claim:** a JSON claim that names the run and its code;
  - **manifest:** schema, seed, claim binding, a 40-hex head, the registered basis, retrain interval, test seasons and feature settings, deterministic LightGBM params and environment, all ten pins and their digest, a complete accepted identity, and an identical self-check;
  - **results:** the six registered units, in order;
  - **retained artifacts:** every profile, scorecard and diff;
  - **reconciliation:** P@1 per season recomputed from the profiles must equal the scorecard and the results; each diff recomputed from the retained scorecards must equal the stored diff; each stored summary recomputed from its diff must match;
- they agree on the accepted identity (review report and its sha256, reviewed and exposure commits, admission record), input pins, LightGBM params, feature settings, basis, retrain interval and test seasons.

**Each run keeps its own commit.** A metadata-only descendant, such as the commit that records Eric's release, is admitted by the shared gate. Equal commits are therefore not required, and the frozen executable identity is compared instead (r2 R2-1).

Anything else is refused.

**What these are not:**
- With 3 seeds, t ≥ 1.5 is a screening convention, not a significance test.
- A and B are reported separately, with no multiplicity adjustment.
- No disposition approves a production change.

## 6. Compute, budgets and stops (r1 B7)
**Estimate (unmeasured):** about 5 CPU-hours per season walk-forward, so about 30 per seed (6 walk-forwards) and about 90 for stage one.

**Seed 1:** declared budget **45 CPU-hours** (`--cpu-hours 45`), within Eric's stage-one approval.
- The combined C1 + C2 ledger was at 0.48 on 2026-10-06, so the launch is within both gates.
- The 50 CPU-hour checkpoint is checked between launches, not during a job, so a job may cross it while running.
- 100 is the hard cap. The launcher refuses any launch whose declared budget would exceed it, and a 50 acknowledgement does not raise it.

**First-walk-forward stop, on every seed:** a seed's first walk-forward (baseline 2024) that uses more than 7.5 CPU-hours (1.5 × the estimate) stops that run after it completes (`STOPPED.json`, exit 3).
- For seed 1 this is Eric's condition.
- The lead reports before anything else runs.
- A stopped run is not a complete seed.

**The seed-1 report to Eric** (through job-search-52, or the fallback file and the manager) gives:
- seed 1's measured CPU, with each walk-forward's cost;
- the projected actual cost of seeds 2–3;
- the ledger total after seed 1;
- the **declared-budget headroom**: what budget per seed would let both seeds launch under the 100 cap;
- what remains for C2's planned jobs (11.1 CPU-hours planned).

**Seeds 2–3:**
- Eric's release row fixes their declared budget per seed. The wrapper uses that number.
- If the 50 checkpoint is reached, his `CHECKPOINT_50_ACK.json` is also needed. The launcher enforces it.
- If the cap cannot fit both seeds, stage one stays incomplete. Raising the cap is a separate decision, never made here.

## 7. Reporting
- The per-seed `results.json` files, then `aggregate` (§5).
- A results note, then an independent acceptance (C2 rule). Then the result goes to Eric through job-search-52:
  - per-seed and mean P@1 deltas per season for A and B;
  - the dispositions;
  - the measured CPU cost.
- The fallback is `~/projects/job-search/mets-2026-10-06/catcher-experiment-result.md`, with the herdr manager told.

## 8. Limits
- **Catcher identity is a postgame, game-level proxy.** It is one catcher per side per game: the first boxscore player whose positions include catcher, with a fallback to the primary position (`bts.data.build._get_starting_catchers`).
  - The selected player **need not be the starter**: boxscore order decides.
  - It is not the catcher of each PA. Substitutions, pitching, umpiring and game context mix into the measure.
  - A 2027 test needs an independently specified pregame catcher identity.
- **No catcher history before 2019.** The pitcher feature has it.
- **Event availability (r1 B4): leak-free is bounded.**
  - Every PA carries its game's official date, including PAs from the resumed portion of a suspended game, which happened later.
  - Date-level shift(1) therefore admits a resumed-portion PA into history from the original official date, before the event happened.
  - This is production's convention, shared by baseline and both variants.
  - The run records the count of such rows per season. The claim is "same measure, date-level shifting, conditional on the official-date convention", not unconditional leak freedom.
- **Labels follow BTS scoring** (resumed portion excluded; void rows dropped). Prior repo screens using the walk-forward's own labels are not comparable on that point.
- **A small, consumed evaluation:** 2 seasons, 3 algorithm seeds.
- **Not comparable to 2026-03-31's 85.1% → 87.0%:** a single model, a 13-feature base, seed 42, and probably the actual_pa basis.
- **No serving input:** production has no pre-game catcher input.
