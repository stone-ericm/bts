# Framing screen, stage two: result over ten seeds (C2 side item (e))

**What this is:** the results note for the catcher-grouped framing screen over all ten registered seeds.
- **The design:** the stage-one pre-registration `docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md` (frozen at `a3f5e3e`), extended by the stage-two addendum `docs/sota_audit/2026-10-08-prereg-c2-framing-stage-two.md` (review s2 plain SIGN, `docs/audit/2026-10-08-c2-framing-stage-two-codex-s2.md`).
- **The question:** does grouping production's framing proxy by the catcher (`fielding_catcher_id`) instead of the pitcher improve the model's pick ranking?
  - Variant **A** replaces `pitcher_catcher_framing` with the catcher-grouped measure.
  - Variant **B** adds it alongside.
  - Both are scored against the same seed's baseline on 2024 and 2025.
- **What it can support** (both documents, unchanged): at most "worth testing on untouched 2027 data", never "ship it". 2024 and 2025 are already consumed seasons. The ten seeds measure the algorithm's seed sensitivity on those two seasons; they are not ten season samples.

**Status:** all ten registered seeds ran; `aggregate-stage-two` passed on the box. **Independent acceptance: pending** (a fresh Codex session, Part 1 blind). By the manager's condition (11:05 EDT 2026-10-09), the result goes to Eric only after a plain SIGN; anything else goes to the manager first. Nothing here approves a production change, a deploy or a further run.

## Reading record
Who read what, and when (all EDT), according to the lead's account. The supplied artifacts support the production chronology, but do not independently audit reads or distribution.
- **Disclosure (addendum §6): stage one's three seeds were opened before stage two ran.** The lead opened them at 14:27 EDT on 2026-10-08, and Eric received them before the addendum was written. The addendum's rules are the mechanical extension of stage one's. This note reports the ten-seed result as the result.
- **Stage two, until 11:03:54 EDT on 2026-10-09, only costs and label counts were read:**
  - the `cpu_s`, `wall_s` and `labels` fields of each seed's units.json;
  - the progress lines of the log tails, and run-directory listings (file names, sizes, times);
  - the C1 launcher's records (PENDING, TERMINAL, RECONCILED, the ledger).
- **The unsealing step.** The lead announced it to the manager at 11:01 EDT, with the ledger, headroom, all ten TERMINAL and RECONCILED records and the exact command. The manager checked them read-only and replied GO at 11:02 EDT.
- **First opening.** bts-lead2 (the BTS lead, herdr pane wA:pQ) first opened the stage-two outcomes at 11:03:54 EDT, when it printed `aggregate-stage-two.stdout.json` (file mtime 11:03:50).
  - The print appeared in the lead's terminal pane, which Eric can view. No message carried it to him or to the manager.
  - The manager checked the file's size, mtime and sha256 without opening it (its account, 11:04:30).
- **Copies.** The seven stage-two run directories were then copied to the Mac (11:04:57): 105 files, with sha256 matching a list taken read-only on the box. Stage one's three copies (45 files) were re-checked against the accepted list. After that, the copies were read only by the lead, for this note's tables and the cross-platform check below.
- **Not sent.** Nothing has gone to Eric or the manager beyond operational status.

## The runs
All ten were launched through the C1 launcher from `~/projects/bts-c1`, one at a time, at nice 10, with `BTS_LGBM_DETERMINISTIC=1`. Each ran six walk-forwards: baseline, A and B, for 2024 and 2025. Each walk-forward was `estimated_pa`, retrained every 7 days, and kept top-10 profiles.

| # | Seed | Unit | Run (claim) | Ran (EDT) | CPU (guard) |
|---|---|---|---|---|---|
| 1 | 2273360 | `c1-c2-framing-seed1-20261008T061510Z-ad237918` | `22d31f2-20261008T061514Z` | 10/08 02:15 → 05:21 | 65,620.4 s = 18.23 h |
| 2 | 260991262 | `c1-c2-framing-seed2-20261008T134258Z-6b2aae98` | `11e0cfd-20261008T134306Z` | 10/08 09:42 → 11:37 | 48,017.4 s = 13.34 h |
| 3 | 1746737973 | `c1-c2-framing-seed3-20261008T163012Z-650ee8e4` | `11e0cfd-20261008T163019Z` | 10/08 12:30 → 14:23 | 47,575.6 s = 13.22 h |
| 4 | 2048 | `c1-c2-framing-seed4-20261008T235024Z-30b27165` | `88e426e-20261008T235044Z` | 10/08 19:50 → 21:43 | 47,306.8 s = 13.14 h |
| 5 | 3629294338 | `c1-c2-framing-seed5-20261009T014545Z-e3b8be82` | `88e426e-20261009T014610Z` | 10/08 21:45 → 23:38 | 47,375.3 s = 13.16 h |
| 6 | 1277948386 | `c1-c2-framing-seed6-20261009T034058Z-4b704778` | `88e426e-20261009T034129Z` | 10/08 23:40 → 10/09 01:33 | 47,078.2 s = 13.08 h |
| 7 | 3219332220 | `c1-c2-framing-seed7-20261009T071152Z-83b9788c` | `88e426e-20261009T071229Z` | 10/09 03:11 → 05:05 | 47,667.3 s = 13.24 h |
| 8 | 2207587974 | `c1-c2-framing-seed8-20261009T091010Z-e91dc500` | `88e426e-20261009T091054Z` | 10/09 05:10 → 07:05 | 48,276.9 s = 13.41 h |
| 9 | 3170105529 | `c1-c2-framing-seed9-20261009T110914Z-d60fb56e` | `88e426e-20261009T111003Z` | 10/09 07:09 → 09:03 | 47,561.5 s = 13.21 h |
| 10 | 2675988121 | `c1-c2-framing-seed10-20261009T130515Z-e70e7501` | `88e426e-20261009T130609Z` | 10/09 09:05 → 10:58 | 47,392.1 s = 13.16 h |

- **Exit and records.** Every unit exited rc 0. Every guard record (TERMINAL) is `exit` with leftover false, and every RECONCILED record is `exit` with no problems, with the same CPU seconds as the guard's. Every walk-forward reports labels changed 0 and void 0.
- **Code.** Seeds 1–3 ran at stage one's commits (`22d31f2`, and `11e0cfd`, which adds only Eric's release row) under stage one's identity. Seeds 4–10 ran at `88e426e` under the stage-two admission (exposure row X-36). The aggregate validated each run under its own stage's identity (addendum §4).
- **Stage two's operating rules held.** Each stage-two seed launched only after the previous seed's TERMINAL and RECONCILED records, a ledger reading and a heads-up to the manager. None launched inside 00:45–03:10 by the box clock. Each passed the guarded-unit check on the box; the manager verified each launch read-only (C2 index, row (e)).
- **Output root:** `data/hetzner_results/c2/framing_screen/seed_<seed>/<run>/` on the box (restic-backed archive set).

## Result

**P@1 by seed** (top-ranked pick hit rate; 185 test days in 2024, 184 in 2025; hits in brackets; rebuilt from the retained profiles by `tables.py`):

| # | Seed | Season | Baseline | A | A − baseline | B | B − baseline |
|---|---|---|---|---|---|---|---|
| 1 | 2273360 | 2024 | 78.9% (146) | 78.9% (146) | **0.00pp** (0) | 77.3% (143) | **−1.62pp** (−3) |
| 1 | 2273360 | 2025 | 67.4% (124) | 70.1% (129) | **+2.72pp** (+5) | 71.7% (132) | **+4.35pp** (+8) |
| 2 | 260991262 | 2024 | 74.6% (138) | 77.8% (144) | **+3.24pp** (+6) | 76.8% (142) | **+2.16pp** (+4) |
| 2 | 260991262 | 2025 | 66.8% (123) | 74.5% (137) | **+7.61pp** (+14) | 69.0% (127) | **+2.17pp** (+4) |
| 3 | 1746737973 | 2024 | 77.8% (144) | 76.8% (142) | **−1.08pp** (−2) | 76.2% (141) | **−1.62pp** (−3) |
| 3 | 1746737973 | 2025 | 67.9% (125) | 71.7% (132) | **+3.80pp** (+7) | 72.3% (133) | **+4.35pp** (+8) |
| 4 | 2048 | 2024 | 77.8% (144) | 76.8% (142) | **−1.08pp** (−2) | 76.8% (142) | **−1.08pp** (−2) |
| 4 | 2048 | 2025 | 69.0% (127) | 72.3% (133) | **+3.26pp** (+6) | 71.2% (131) | **+2.17pp** (+4) |
| 5 | 3629294338 | 2024 | 75.7% (140) | 76.8% (142) | **+1.08pp** (+2) | 74.6% (138) | **−1.08pp** (−2) |
| 5 | 3629294338 | 2025 | 68.5% (126) | 70.7% (130) | **+2.17pp** (+4) | 72.3% (133) | **+3.80pp** (+7) |
| 6 | 1277948386 | 2024 | 76.8% (142) | 78.4% (145) | **+1.62pp** (+3) | 77.3% (143) | **+0.54pp** (+1) |
| 6 | 1277948386 | 2025 | 67.4% (124) | 70.7% (130) | **+3.26pp** (+6) | 70.1% (129) | **+2.72pp** (+5) |
| 7 | 3219332220 | 2024 | 74.1% (137) | 76.8% (142) | **+2.70pp** (+5) | 76.8% (142) | **+2.70pp** (+5) |
| 7 | 3219332220 | 2025 | 67.9% (125) | 69.6% (128) | **+1.63pp** (+3) | 71.2% (131) | **+3.26pp** (+6) |
| 8 | 2207587974 | 2024 | 78.9% (146) | 76.2% (141) | **−2.70pp** (−5) | 77.3% (143) | **−1.62pp** (−3) |
| 8 | 2207587974 | 2025 | 68.5% (126) | 70.1% (129) | **+1.63pp** (+3) | 67.4% (124) | **−1.09pp** (−2) |
| 9 | 3170105529 | 2024 | 75.1% (139) | 78.9% (146) | **+3.78pp** (+7) | 76.8% (142) | **+1.62pp** (+3) |
| 9 | 3170105529 | 2025 | 70.1% (129) | 71.7% (132) | **+1.63pp** (+3) | 69.6% (128) | **−0.54pp** (−1) |
| 10 | 2675988121 | 2024 | 75.1% (139) | 77.8% (144) | **+2.70pp** (+5) | 77.8% (144) | **+2.70pp** (+5) |
| 10 | 2675988121 | 2025 | 67.9% (125) | 69.0% (127) | **+1.09pp** (+2) | 70.7% (130) | **+2.72pp** (+5) |

Seeds 1–3 are stage one's runs; their rows equal the stage-one note's.

**The addendum §5 quantities** (`tables.py` computes this table independently from the retained diffs; the box aggregate agrees on the season means, seed-level d, m, t, passes and dispositions; sd is computed here because the box aggregate has no sd field):

| | A | B |
|---|---|---|
| Mean 2024 delta | +1.03pp | +0.27pp |
| Mean 2025 delta | +2.88pp | +2.39pp |
| Seed-level d (seeds 1–10) | +1.36, +5.43, +1.36, +1.09, +1.63, +2.44, +2.17, −0.54, +2.71, +1.89pp | +1.36, +2.17, +1.36, +0.55, +1.36, +1.63, +2.98, −1.35, +0.54, +2.71pp |
| m (mean of d) | +1.95pp | +1.33pp |
| sd of d | 1.52pp | 1.24pp |
| t = m / (sd/√10) | 4.08 | 3.39 |
| Per-seed rule passed | **6 of 10** (seeds 2, 5, 6, 7, 9, 10) | **4 of 10** (seeds 2, 6, 7, 10) |
| **Disposition** | **positive** | **inconclusive** |

**The rule** (addendum §5): a variant is positive when all four hold: the mean 2024 and mean 2025 deltas are both above 0; m ≥ +0.3pp; t ≥ 1.5; and the per-seed rule holds on a majority of the seeds, at least 6 of 10. It is negative when m ≤ 0 and the per-seed rule holds on fewer than 6. Anything else is inconclusive.

**A is positive.** It meets all four conditions: both season means are above 0, m = +1.95pp, t = 4.08, and the per-seed rule holds on 6 seeds. That is exactly the 6 the rule requires.

**B is inconclusive.** It meets three conditions (both season means above 0, m = +1.33pp, t = 3.39) and fails the fourth: the per-seed rule holds on only 4 seeds. It is not negative because m > 0.

**The per-seed rule** (`bts.experiment.runner.evaluate_pass_fail`, unchanged) passes a seed when either:
1. P@1 improves in both seasons; or
2. the neutral fallback holds: no season's P@1 drops more than 0.3pp (delta ≥ −0.003), the `mean_max_streak` delta is ≥ 0, and exact P(57) strictly improves.

Seed by seed:
- **Every passing seed passes on condition 1,** for both variants. No seed passes on the fallback.
- **Every failing seed but one fails because a season drops more than 0.3pp,** so it never reaches the fallback's streak and P(57) tests:
  - A: seeds 3 (2024 −1.08pp), 4 (2024 −1.08pp) and 8 (2024 −2.70pp);
  - B: seeds 1 and 3 (2024 −1.62pp), 4 and 5 (2024 −1.08pp), 8 (both seasons down) and 9 (2025 −0.54pp).
- **Seed 1, A** (as in stage one): 2024 is flat (0.00pp), so it fails condition 1 and reaches the fallback. There, `mean_max_streak` is +0.42, but the exact-P(57) delta is exactly 0: the seed's three variants share one bit-identical exact P(57). So P(57) does not strictly improve, and the seed fails.
- **The wording gap from stage one.** Pre-registration §5 paraphrases the fallback as "P@1 within 0.3pp"; the code that ran means "no drop beyond 0.3pp". Only seed 1's A reaches the fallback, and under the "within" reading it fails one step earlier, on 2025's +2.72pp. Every verdict, and both dispositions, are the same under both readings.

**Secondary metrics** (decision weight only through the per-seed rule; from the retained diffs):

| # | Seed | A: mean_max_streak Δ | A: exact P(57) Δ | B: mean_max_streak Δ | B: exact P(57) Δ |
|---|---|---|---|---|---|
| 1 | 2273360 | +0.42 | 0 | +0.74 | 0 |
| 2 | 260991262 | −0.40 | −1.16e-08 | −1.58 | −1.16e-08 |
| 3 | 1746737973 | −3.55 | −5.29e-09 | −1.35 | −5.51e-09 |
| 4 | 2048 | −0.30 | 0 | −0.80 | −3.41e-11 |
| 5 | 3629294338 | −0.13 | +1.46e-08 | −1.43 | −1.33e-08 |
| 6 | 1277948386 | −1.90 | −1.21e-10 | −0.42 | +1.06e-09 |
| 7 | 3219332220 | −0.61 | 0 | −1.35 | 0 |
| 8 | 2207587974 | −1.29 | −8.73e-11 | −1.27 | −8.73e-11 |
| 9 | 3170105529 | +2.95 | −3.04e-09 | +2.06 | −3.04e-09 |
| 10 | 2675988121 | −1.53 | −7.63e-10 | −0.80 | −7.63e-10 |

## Things to weigh in the result
- **A is positive by the narrowest margin the rule allows on its fourth condition.** Six passing seeds is the minimum majority of ten; with five, A would be inconclusive (m > 0 rules out negative). Its other three conditions hold with room: m is about 6.5 times the +0.3pp threshold, and t is 4.08 against 1.5.
- **The seasons still disagree.** Under A, 2025 improves on all ten seeds (+1.09 to +7.61pp). 2024 is mixed: six seeds improve, one is flat and three drop (−1.08, −1.08 and −2.70pp). Each of A's four failing seeds misses on 2024 (one flat, three down). B's 2024 mean is +0.27pp, with five of ten seeds below zero.
- **The streak metrics point the other way.** `mean_max_streak` falls on 8 of 10 seeds under each variant (mean −0.63 under A, −0.62 under B). Exact P(57) falls on 6 seeds under A (3 equal, 1 up) and on 7 under B (2 equal, 1 up). These carry no decision weight beyond the per-seed rule, and the screen does not diagnose why they diverge from P@1. They are reported because BTS's objective is a streak, and they do not move in the same direction as P@1.
- **Baseline P@1 varies by seed.** It spans 74.1–78.9% in 2024 and 66.8–70.1% in 2025. The paired design compares each variant only with its own seed's baseline.
- **Two stages, one rule.** Stage two ran because stage one was inconclusive (addendum, "Authority"; pre-registration §5). The ten-seed rule is applied to the ten seeds as a single set, as the addendum fixes; no adjustment is made for the continuation decision. Descriptively, A's six passes are one from stage one's three seeds (seed 2) and five from stage two's seven; B's four are one and three. With ten seeds, t ≥ 1.5 remains a screening convention, not a significance test (addendum §5), and A and B are reported separately with no multiplicity adjustment.

## Reconciliation
- **On the box, it passes.** `aggregate-stage-two` ran on the box from `~/projects/bts-c1` at `88e426e` (clean; the command refused otherwise) at 11:02:50–11:03:50 EDT on 2026-10-09, with rc 0.
  - It ran under the passing admission gate and validated all ten runs. Stage one's three were checked byte for byte against the accepted hash list and re-validated under stage one's identity; the seven stage-two runs were validated under the stage-two admission. For each: claim, manifest, pins, identity, code descent and closure, season evidence, and every scorecard, diff and summary recomputed from the retained profiles with exact equality. All ten agree on pins, inputs, LightGBM parameters, feature settings, basis, retrain interval and test seasons.
  - Its output is `aggregate.box.json` (sha256 `684ac5a6…`), with `aggregate.box.stderr.txt`, in `docs/audit/2026-10-09-c2-framing-stage-two-evidence/`.
- **On the Mac, exact equality fails, as in stage one.** Re-running `aggregate_stage_two` on the hash-matched copies, with the copies as the namespace root, refuses seed 1's baseline scorecard (`aggregate.mac.out`), the same refusal stage one recorded. It stops at the first refusal.
  - **The stage-two runs, measured** (stage one's `rescore_diff.py`, run on the seven stage-two copies; output `rescore_diff.out`): over the 21 scorecards, 36 values differ, in only two fields, `p_57_exact` and `p_57_mdp`. Every difference is in the last digits, with relative size at most 1.25e-15. Every other field is equal, including every P@1 and streak metric. Stage one measured the same pattern on its nine scorecards.
  - **Mechanism:** platform floating-point differences (box x86_64, Mac arm64), inferred and not tested, as in stage one.
- **No decision changes.**
  - `p_57_mdp` enters no rule.
  - `p_57_exact` enters only the per-seed fallback, which only seed 1's A reaches. There the three variants share one bit-identical value on each platform (stage one note), so the delta is exactly 0 on both.
  - Every other failing seed fails on a season drop beyond 0.3pp, before P(57) is consulted, and every passing seed passes on condition 1, which does not use P(57).
- **`tables.py`** rebuilds the P@1/hit, §5 and secondary-metric tables from the copies and prints the in-process cost totals. P@1 and hits come from the profiles' rank-1 rows; the rule and §5 use its own arithmetic over the retained diffs. Its 53 cross-checks pass (`tables.out`), covering profile/diff P@1 agreement, selected box-aggregate fields and the aggregate's in-process CPU total.
  - **Checker limits:** it does not rebuild the run/guard table or compare the aggregate's per-seed delta maps. Its seed-level d check uses `zip` without a length check, so an empty or shortened aggregate array can pass. The secondary table comes from retained diffs and is not compared with the aggregate, which carries no secondary metrics. These passing cross-checks are partial evidence; acceptance also checks array lengths, per-seed maps and retained scorecard/diff reconciliation.

## Cost
- **Stage two (seeds 4–10):** 332,658.1 CPU-s = **92.41 CPU-hours** (guard; the ledger's journal figures sum to 92.4051). The in-process measurement is 332,376.2 s = 92.33 h.
  - Per seed: 13.08 to 13.41 CPU-h, a mean of 13.20. No stage-two seed ran during production's 03:00 chain: seed 6 ended at 01:33, and seed 7 launched at 03:11 after the lead saw the chain's blend written (03:05:15).
  - Seed 8 was the costliest (13.41 h, mostly in B 2025). The manager recorded that as an observation, not a finding.
  - Every first walk-forward stayed under the 7.5 CPU-h stop (1.98 to 2.12 CPU-h).
- **All ten seeds:** 493,871.5 CPU-s = 137.19 CPU-hours (guard). `aggregate-stage-two`'s `total_cpu_h`, 137.0998, is the sum of the ten `results.json` `total_cpu_s` values, the in-process measurement.
- **The cap.** The shared C1 + C2 ledger after seed 10 is **137.6679** CPU-h over 27 jobs, under the 165 cap of Eric's row C2-framing-stage-two-cap. The 50 checkpoint was acknowledged before seed 4 (row C2-framing-checkpoint-50).
- **The off-launcher aggregates.**
  - Stage two's `aggregate-stage-two` ran outside the launcher: user 60.624 s + sys 0.341 s = 60.97 CPU-s = 0.0169 CPU-h. The manager accepted the off-launcher row (11:04:30); as with stage one's 0.0053, the C2 index row on main is the record, with no box ledger write.
  - The effective shared total is therefore **137.6901** CPU-h (137.6679 + 0.0053 + 0.0169), leaving 27.31 under the cap. The box's `compute_ledger.tsv` undercounts it by 0.0222 CPU-h, and a reconciliation between the two should not read that gap as a problem.
- **Mac CPU:** the Mac re-run, the rescore diagnostic and `tables.py` used Mac CPU only.

## What follows
1. **Independent acceptance** of this note and the run artifacts, by a fresh Codex session that has not reviewed this item, Part 1 blind. By the manager's condition, only a plain SIGN lets the result go to Eric; anything else goes to the manager first.
2. **Then the result goes to Eric in the lead's pane, plain language first,** with the ten-seed dispositions and the 6-of-10 rule in words, and then to the manager. (The addendum named job-search-52 as the channel; that session ended on 2026-10-08, and the manager set this route.)
3. **What a positive can support:** at most "worth testing on untouched 2027 data". A 2027 test would need a pregame catcher identity, specified independently (Limits). Whether to pursue it is Eric's call; nothing here asks for or approves a production change, a deploy or a further run.

## Limits (from the pre-registration and the addendum)
- **Seasons:** 2024 and 2025 are consumed seasons. Ten seeds measure the algorithm's seed sensitivity, not new season samples.
- **The seeds:** they are positions 0–9 of `canonical-n10.json`, which comes from an outcome-ranked, stratified historical baseline distribution: neither an outcome-independent random sample nor full-range coverage. Five of the seven stage-two seeds exceed 2³¹−1, and LightGBM 4.6.0 reads them with 32-bit wraparound; the seven stay distinct after wraparound (addendum §2).
- **Catcher identity** is a postgame, game-level proxy: one catcher per side per game, which need not be the starter. A 2027 test needs a pregame catcher identity, specified independently.
- **History:** there is no catcher history before 2019; the pitcher feature has it.
- **Event availability:** feature history keeps resumed-portion PAs at the original official game date. Date-level shift(1) can therefore admit events before they happened. Baseline and both variants share this convention; the screen does not establish unconditional pregame availability (pre-registration §8).
- **Labels:** the run rebuilds labels through `filter_out_resumed_portion` and drops void profile rows. All sixty walk-forwards report zero label changes and zero void rows. The empty resumed-row record cannot tell "no flagged rows" from "no flag column"; these copies do not independently certify original-portion label correctness.
- **Multiplicity and sequencing:** A and B are reported separately, with no multiplicity adjustment, and stage two's continuation on stage one's inconclusive result is not adjusted for.

## Evidence
`docs/audit/2026-10-09-c2-framing-stage-two-evidence/`:
- `aggregate.box.json`, `aggregate.box.stderr.txt`: the box aggregate's output and its timing.
- `runs.sha256`: the 105 files of the seven stage-two run directories, hashed on the box (stage one's 45 are in `docs/audit/2026-10-08-c2-framing-stage-one-evidence/runs.sha256`).
- `aggregate.mac.out`: the Mac re-run's refusal.
- `rescore_diff.out`: the stage-two scorecards recomputed on the Mac, with every differing field (script: stage one's `rescore_diff.py`).
- `tables.py`, `tables.out`: this note's profile and derived tables, with selected cross-checks against the box aggregate (see the checker limits above).
