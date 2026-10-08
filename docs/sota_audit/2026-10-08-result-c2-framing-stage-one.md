# Framing screen, stage one: result (C2 side item (e))

**What this is:** the results note for stage one of the catcher-grouped framing screen, pre-registration `docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md` (frozen at the reviewed commit; review r10 plain SIGN, `docs/audit/2026-10-08-c2-framing-codex-r10.md`).
- **The question:** does grouping production's framing proxy by the catcher (`fielding_catcher_id`) instead of the pitcher improve the model's pick ranking?
  - Variant **A** replaces `pitcher_catcher_framing` with the catcher-grouped measure.
  - Variant **B** adds it alongside.
  - Both are scored against the same seed's baseline on 2024 and 2025.
- **What it can support** (pre-registration, "What it can support"): at most "worth testing on untouched 2027 data", never "ship it". 2024 and 2025 are already consumed seasons. The three seeds measure sensitivity to the algorithm's seed; they are not three new season samples.

**Status:** all three registered seeds ran; `aggregate` passed. **Independent acceptance: pending.** Per pre-registration §7, nothing in this note goes to Eric or job-search-52 before the acceptance. Nothing here approves a production change, a deploy or stage two.

## Reading record
Who read what, and when (all 2026-10-08, EDT):
- **Until 14:27:11 EDT, only costs and label counts were read:**
  - the `cpu_s`, `wall_s` and `labels` fields of units.json;
  - the progress lines of the log tails;
  - the launcher's records.
- **First opening.** bts-lead2 (the BTS lead, herdr pane wA:pQ) first opened the outcomes at 14:27 EDT, when it printed `aggregate.stdout.json` (file mtime 14:27:11). No one else had read them.
- **Copies.** The three run directories were then copied to the Mac: 45 files, with sha256 matching the box. They were read after that only by the lead, for this note's tables and the cross-platform diagnosis below.
- **Not sent.** Nothing has gone to job-search-52 or Eric. Per §7, no outcome goes to them before the independent acceptance.

## The runs
All three were launched through the C1 launcher from `~/projects/bts-c1`, at nice 10, with `BTS_LGBM_DETERMINISTIC=1`. Each ran six walk-forwards: baseline, A and B, for 2024 and 2025. Each walk-forward was `estimated_pa`, retrained every 7 days, and kept top-10 profiles.

| Seed | Unit | Run (claim) | Code | Ran (EDT) | CPU (guard / ledger) |
|---|---|---|---|---|---|
| 2273360 | `c1-c2-framing-seed1-20261008T061510Z-ad237918` | `22d31f2-20261008T061514Z` | `22d31f2` | 02:15:10 → 05:21:53 | 65,620.4 s = 18.23 h |
| 260991262 | `c1-c2-framing-seed2-20261008T134258Z-6b2aae98` | `11e0cfd-20261008T134306Z` | `11e0cfd` | 09:42:58 → 11:37:07 | 48,017.4 s = 13.34 h |
| 1746737973 | `c1-c2-framing-seed3-20261008T163012Z-650ee8e4` | `11e0cfd-20261008T163019Z` | `11e0cfd` | 12:30:12 → 14:23:18 | 47,575.6 s = 13.22 h |

- **Exit and records.** Every unit exited rc 0, and its guard records `exit` and RECONCILED with no problems. Every walk-forward reports labels changed 0 and void 0.
- **Code.** `11e0cfd` adds only Eric's release row to the register: it is a metadata-only descendant of `22d31f2`. `validate_run`'s descent-and-closure rule admitted both commits.
- **Seed 1's slow walk-forward.** Seed 1's A 2024 walk-forward cost 7.04 CPU-h against about 2 for its peers, because it collided with production's 03:00 `bts preview` (C2 index, row (e)).
  - The collision is observed. That it changed nothing but cost is inferred: LightGBM ran with `deterministic=True` and `force_row_wise=True`, with no thread count among the manifest's recorded parameters, so the thread count does not depend on load. This is not tested.
- **Output root:** `data/hetzner_results/c2/framing_screen/seed_<seed>/<run>/` on the box (restic-backed archive set).

## Result
**Both variants are inconclusive.** Neither is positive, so neither reaches "worth testing on untouched 2027 data".

**P@1 by seed** (top-ranked pick hit rate; 185 test days in 2024, 184 in 2025; hits in brackets):

| Seed | Season | Baseline | A | A − baseline | B | B − baseline |
|---|---|---|---|---|---|---|
| 2273360 | 2024 | 78.9% (146) | 78.9% (146) | **0.00pp** (0) | 77.3% (143) | **−1.62pp** (−3) |
| 2273360 | 2025 | 67.4% (124) | 70.1% (129) | **+2.72pp** (+5) | 71.7% (132) | **+4.35pp** (+8) |
| 260991262 | 2024 | 74.6% (138) | 77.8% (144) | **+3.24pp** (+6) | 76.8% (142) | **+2.16pp** (+4) |
| 260991262 | 2025 | 66.8% (123) | 74.5% (137) | **+7.61pp** (+14) | 69.0% (127) | **+2.17pp** (+4) |
| 1746737973 | 2024 | 77.8% (144) | 76.8% (142) | **−1.08pp** (−2) | 76.2% (141) | **−1.62pp** (−3) |
| 1746737973 | 2025 | 67.9% (125) | 71.7% (132) | **+3.80pp** (+7) | 72.3% (133) | **+4.35pp** (+8) |

**The §5 quantities** (from `aggregate`):

| | A | B |
|---|---|---|
| Mean 2024 delta | +0.72pp | **−0.36pp** |
| Mean 2025 delta | +4.71pp | +3.62pp |
| Seed-level d (seeds 1, 2, 3) | +1.36, +5.43, +1.36pp | +1.36, +2.17, +1.36pp |
| m (mean of d) | +2.72pp | +1.63pp |
| t = m / (sd/√3) | 2.00 | 6.08 |
| Per-seed rule passed | **1 of 3** (seed 2) | **1 of 3** (seed 2) |
| **Disposition** | **inconclusive** | **inconclusive** |

**Why:**
- **A** meets three of the four positive conditions: both season means are above 0, m ≥ +0.3pp, and t ≥ 1.5. It fails the fourth: the per-seed rule holds on only 1 seed, not the 2 required. It is not negative because m > 0.
- **B** fails two positive conditions: its 2024 mean is below 0, and the per-seed rule holds on only 1 seed. It is not negative because m > 0.

**The per-seed rule** (`bts.experiment.runner.evaluate_pass_fail`) passes a seed when either:
1. P@1 improves in both seasons; or
2. the neutral fallback holds: no season's P@1 drops more than 0.3pp (delta ≥ −0.003), `mean_max_streak` ≥ 0, and exact P(57) strictly improves.

Seed by seed:
- **Seed 2** passes for both variants on condition 1.
- **Seeds 1 and 3, B:** each has a 2024 drop of 1.62pp, so they fail both conditions.
- **Seed 3, A:** the 2024 drop is 1.08pp, so it fails both conditions.
- **Seed 1, A:** 2024 is flat (0.00pp), so it fails condition 1 but reaches the fallback. There, `mean_max_streak` is +0.42, but the exact-P(57) delta is exactly 0. The three variants' exact P(57) values for that seed are equal to the last bit (7.742570144719881e-12), so P(57) does not strictly improve and the seed fails.
- **Wording gap.** Pre-registration §5 paraphrases the fallback as "P@1 within 0.3pp". The code that ran (reviewed) means "no drop beyond 0.3pp". Only seed 1's A is affected: under the "within" reading it fails one step earlier, on 2025's +2.72pp. The verdict is the same under both readings.

**Secondary metrics** (decision weight only through the per-seed rule; from the retained diffs):

| Seed | A: mean_max_streak Δ | A: exact P(57) Δ | B: mean_max_streak Δ | B: exact P(57) Δ |
|---|---|---|---|---|
| 2273360 | +0.42 | 0 | +0.74 | 0 |
| 260991262 | −0.40 | −1.16e-08 | −1.58 | −1.16e-08 |
| 1746737973 | −3.55 | −5.29e-09 | −1.35 | −5.51e-09 |

## Things to weigh in the result
- **The seasons disagree.** Every seed improves 2025 under both variants, by +2.2 to +7.6pp. 2024 is mixed: A gives 0.00, +3.24 and −1.08pp; B gives −1.62, +2.16 and −1.62pp.
- **The seed noise is as large as the deltas.** The baseline's own P@1 spans 74.6–78.9% in 2024 across the three seeds.
- **B's t of 6.08 comes from a coincidence of whole hit counts, not from consistency.**
  - Seeds 1 and 3 give B the same deltas: −3 of 185 in 2024 and +8 of 184 in 2025.
  - The underlying counts differ: B 143 and 132 hits against a baseline of 146 and 124, versus B 141 and 133 against 144 and 125.
  - Two of the three seed-level values are therefore identical, so the standard deviation is small (0.47pp) and t is large.
  - With 3 seeds, t ≥ 1.5 is a screening convention, not a significance test (pre-registration §5).
- **The streak metrics point the other way.**
  - On seeds 2 and 3, exact P(57) falls for both variants.
  - `mean_max_streak` falls on seeds 2 and 3, by up to 3.55 for A on seed 3.
  - These carry no decision weight beyond the per-seed rule. They are reported because they do not move in the same direction as P@1.

## Reconciliation
- **On the box, it passes.** `aggregate` ran on the box from `~/projects/bts-c1` at `11e0cfd` (clean) at 14:26:52–14:27:11 EDT, with rc 0. It ran under the passing admission gate and validated all three runs: claim, manifest, pins, identity, code descent and closure, season evidence, and every scorecard, diff and summary recomputed from the retained profiles with exact equality.
  - Its output is `aggregate.box.json` (sha256 `8d02d0bc…`), in `docs/audit/2026-10-08-c2-framing-stage-one-evidence/`.
- **On the Mac, exact equality fails.** Re-running it there on the hash-matched copies, with the copies as the namespace root, refuses seed 1's baseline scorecard: the stored value does not equal the Mac's recomputation.
  - **The cause, measured** (`rescore_diff.py`, output `rescore_diff.out`): over the nine scorecards, 17 values differ, and only two fields: `p_57_exact` and `p_57_mdp`. Every difference is in the last digits, with relative size at most 5.85e-16. Every other field is equal, including every P@1, streak metric, calibration and precision value.
  - **The platforms:** the box is x86_64 with Python 3.12.6; the Mac is arm64 with Python 3.12.13. Both have numpy 2.4.3 and pandas 3.0.1.
  - **Mechanism:** platform floating-point differences, inferred and not tested.
- **No decision changes.**
  - `p_57_mdp` enters no rule.
  - `p_57_exact` enters only seed 1's A fallback, through its delta. On each platform, seed 1's three variants share one bit-identical value (box 7.742570144719881e-12, Mac 7.74257014471988e-12), so the delta is exactly 0 on both.
- **What this means for later checks.** The exact-equality reconciliation is a check on the producing platform, and it passed there. A check on another platform needs a tolerance for these two fields.

## Cost
- **Stage one:** 161,213.5 CPU-s = **44.78 CPU-hours** (guard and ledger). The in-process measurement is 161,183.2 s = 44.77 h; `aggregate`'s `total_cpu_h` is the sum of `results.json`'s `total_cpu_s`.
- **Per seed:** 18.23, 13.34 and 13.22 CPU-h. Seed 1 includes about 5 CPU-h of contention from the 03:00 collision. Seeds 2–3 ran uncontended; their walk-forwards cost 1.98–2.42 CPU-h each.
- **The 50 checkpoint.** The shared C1 + C2 ledger after seed 3 is **45.2628** CPU-h over 20 jobs, so it never came into play, and no acknowledgement was needed or given.
- **The off-launcher aggregate.**
  - The box `aggregate` ran outside the launcher: 19.04 CPU-s = 0.0053 CPU-h, which brings the effective shared total to **45.2681**. An earlier attempt at 14:26:45 exited 127 before Python started.
  - By the manager's ruling, the C2 index row on main is the record for it. The box's `compute_ledger.tsv` therefore undercounts the shared total by 0.0053 CPU-h, and a reconciliation between the two should not read that gap as a problem.
- **Mac CPU:** the Mac re-run and the diagnostics used Mac CPU only.

## What follows
1. **Independent acceptance** of this note and the run artifacts, by a fresh Codex session that has not reviewed this item.
2. **Then the result goes to Eric through job-search-52** (§7): per-seed and mean P@1 deltas per season for A and B, the dispositions and the measured cost. The fallback is `~/projects/job-search/mets-2026-10-06/catcher-experiment-result.md`, with the herdr manager told.
3. **Stage two is Eric's call.** Under §5, only an inconclusive variant could justify stage two (to 10 seeds), and both variants are inconclusive. Stage two needs his second go-ahead; nothing here asks for it or approves it.
   - **The cost arithmetic** (projection from the two uncontended seeds, n = 2): 7 more seeds at about 13.2–13.3 CPU-h each come to about 93 CPU-h. Only about 54.7 remain under the 100 cap, and C2's planned jobs need about 11.1 CPU-h of it.
   - Stage two therefore does not fit under the current cap, and it would also need the 50 acknowledgement. Raising the cap is a separate decision, never made here (§6).
4. **The Mets application stays held** until Eric has this result.

## Limits (from the pre-registration)
- **Seasons:** 2024 and 2025 are consumed seasons. Three seeds measure the algorithm's seed sensitivity, not new season samples.
- **Catcher identity** is a postgame, game-level proxy: one catcher per side per game, which need not be the starter. A 2027 test needs a pregame catcher identity, specified independently.
- **History:** there is no catcher history before 2019; the pitcher feature has it.
- **Multiplicity:** A and B are reported separately, with no multiplicity adjustment.
- **The seeds:** they are positions 0–2 of `canonical-n10.json`, which is neither an outcome-independent random sample nor full-range coverage.
