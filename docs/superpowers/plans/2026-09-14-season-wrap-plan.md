# BTS 2026 Season Wrap Plan

**Date:** 2026-09-14 · **Status:** DRAFT v2.1 — awaiting owner approval · Codex r3 verdict: **SIGN WITH EDITS**, both minimum edits applied verbatim · **Owner:** Eric
**Purpose:** close the 2026 season honestly, analyze what the field did, refresh the literature review, and leave 2027 with a small set of pre-registered, killable improvement candidates.

**Review provenance.** Codex (`gpt-6-astra`, xhigh), 2026-09-14, via herdr:
- r1 attacked the outline (`.codex-review/season-wrap/r1-codex.md`, gitignored). Structural edits from it are marked **[r1]**.
- r2 reviewed DRAFT v1 of this document: **sign-off withheld** pending six blocking edits (capture contract, dependency graph, snapshot/retention, continuation semantics, rank-4 kill criteria, definition of done). All six plus the non-blocking edits are applied in v2 and marked **[r2]**.
- r3 narrow verification of v2 (`r3-codex.md`): five of six blocking edits APPLIED, one PARTIAL (retention wording), plus one new defect (surface E as a universal gate). Verdict **SIGN WITH EDITS**; the two minimum edits are applied in v2.1 and marked **[r3]**. Codex's scope statement, verbatim: "This is approval of the plan's scope and gates, not certification of September measurements, completed implementation, or authorization to spend, commit, deploy, or activate production."
Codex verified the repo code/docs it cites and I independently re-verified every citation and the five literature URLs. Codex did **not** reproduce the September production numbers; those remain brief-supplied facts from the 9/11 read-only scorecard until W1.1 rebuilds them. Two review rounds describe process history; they do not authorize production changes.

---

## 0. Where the season ended

| Fact | Value | Source |
|---|---|---|
| Season declared over by owner | 2026-09-14 | this session |
| Contest season-best streak | **18** | `data/picks/account_state/contest_streak.json` (MLB profile, trusted) |
| Contest streak at close | 0 (9/12 and 9/13 doubles missed) | same + `decision.json` 9/12–9/13 |
| Streak saver | used 2026-08-27 | `account_state/saver_state.json` |
| Game dates remaining | 14 (9/14–9/27) | scheduler plan |
| Box state | private mode, DMs off, nag cron disabled; picks still computed + graded; shadows continue | `~/.bts-orchestrator.toml`, crontab (snapshots `*.bak-20260914-season-over`) |
| Tail policy | stops computing picks once contest streak 0 with ≤9 dates left → **9/19** unless D8 changes it | `bts.simulate.tail_policy` stop rule |
| Headline live numbers (brief-supplied, 9/11 scorecard) | primaries 96/141 = 68.1% full season; 79/109 = 72.5% current recipe; DD legs 58/82 = 70.7%; mean stated primary p 0.765 | `bts_index` 2026-09-11 entry — **rebuilt in W1.1** |
| Skip-policy shadow | 15 resolved divergent days, skipped band 86.7% CI[62.1, 96.3] vs 0.744 breakeven, `insufficient_n` | `bts skip-policy-shadow-status` 9/14 |
| Context-stack shadow v2 | prod 55.6% vs shadow 55.6%, n=54 paired | `bts shadow-status` 9/14 |
| Naive count of graded pick files on box | 191 primaries / 157 DD legs | this session — a **different population** from the scorecard's 141/82 (undelivered/preview files were graded locally); W1.1 reconciles the two with reasons, it does not reproduce 141 |

---

## 1. Ground rules for the whole wrap **[r1]**

1. **Four records, never merged:** (a) *contest record* — what Eric entered and MLB settled; (b) *delivered-recommendation record* — what the system locked/delivered (incl. late/unknown/undelivered); (c) *forecast record* — probability quality on timestamped selectable candidates; (d) *policy record* — actions and full counterfactual trajectories under a frozen rule. A DM is not a contest entry. Post-9/14 private picks are research observations only; best = 18 stays the contest fact.
2. **2026 is not an untouched lockbox.** 2026 outcomes were already consumed by feature screens (park drag 7/08), calibration reads, the M3 audit, the June/July policy investigations, the 8/09 tripwire interim look and this plan's own headline. Before any further outcome read, freeze an **analysis-exposure register** (candidate · freeze date · fitting dates · outcomes already examined · protocol · allowed next look). New September protocols cannot retro-pre-register analyses suggested by observed 2026 results; where that is the case, 2027 forward evidence is the final gate.
3. **Backtest 86% and live 68–72% are not comparable estimates.** The historical path ranks after compounding over *realized* PA rows (`_actual_pa_game_predictions`); even the `estimated_pa` diagnostic still uses actual participants/starter matchups. The gap is an information-set mismatch **before** it is evidence of model deterioration. W1.2 builds the bridge; nothing in W4 is justified by the raw gap.
4. **Calibration is a diagnosis, not the first remedy.** A monotone recalibration cannot improve ranking, and the MDP bins on empirical quantiles, so a consistently-remapped map changes displayed p while leaving policy behavior essentially unchanged. Recalibrating against *old* boundaries changes policy behavior and must be evaluated as a policy change.
5. **No universal "24 seeds + replay" gate.** Seeds reuse the same baseball outcomes; five seasons are not 120. Realized-sequence replay is required for policy comparisons but cannot resolve P(57). Ops fixes need failure-path fixtures, not seeds. P(57) is reported as a **model-dependent projection** with dependence/calibration stress scenarios; robust lower-milestone improvement is evidence, not proof.
6. **Closed-work registry, not a word blacklist.** Prior rejections are recorded with the layer/basis they were tested on (April's validation used the optimistic actual-PA surface; June's "near ceiling" is broader than its tested mechanisms). Reopen only for a specific corrected information set, a new signal, or a newly met trigger.
7. **Owner's standing rules still apply:** no time estimates; label every claim backtested (n, result) or *reasoning only*; estimated_pa never actual_pa for policy work; Codex reviews design before build and code after; **execution requirements are not owner decisions** — preserving original protocols, valid statistics, the existing capture cap and verified backups are simply done. **[r2]**

---

## W0-NOW — Preserve and freeze (start immediately; nothing here waits on a decision) **[r2: split from scheduled work]**

| # | Item | Why | Done when |
|---|---|---|---|
| W0.1 | **Back up the estimated-PA 24-seed profile run** (`data/hetzner_results/mdp_estpa_run`, 7.1 MB, exists on this Mac ONLY): copy to the box under `data/hetzner_results/` (already inside the restic `archive` set), hash manifest both sides | The shipped policy's own profiles were lost with the 6/05 MacBook; this is the only valid re-solve basis left | sha256 manifest matches on Mac, box and a restic **restore into an empty directory**; the pinned diagnostic below reproduces its recorded output |
| W0.1b | **Pinned restore diagnostic:** `scripts/audit/confirm_mdp_policy_replay.py` with inputs fixed — profiles dir, base policy `mdp_policy.npz` (sha 66d15471…, read from `data/models/`, i.e. outside the profile dir), tail `mdp_tail_policy.npz` (sha dc5d0c99…), seeds, seasons — and its expected output hash recorded | A clean reader must reproduce a *named* old diagnostic; the script's known replay shortcuts are acceptable for reproducing that diagnostic, not for new policy acceptance **[r2]** | runbook + expected-output hash committed with the manifest |
| W0.2a | **Preliminary evidence checkpoint (today):** box journal export since 9/01, `~/.bts-orchestrator.toml` + its 9/14 snapshot, crontab + snapshot, all `data/picks/**` incl. `decision.json`/shadow files/`account_state/*`, `data/health_state/**`, **all of `data/validation/**`** (status JSONs *and* candidate manifests/ranked surfaces needed for the #16 and skip/DD reads), `data/models/*.npz`, deployed SHAs per day, `uv.lock` | Today's configuration/journal evidence is otherwise lost to log rotation and rolling retention; `data/validation` is in **no** restic set (`BACKUP_SETS`: ops = picks+health_state, archive = leaderboard+hetzner_results+external) **[r2]** | staged copy under `data/hetzner_results/season_2026_snapshot/prelim-20260914/` with sha256 manifest; restic snapshot ID recorded |
| W0.3 | **Analysis-exposure register** (`docs/audit/2026-09-15-exposure-register.md`) covering every 2026 outcome read to date: park-drag screen, M3, DD tripwire interim looks (8/09 n=57), skip-shadow displays, 9/11 scorecard, 7/03 me-vs-leaderboard, this plan | Rule 1.2 | each entry has candidate/freeze/fitted/examined/protocol/next-look; every later read cites an entry |
| W0.4 | **Protocol inventory for the due reads** (W1.4) — locate and quote the original protocol text for DD tripwire, skip shadow, context shadow, #16, tail | Rule 1.2; reads run under ORIGINAL protocols | protocols quoted verbatim in the register |
| W0.5 | **Owner decisions needed before the calendar bites:** **D3** (2026 data use) before any new outcome exploration; **D5** recorded as its already-authorized default (final-leader allocation, Eric 9/14) unless Eric changes it; **D8** (research capture past 9/19) settled before any continuation change is built **[r2]** | D3 governs which reads may run; D8 has a 9/19 deadline | decisions recorded in the register with date |

## W0-SCHEDULED — Capture and final snapshot (calendar-bound) **[r2]**

### W0.6 Bounded end-of-season grab (~9/28; ONE authenticated operation, owner-authorized 9/14)
**Verified constraints on the existing tooling:** `bts leaderboard scrape` has no dry-run; it defaults to deep paging (`--deep-max-pages 100`, `--deep-limit 300`, `--deep-min-streak 3`), fills `--profile-top-n 300` by walking active-streak users first and deduplicating by username, and prints `scrape complete` even after `run()` swallowed a rate-limit abort. `backfill` fetches ONE profile response (no pagination). Neither implements a final-tab / early-cohort allocation. So:
1. **Reviewed one-shot runbook or thin wrapper** (Codex code review before it runs) specifying: tab names to capture; **a full-depth walk of `all_season` (board rows only, limit 300, until a short page; ~190 requests; `allParticipantsCount` recorded) — added 9/14 after Eric's correction, this is what makes the whole-field season-best distribution identifiable**; the `active_streak` deep walk only if still wanted (default OFF); profile allocation per D5 (see below); exact distinct-user cap (300 by `user_id`, not username); page/request ceilings per endpoint; raw HTTP response retention (gzipped bodies + timestamps + status) *before* parsing; per-request success/partial/abort status file.
2. **Global 403/429 stop across the whole operation**, including the backfill loop; no automatic retry; **no second authenticated attempt** without a fresh owner decision.
3. **Offline fixture rehearsal** (recorded responses) is the only rehearsal; no live dry run; the CLI's completion text is not acceptance evidence.
4. Visible-history coverage per user is an **outcome recorded** (first/last visible date); incomplete history stays incomplete — never assumed March–September.
5. Acceptance = status file shows every planned request accounted for (success/partial/abort), raw bodies hashed, cohort manifest written.

### W0.7 Final immutable snapshot (after the grab attempt AND a declared grading/reconciliation cutoff)
- Declare the cutoff date/time; list unresolved items (pending results, corrections) rather than waiting for them.
- Stage a copy (or verified retry-on-change) of: everything in W0.2a refreshed, plus `data/leaderboard/**` (deep corpus + static captures + grab output), policy artifacts, pinned code SHA + `uv.lock`, configuration history; sha256 per file; capture boundaries and correction-version rules stated.
- Store under `data/hetzner_results/season_2026_snapshot/final-<date>/` (inside the `archive` set) **and** create a dedicated restic backup with **only** the `season2026` tag; record its exact snapshot ID. Do not apply a `season2026` forget policy before the 2027 activation review. Verify, using non-destructive retention dry runs, that every scheduled forget selector (`forget --tag ops`, `forget --tag archive` in `backup.py`) excludes the protected snapshot IDs — tag isolation, not a snapshot count, is what protects it. Keep the staged evidence and perform the specified **empty-directory restore + hash comparison** as the acceptance test. **[r3]**
- Mark unavailable history explicitly (e.g. pre-6/22 decision files, pre-5/26 feature-env hashes).

### W0.8 Research-capture continuation — **D8 = continue (Eric 9/14); BUILT 9/14** **[r2]**
Continuation is a **separate forecast/research capture stream** that preserves the official tail decision, contest state, delivery settings and every original protocol's eligibility rules. It does not override the tail stop, does not touch `decision.json` semantics, and adds **no** eligible skip-policy observations (`record_skip_from_decision` requires `is_reach57_mdp_skip`; tail-objective skips are excluded by construction). What it buys: more diagnostic forecast observations for W1.2/W2.3, not extra shadow trials.

**Implementation (Codex design review `d8-codex.md` → 2 BLOCK + 3 SHOULD adopted; code review 3 rounds → SIGN; DEPLOYED `f1b07e9` 2026-09-14 21:39 ET inside the idle window, canary green; unit reinstalled with the flag; rehearsal: real skip day 8/23 → `research_deadline_passed` with nothing written, isolated fixture root → genuine 10-row export + valid sidecar + idempotent restart, production picks tree hash unchanged):** `scripts/live_forward_capture_once.py --capture-research-on-skip` (flag added to the repo unit `scripts/systemd/bts-live-forward-capture.service`). On a day with no pick file, if `scheduler_state.json` carries a CURRENT skip intent (`final_skip_candidate`; `skip_notified_at` alone is ignored because a pick attempt clears only the former) and `decision.json` is not scoreable, the wrapper exports the ranked slates into `data/validation/decision_weighted_lgbm_v0_live_forward_research/<date>/`, verifies them without the pick-snapshot requirement, and writes the sidecar `research_capture.json` **last, atomically, with sha256 of every contract file** — that sidecar is the only thing that makes a capture count. Guards: root-overlap refusal; all reads side-effect-free; **start before first pitch − 5 min, complete before first pitch** (else discarded to `<date>.late.*`); fail closed without game times; partial/corrupt dirs quarantined to `<date>.partial.*` and redone only pregame. Resolution of this stream is offline (W1.1 joins outcomes by `batter_id, game_pk`); the trigger is labelled provisional and the final day classification is reconciled in the ledger. 34 tests (`tests/scripts/test_live_forward_capture_once.py`); two-phase acceptance marker `research_capture.accepted.json` (published_at sampled after the sidecar rename) is required for a capture to count. This branch covers the **11 historical skip days** that the official stream missed, going forward only.

---

## W1 — Season ledger, due reads, benchmark bridge **[r2: due DD read moved ahead of the bridge]**

### W1.1 Canonical ledger (the fact base)
- **Keys:** ET contest date · round/unit id where available · game_pk · batter_id · slot (primary/DD) · decision revision + objective. Append-only source observations → one reconciled view with explicit **conflict** and **missingness** flags.
- **Fields:** prediction/lock/delivery/entry/outcome timestamps; attempt vs confirmed delivery; game eligibility; void/pending/miss; projected vs confirmed lineup; stated p per slot; snapshot ref for the full selectable slate; contest streak before/after (contest ledger as-of); saver state.
- **Recipe epochs** (not per-day model hashes): deployed code SHA, feature schema/env hash (from 5/26), blend/calibration config, PA aggregation, training cutoff/retrain rule, determinism settings, base/tail policy hashes + boundaries. Pre-fingerprint rows = `epoch_unknown`, never silently the current recipe. The canonicalizer's `post_bpm` label is one partition, not a registry. **[r1]**
- **Scoring contract:** per-slot results first; legacy single-day fallback only where justified; doubleheaders resolved by game; suspended-game scoring per `filter_out_resumed_portion` (separate from training PA).
- **Sources:** `data/picks/<date>.json` + `<date>/decision.json` (v1–v3) + `contest_ledger.jsonl` + journal delivery lines + `late_delivery`/`refused_delivery_*` archives. Absence of `decision.json` before 6/22 is not a skip.
- **Acceptance:** a **reconciliation table** between the 191/157 file tally and the 141/82 scorecard — inclusion/exclusion reason per row, date windows matched, unresolved differences listed. Not "reproduce 141". **[r2]** Fixtures: undelivered-but-graded, preview overwritten, DD leg void, doubleheader, private_locked, tail-period.
- **Minimum-viable first pass:** enough reconciliation to identify the DD legs and their inclusion status for W1.4a — the bridge does not gate the due read. **[r2]**
- **Outputs:** `data/validation/season_2026_ledger.parquet`, `scripts/audit/build_season_ledger.py`, `tests/scripts/test_build_season_ledger.py`.

### W1.4a Due read that is already late: DD tripwire (runs right after the minimum ledger) **[r2 reorder]**
Rule: first snapshot with n ≥ 80 legs, that snapshot's frozen inclusion/grading recipe, ≥10pp shortfall vs stated mean p; an escalation threshold, not a ship authorization. Recover *that* snapshot (n crossed 80 by 9/11). If the original snapshot cannot be recovered, **record the protocol deviation** and produce a separately labeled reconstruction; do not manufacture a formal historical read. The season-end update is a second, separately labeled read.

### W1.2 Benchmark bridge (bounded; NOT a point-in-time platform; surfaces A–D) **[r1][r2: E moved to W4]**
| Surface | What it is | Question |
|---|---|---|
| A | Legacy actual-PA walk-forward (README 86%) | What did the headline measure? Never a comparator. |
| B | Estimated-PA participant diagnostic (`mdp_estpa_run` basis) | How much optimism is exposure/aggregation, before any drift? |
| C | At-cutoff reconstruction on reconstructable days (available slate, projected/confirmed lineups, probable starters, lookup age, eligible-game rules) | What could this recipe have predicted/selected then? |
| D | Archived production (served scores + final action) | Does C reproduce the real serving path? |

- Per comparison: **same-batter paired score changes** *and* **reranked decision changes**; candidate overlap, changed picks, discordant-outcome days, coverage exclusions, top-1 hit rate, within-slate discrimination, Brier/log loss, mean stated−realized residual. Unreconstructable dates are a reported missing fraction.
- Separate *count-only* from *starter/reliever-context* changes where feasible; anything using actual counts/participation is an **oracle diagnostic**.
- Separate "frozen coefficients trained through 2025" from "frozen daily-learning recipe" (the walk-forward loop appends earlier test-season dates to training). Neither is a pristine 2026 holdout (rule 1.2).
- Scope guard: reuse live-forward/production snapshots + existing M3/backtest tooling; July's decision against a broad PIT platform stands. A partly-unexplained gap is an acceptable result.
- Prior results carried, not re-derived: May PA-basis memo (rank-1 realized PA 5.500 vs estimated 4.429); M3 39 discordant days, −0.67pp CI[−3.4, +2.0], held; June 29 PA-tilt paired 146 vs 143, p=0.91 (no standalone PA-volume lever); May gate-B `OUTCOME_MIXED_HEADLINE_FAILS_STABILITY_BAR` for the boundary-only candidate.

### W1.3 Distinguishing the explanations (pre-declared tests) **[r1]**
| Explanation | Evidence sought | What weakens it |
|---|---|---|
| Benchmark optimism | A→B→C reduction; reranking; realized-PA distribution of selected players | similar gap surviving C vs D |
| Serving drift / stale inputs | paired saved-vs-reconstructed scores; feature age, lineup/pitcher changes, recipe mismatches | good parity with persistent residuals; M3's null |
| PA-opportunity forecast error | at-lock vs realized PA distributions (total/starter/reliever), out-of-time count scoring, hit-score change with PA model fixed | actual-N oracle lift is not achievable lift |
| Conditional hit-model drift | stable parity but shifted PA-/game-level residuals under a fixed learning recipe | one evolving season |
| Calibration without ranking loss | poor held-out proper scores/intercept/slope with stable within-slate ordering | better Brier ≠ better top-1 or policy value |
| Ball regime | league + selected-player residual co-movement vs independently dated drag regimes (5/24 change-point, park rollout, late-July reversion) | change-points chosen to maximize our misses; confounds |
| Selection/composition | within-day vs across-day discrimination; rank-1 vs all candidates; played vs skipped; slate size, lineup status | mean-p shift may be availability |
| Sampling variation | paired date-block intervals, discordant-day counts, matched-probability null | "not significant" ≠ luck; cold month ≠ drift |

Descriptive Wilson intervals on the brief numbers (60–75% for 96/141; 63–80% for 79/109) show why "72.5% vs a serving-realistic mid-70s" is a different question from "68% vs 86%".

### W1.4b Remaining due reads — under their ORIGINAL protocols (after the register + protocol inventory) **[r1]**
| Read | Rule | Do / don't |
|---|---|---|
| Skip-policy shadow | pre-registered checkpoints at Bonferroni-split alpha (3 looks), terminal on a decisive look | 15 resolved days is not a terminal look; stays `insufficient_n` unless a checkpoint fires; full alternative-trajectory replay is a separate, labeled analysis |
| Context shadow v2 | frozen eligibility rules | report agreement, discordant outcomes, coverage, interval; equal aggregates ≠ equivalence; no-promote stands |
| Decision-weighted `decision_weighted_lgbm_v0` (tracker #16) | live-forward logging gated on production-pick parity; n≥120 paired `p_57_mdp` protocol | **Inventory finding 9/14:** the logging IS running — `bts-live-forward-capture.timer` every 15 min 05:00–22:00 ET + `bts-live-forward-resolve.timer` 4×/day; 68 of 79 decision days since 5/09 captured (top-10 ranked slates, production + candidate, `bts_candidate_ranked_slate_pair_v2`), the 11 missing are all MDP skip days with no pick file (capture returns `pending_pick`). Remaining inventory: verified pre-outcome rows, prior reads, n vs the 120 threshold → completed / ready-but-unread / insufficient / invalid; interpret `p_57_mdp` against the July evaluator findings; label amendments |
| Tail-policy period (9/03→) | objective/state/provenance audit; stop behavior; delivered/entered/private separation | a few dates validate the mechanism, not the rates or E[best] optimality |
| `mdp_policy_alignment` bin collapse | quantify served-p vs boundary distribution by epoch | a five-bin histogram is not evidence of useful discrimination |

### W1.5 Incident register → ops candidates
7/16 singleton-slate gap · 8/11 MLB auth flap · 8/13 silent pass (Warmup) · 8/30 late pick (Kwan) · 9/03 idle (all-skip table) · any private-mode/tail anomalies through 9/27. Each → mechanism, fix status, **failure-path fixture** that reproduces it, residual gap. Feeds W4 rank 2.

### W1.6 Claims hygiene **[r2: immutable reports]**
Historical reports are **not edited**. A single **corrections index** (`docs/audit/2026-09-corrections-index.md`) lists each superseded claim, its benchmark basis, and the correcting artifact; README's performance table is the one live document that is rewritten (86% = actual-PA walk-forward; 2026 live recommendation record = W1.1 number with denominator; current-era scorecard kept separate from the season-long recommendation mixture).

---

## W2 — Field analysis: three bounded products + one registered execution **[r1: cut list applied]**

**Identifiability first — corrected 9/14 (Eric): the leaderboard itself paginates all the way down.** The endpoint reports `allParticipantsCount ≈ 57k` and honors deep `page/limit` (limit 300 accepted; verified 7/03, `scraper.py`); the 7/04 deep walk of the *active-streak* tab reached rank 22,573 (streak ≥2), while the *all_season* (season-best) tab was only ever captured top-100. So a **full-depth walk of `all_season` on 9/28 (~190 requests at limit 300, jittered, deduped by `user_id`, `allParticipantsCount` recorded; rank ties make it an approximation, not a census)** makes the **whole-field season-best distribution, the counts reaching 20/30/40, and our percentile identifiable** from the ranked board. What stays survivor-selected is the *pick-level* corpus: `user_picks` profiles exist only for users who appeared in a top-100 tab (2,042 users) plus whoever the 300-profile backfill covers. No inverse-probability "survivorship correction" exists for that pick-level sample without inclusion probabilities. Raw 2.81M rows are an archive size (every observation appended); analysis units are unique (user_id, round/date, slot) after revision handling — the existing `latest_per_pick_date` dedup drops DD slots, so build a proper dedup. **Predictive uncertainty is clustered by baseball dates/games, never by user rows.** **[r2]**

| Estimand | Status |
|---|---|
| Final captured ranks, best streaks, trajectories of captured users | **Keep** (capture-as-of; listing ≠ awarded prize) |
| Counts reaching 20/30/40 | **Keep — whole-field** from the full-depth `all_season` walk (tie/rank semantics + completeness recorded); observed-cohort lower bound is the fallback if the walk ends incomplete |
| Entrant-wide season-best distribution, our percentile | **Keep — whole-field** from the same walk (denominator = `allParticipantsCount`; a participant with best 0 may not be listed — record the listing floor). "Our skill rank" on *hit rate* stays **cut**: per-pick records exist only for the survivor-selected profile corpus |
| DD frequency/concentration/pair choices | **Keep descriptive** within a named cohort; missing dates ≠ skips |
| Retrospective consensus vs our locked picks | **Keep as selected-cohort description** |
| Forward performance of an **as-of-defined historical cohort** (users present in a named early snapshot/tab, followed to 7/03 with attrition audited) | **Keep** — the most defensible comparison; it is *not* pre-registered by being chosen now **[r2]** |
| 7/04–9/27 via final backfill | **Keep as a separately labeled survivor case series**; never appended to the cohort curve |
| Causal value of copying/doubling/skipping/stacking | **Cut** |
| MLB `probabilityStarter` forecast quality | **Keep**, gated (below) |

**Products**
1. **Final-leader case series** + observed threshold counts, with tie/rank/tab definitions and completeness checks.
2. **As-of-defined cohort comparison** — exact snapshot date, tab, and cohort manifest; attrition table (which later logs exist for departed users); plus the separately labeled final-survivor extension from the 9/28 grab.
3. **MLB forecast benchmark** (7/04→9/27 static captures), gated: (i) as-of join `(round_id, unit_id, batter_id, game_pk, captured_at, forecast)` using the last capture at or before OUR cutoff, today/tomorrow never conflated; (ii) establish target semantics (conditional on starting? on an AB? unconditional?) — if unresolvable, report association not a certified forecaster; (iii) freshness uncertainty from content-dedup gaps stated (logs if retained); (iv) `numberSelections` is global popularity — never per-round weights; (v) score MLB vs our **archived** served p on identical shared candidates, date-weighted: proper scores, within-slate discrimination, shared-set top-1, disagreement-only outcomes, coverage; (vi) **no blend/gate search on this window** — one combination rule is nominated afterwards as W4 rank 1 and validated on later untouched dates (2027 forward if necessary).
4. **#87 mechanism mining — inventory, then execute under the FULL original protocol** (`docs/sota_audit/2026-05-10-leaderboard-mechanism-mining-prereg.md`; script `scripts/leaderboard_mechanism_mining.py` + tests exist; no result artifact or memo exists → never run). Primary read = consensus top-N coverage; nomination requires the full conjunction: ≥30 disagreement units, ≥5pp, BH≤0.10, **BY sensitivity reported**, **all-tracked-direction check**; n≥15 = testable cell, n≥30 = nomination threshold (distinct); legal DD pair/tie handling specified before comparing two modal slots; untestable sparse cells reported. A passing cell nominates a feature hypothesis (W4 rank 7), never a copying strategy. **[r2]**

---

## W3 — Literature / SOTA refresh

**Method [r1][r2]:** for each of the 17 tracker areas record two separate things: **implementation state** ∈ {unstarted, built, built-and-measured} and **evidence disposition** ∈ {positive, negative, inconclusive, invalid, parked, deployment-cleared, retired}; plus frozen implementation ref · real evaluation basis (actual-PA / estimated-PA / at-cutoff) · prior negative result · whether 2026 outcomes were consumed · specific reopening trigger. A built-but-failed method is "built / negative", never "deployment-cleared" and never forcibly "retired". Do not reopen all 17 or mint a candidate per paper; a paper may earn its place by ruling something out. Primary-source links stay in the final memo.

**Nominations from Codex, primary pages verified to exist and match titles on 2026-09-14 (applicability still to read in W3):**

| Item | Area | Proposed use | Kill / boundary |
|---|---|---|---|
| Gorishniy, Kotelnikov, Babenko — *TabM* (ICLR 2025) | #17 model-class | one frozen small config vs identical LightGBM feature/serving contract | no held-out selected-game score/rank gain at agreed compute; no architecture sweep |
| Holzmüller, Grinsztajn, Steinwart — *Better by default* / RealMLP (NeurIPS 2024) | #17 | alternative fixed-default challenger; comparison design | pick TabM **or** RealMLP, not both; no LightGBM re-tuning |
| Hollmann et al. — *TabPFN v2* (Nature 2025) | #17 / #10 | compact game-level residual/reranking experiment on strictly out-of-time base predictions | small/medium-table method; park unless residual headroom is measured |
| Erickson et al. — *TabArena* (NeurIPS 2025) | #5 / #17 methodology | challenger selection + budget/version tracking | not a substitute for temporal baseball evaluation |
| Angelopoulos, Barber, Bates — *Online conformal prediction with decaying step sizes* (ICML 2024) | #8 | sequential coverage monitor for a PA-count model, if one is justified | coverage ≠ calibrated hit probability; **not applicable** for game-p recalibration |

**Search questions (primary sources + as-of data required; public GitHub/blog/Reddit read for candidate *definitions*, not benchmarks):** public BTS models/write-ups 2026 · MLB `probabilityStarter` methodology · Statcast bat-tracking 2026 metrics (miss distance, swing path, squared-up) and their coverage eras · calibration under regime shift · optimal stopping/MDPs with resets and milestone objectives · forecast combination with a public benchmark · ball-drag literature 2026 · streak-contest analyses (any sport) with a public-consensus feed.

**Outputs:** tracker status section dated 2026-09; `docs/sota_audit/2026-<date>-literature-refresh.md` mapping each item → W4 experiment | not applicable (reason) | closed (reason).

---

## W4 — Conditional 2027 candidates (ranked; **reasoning only**, conditional on W1) **[r1 order][r2 rank-4 gates]**

Prerequisites that are *not* alpha candidates: pin the old recipe; measure `BTS_LGBM_DETERMINISTIC` separately (paired seeds); multi-seed re-validation of shipped changes whose justification depended on seed 42; README hygiene. None may change training while W1 estimates the old system's gap. **Evaluation is per candidate type [r3]:** prediction-changing candidates (ranks 1, 3, 5, 6, 7, and 4a) use the applicable frozen baseline/candidate forecast comparison (bridge surface C vs one frozen modification, "surface E"); policy-only candidates (4b) use frozen forecasts with independent state/eligibility replay; ops candidates (rank 2) use their declared failure-path fixtures and recovery checks — a missing reconstructable forecast surface never blocks an otherwise validated ops repair. Each selected candidate receives its corresponding result and an independent acceptance disposition.

| Rank | Candidate | Pre-registration stub | Kill condition |
|---|---|---|---|
| 1 | **MLB forecast benchmark → at most one blend/gate** | frozen benchmark on shared as-of candidates (W2.3) → if residual information exists, ONE combination rule → validate on later untouched dates | target/availability unresolvable; no residual gain; no valid follow-up sample |
| 2 | **Outcome/eligibility/entry watchdog + restore checks** | fault fixtures from W1.5 incidents; detection/recovery at delivery, entry, restart, singleton-slate, private-vs-contest boundaries | adds state races; cannot distinguish private picks from entries. No P(57) significance required |
| 3 | **PA-opportunity distribution / aggregation challenge** | same PA models + slates; fixed slot-count baseline (2.5 starter PAs + lineup-slot table) vs ONE pregame count-distribution/starter-allocation candidate: `1−E[(1−q)^N]` vs plug-in `1−(1−q)^{E[N]}`; earlier-fit/later-test proper scores primary | bridge finds no residual beyond the June PA-tilt null; candidate needs actual-exposure inputs; hits-per-PA target changed inconsistently |
| 4 | **Calibration map and/or policy repair — two different artifacts, two different gates** | *(a) Calibration map:* identity vs one regularized intercept map (slope challenger only with support); gate = **held-out proper-score improvement**. *(b) Policy-only change* (boundary-only, transition-rate re-estimation, deterministic-training, objective, action-rule — each separate, never combined in one A/B): gate = **forecast parity where applicable + valid state/eligibility replay + the declared downstream-value criterion + regression limits**, reporting **state-weighted decision consequences** (a few high-streak action changes can matter more than global change frequency); comparators = deployed, always-single, legal always-double; each candidate replays its OWN streak/saver/best state through the calendar with its own boundaries | (a) proper scores don't improve. (b) downstream-value criterion not met or regression limits breached. **Never** kill (b) solely on unchanged Brier/log loss or low global action-change frequency; **never** promote a gain that depends on iid rare-event values or fragile calibration |
| 5 | **Miss-distance / contact-suppression feature** | one frozen lagged definition; coverage era, swing-conditioned denominator, as-of lag, missingness/permuted controls fixed before labels | benefit is coverage-driven; no selected-game incremental value; below practical effect bar |
| 6 | **One model-class challenger** (TabM or RealMLP) | frozen input set, budget, temporal comparator, paired seeds | proper-score/ranking evidence fails; improvement needs data unavailable to baseline |
| 7 | **Consensus-derived contexts** (from #87) | only an independently operational, pre-lock-available context; validate outside the selected window | reduces to winner identity, postgame exposure, missing-at-lock info, or an already-rejected feature |

Candidate registrations (only for candidates actually **selected** for a run) must state: data/recipe hashes · exact as-of information set · temporal splits · primary metric · practical effect threshold · guardrails · MDE/power rationale · family-control rule · missingness rule · stopping rule · positive/negative/inconclusive dispositions. Ship thresholds are fixed before results are seen. Small prespecified screens reject; 24 paired seeds are reserved for surviving stochastic-model comparisons and final rebaselines.

**Policy replay rules:** replay each candidate's own state through the original calendar; audit availability on unplayed days and partnerless-double fallback (never +2 for a missing partner); the July replay's known caveats (180-position clock dropping rows; partnerless-double discrepancy) are fixed or disclosed before use. P(57) reported per rule 1.5.

---

## W5 — Sequencing and gates **[r2: dependency graph repaired]**

```
W0-NOW  (today→): W0.1 profile backup + W0.1b pinned diagnostic
                  → W0.2a preliminary checkpoint (incl. data/validation)
                  → W0.3 exposure register + W0.4 protocol inventory
                  → W0.5 owner decisions D3 (before any new read), D5 default recorded, D8 (before 9/19)
W1 first pass:    W1.1 minimum ledger (DD legs + inclusion) → W1.4a DD tripwire under the ORIGINAL snapshot rule
W0-SCHEDULED:     W0.6 grab ~9/28 (reviewed runbook; fixture rehearsal only) → W0.7 final snapshot after declared cutoff (restore-tested)
W1 rest:          W1.1 full ledger + reconciliation table → W1.2 bridge A–D → W1.3 tests → W1.4b remaining reads → W1.5 incidents → W1.6 corrections index
W2 ∥ W3:          three field products + #87 under full protocol ∥ literature refresh
Decision memo:    D1, D2, D4, D6, D7 with frontier tables in natural units (D3/D5/D8 already recorded)
W4:               at most a small declared candidate family → registrations for SELECTED candidates → approved compute → evaluation per candidate type (forecast comparison / frozen-forecast state replay / failure-path fixtures) → independent (Codex) acceptance memo per candidate
Close:            completion matrix + 2027 season-start checklist delivered (activation itself is a later authorized event)
```
A complete result may be **"retain baseline"**: corrected benchmark, retired invalid claims, completed old reads, better observability, no alpha. Kill criteria are not lowered to make W4 ship something. The one authorized capture may end incomplete (throttle, unavailable profile, truncated history); the plan accepts that outcome and does not quietly authorize another attempt.

---

## Owner decisions — plain language **[r2 phrasing]**

Recorded in the exposure register when made. D3, D5, D8 are needed early (W0.5); the rest belong to the decision memo.

| # | Question | Options | Cost if wrong |
|---|---|---|---|
| D1 | **What should the system optimize?** | Prioritize the best chance of 57, using season best only to break ties; **or** accept a specified reduction in projected jackpot chance for a better longest streak. Both outcomes shown in a table before choosing; no unexplained mixture weight | trading away projected jackpot wins; projections themselves are uncertain |
| D2 | **When is evidence enough to change picks?** | Keep the current system pending fresh-season confirmation; **or** permit a *named* change supported by held-out forecast/milestone evidence and explicit stress tests, knowing its jackpot effect cannot be verified from this sample | a change that lowers the true chance of 57 while looking better on proxies |
| D3 | **How should remaining unseen 2026 data be used?** (early) — **Eric 9/14: "not sure" → adopted recommendation: RESERVE** (the unread late-season outcomes stay with the existing frozen tests; every new idea from this wrap validates prospectively in 2027; recorded as a methodology default, reversible by Eric) | Reserve eligible unseen outcomes for existing frozen tests; **or** spend them on exploration and validate new ideas prospectively in 2027. Existing protocol history cannot be waived | exploration spends the data's independent-test value; with ~14 dates left the exploration value is small and the cost is permanent |
| D4 | **What research is funded?** | No new paid runs; **or** one named candidate cycle covering the whole W4 program, with a total spend cap and stop conditions; another cycle needs a recorded scope decision | money and attention on candidates with no path to acceptance |
| D5 | **Which profiles use the existing 300-profile cap?** (early) — **Eric 9/14: "not sure" → adopted recommendation: SPLIT 150/150.** The whole-field distribution now comes from the board walk, not profiles, so profiles exist to answer pick-level questions: 150 = the top of the final `all_season` board (describes what the winners actually picked), 150 = users from the **5/01 first snapshot** (the earliest as-of cohort; their 7/04–9/27 logs give a follow-up window chosen *before* their outcomes were known). Recorded as a methodology default, reversible by Eric before 9/28 | Keep the authorized final-leader allocation; **or** reserve an exact number for the early cohort | all-final-leaders = pure description, no unselected follow-up; all-early = weaker picture of the winners |
| D6 | **How should 2027 operate?** | Keep private research mode; **or** activate recommendations after the checklist, naming who checks contest entry and official state | unwanted messages, or picks left unentered |
| D7 | **Approve concrete production changes** | Engineers measure and recommend (incl. the deterministic-training transition); Eric approves each actual prediction/policy change after seeing its paired effects. Repeatability alone does not establish better picks | shipping on a green harness rather than evidence |
| D8 | **Collect forecasts after policy play stops on 9/19?** (before 9/19) | Accept the shorter research record; **or** continue a separately labeled forecast stream through 9/27 while preserving official policy/state (W0.8). Buys diagnostic observations, not valid extra skip-policy trials | a small code change on the box during the last two weeks vs nine fewer diagnostic days |

---

## Deliverables, completion matrix, definition of done **[r2]**

**Path rule:** every memo is `docs/audit/YYYY-MM-DD-<slug>.md` or `docs/sota_audit/YYYY-MM-DD-<slug>.md` dated on the day it is finalized; one index `docs/audit/2026-season-wrap-index.md` links the exact final artifacts.

| Deliverable | Path | Completion evidence |
|---|---|---|
| Profile backup + pinned diagnostic | manifest in `data/hetzner_results/season_2026_snapshot/`, runbook `docs/audit/<date>-profile-restore.md` | restic snapshot ID; empty-dir restore hash match; diagnostic output hash |
| Preliminary + final snapshots | same tree; restic tag `season2026` | snapshot IDs; restore test; unresolved-items list |
| Exposure register + protocol inventory | `docs/audit/2026-09-15-exposure-register.md` | every read cites an entry |
| Season ledger + builder + tests + reconciliation table | `data/validation/season_2026_ledger.parquet`, `scripts/audit/build_season_ledger.py`, `tests/scripts/test_build_season_ledger.py`, memo | tests green; table explains 191/157 vs 141/82 |
| Due-read memos (DD tripwire; skip shadow; context shadow; #16; tail; bin collapse) | one memo each | original protocol quoted; disposition ∈ {completed, deviation-recorded, ready-but-unread, insufficient, invalid} |
| Bridge memo A–D | memo | missing-fraction reported |
| Grab runbook + status file + cohort manifest | `docs/audit/<date>-final-grab-runbook.md`, `data/leaderboard/final_grab_<date>/` | every planned request accounted for |
| Field memos (products 1–3) + #87 result | `docs/sota_audit/<date>-field-*.md`, `<date>-mechanism-mining-result.md` | protocol conjunction reported incl. untestable cells |
| Literature refresh + tracker section | `docs/sota_audit/<date>-literature-refresh.md`, tracker | each item → experiment / n.a. / closed |
| Corrections index + README | `docs/audit/2026-09-corrections-index.md`, `README.md` | |
| Decision memo | `docs/audit/<date>-2027-decisions.md` | structure below |
| W4 registrations / results / dispositions — **selected candidates only** | `docs/sota_audit/<date>-prereg-<slug>.md`, `<date>-result-<slug>.md`, Codex acceptance memo | or an explicit "deferred: <reason>" line in the completion matrix |
| 2027 season-start checklist | `docs/ops/2027-season-start.md` | executor named; acceptance evidence per item |
| Completion matrix | in the index | every row ∈ {completed + evidence, unavailable + reason, deferred + reason} |

**Decision memo structure:** reconciled baseline and denominators · tested changes with uncertainty · observed results vs projected P(57) · failed/inconclusive candidates · Eric's decisions D1–D8 · approved/deferred changes · remaining gates.

**2027 season-start checklist must cover:** artifact restore verified · recipe/policy pairing (base sha ↔ tail sha) · current contest rules/calendar/state · authentication + entry checks (cookie re-capture procedure) · DM/cron configuration (restore `pick_delivery`, `cron-setup-hetzner.sh install`) · rehearsal on a fixture day · rollback · **explicit activation approval (D6)**. Delivering the checklist closes the wrap; activation is a later authorized event.

**Done** = the completion matrix has no empty rows; every outcome-bearing read cites its exposure-register entry and original protocol; the profiles and snapshots are verified restorable; README is corrected; the decision memo has been put to Eric with review status stated.

## Open verification items (not established by anyone yet)
- Whether `decision_weighted_lgbm_v0` live-forward logging ever ran in 2026 (W1.4b inventory).
- `probabilityStarter` target semantics and historical availability at our cutoff (W2.3 gate).
- Static-capture fetch log retention (freshness uncertainty for the MLB benchmark).
- How far back the profile backfill actually paginates for the cohort (W0.6 records it).
- Exact composition of the 191/157 vs 141/82 difference (W1.1 reconciliation table).
- Whether the original DD-tripwire n≥80 snapshot is recoverable (W1.4a).
