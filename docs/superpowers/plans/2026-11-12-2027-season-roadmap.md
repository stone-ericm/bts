# 2027 Season Roadmap

> **For agentic workers:** this is a roadmap across independent threads, not one build plan. Each thread marked **[own plan]** gets its own spec and plan (superpowers:brainstorming, then superpowers:writing-plans) when it starts. Threads marked **[inline]** are small enough to do directly. Steps use checkbox syntax for tracking.

**Goal:** have BTS entering picks on the first day of the 2027 contest, more reliable than 2026, with the data captures running that cannot be repaired afterwards.

**Trigger:** Eric, 2026-11-12: "We're gonna play in 2027."

**Sources:** `docs/ops/2027-season-start.md` (checklist, FINAL), `docs/audit/2026-10-04-2027-decisions.md` (FROZEN, rulings at the end), `docs/sota_audit/2026-10-04-c1-cycle-index.md` (research cycle C1), `docs/ops/2026-10-10-bts-lead3-handoff-c2-framing-2026.md`, memory `project_bts_hetzner_offseason_2026_10_21.md` (box restore recipe).

---

## 1. The situation in plain words

**The calendar (checked 2026-11-12).**
- Opening Night is Wednesday 2027-03-24, one game. Traditional Opening Day is Thursday 03-25 with 14 games. The first 30-team slate is Saturday 03-27. The regular season ends Sunday 09-26.
- The labor contract expires 2026-12-01 at 23:59 ET. A lockout is widely expected, the two sides are far apart on a salary cap, and talks are not expected to resume before mid to late January. In 2022 the lockout pushed the opener to April 7.
- So the opener is 19 weeks away at the earliest and may slip. Everything that costs money or cannot be undone waits for a confirmed start; everything else can be built now.

**What exists today.**
- The box is deleted. A snapshot, the kept IPv4, both backup sets and the config bundle on this Mac are enough to bring it back in under an hour. It costs about €70 a month while it exists.
- Production code as deployed on 10/21 is complete for 2026 and has two known 2027 blockers (section 3, thread B).
- The October research cycle approved five items. Four were deferred after failing their review rounds (watchdog, calibration map, plate-appearance count model, longest-streak policy). One has a frozen design and no build (the MLB-forecast capture prerequisites). No research result is approved for production.

**The honest read of October (my interpretation, not a measurement).** The cycle spent most of its effort on review rounds. Roughly twenty review rounds returned BLOCK across five items, and most findings were about evidence controls and launch accounting, not about the maths or the deployed behaviour. If 2027 preparation runs the same way, the watchdog and the day-one captures will again not ship. Section 2, decision 2 proposes a fix.

---

## 2. Decisions needed from Eric

**Answered 2026-11-12** (register §C rows D6-2027-play, WP-2026-11-12, C1-4b-reopen, C1-r2-watchdog-reopen, C2-framing-2026-run):
- **1: "Activation approved".** A8 is given. The rehearsal and rollback items still run before picks are delivered.
- **2: "Update all our work policies to match those developed in the eor project".** This replaces the three-tier recommendation below. See `docs/superpowers/notes/work-policies-2026-11-12.md`; threads marked "production tier" are tier A there, "capture tier" is tier B, "research tier" is tier C or D.
- **3: "Reopen it".** Thread E is unblocked.
- **4: "Sure".** Thread C is unblocked.
- **5: "Let's run it".** Read as: run the 2026 framing test. The reading, and where it runs now that the box is deleted, are back with him. The pregame catcher capture in thread D stays either way.
- **9: "Push".** Done: `origin/main` = `6fcfc93`.
- **6, 7, 8: not answered.** Under the adopted design authority the recommendations stand as provisional rulings: day-one captures for all three studies and the GitHub sweep proceed; field capture is decided in February; C3, C4, Phase 2 and the determinism flag wait until after the opener. He can overrule any of them.

The table below is the question as it was put.

| # | Decision | Recommendation | Cost if wrong |
|---|---|---|---|
| 1 | **Is "we're gonna play" the activation approval (checklist A8), or the intent to play with A8 given after the rehearsal?** | Intent now; A8 after the fixture-day rehearsal passes. This is what the checklist describes. | None either way; it only decides whether there is one more yes/no in March. |
| 2 | **Review weight for 2027 preparation.** October used certification-grade review on everything. | Three tiers. Production code that touches picks, delivery or entry: review until SIGN, as now. Capture and instrumentation code: tests, one Codex round, fix, ship, with limits written down. Research registrations: two rounds maximum (the existing pace rule). | Too light: a capture bug loses 2027 data that cannot be re-collected. Too heavy: nothing ships before the opener, as in October. |
| 3 | **Which objective does 2027 play?** D1 chose "better expected longest streak at a stated cost in the chance of 57". The policy for that (4b) was deferred with no trade table. | Reopen 4b now, narrowed to producing the trade table from existing profiles. If no approved table exists by March 1, 2027 opens on the 2026 policy (reach-57 plus the tail policy) and 4b can switch in mid-season with his approval. | Reopening costs review effort. Not reopening means D1 has no effect on 2027 play. |
| 4 | **The watchdog.** An independent check that each day ended in a known state (pick delivered and entered, or a deliberate skip). Deferred after three BLOCK rounds. | Rebuild it narrower: read-only, one daily job, DM on an unknown end state. 2026's three worst days (7/16 undelivered, 8/13 silent pass, 8/30 late pick) were all days nobody was told. | It is production code, so it needs review until SIGN; if that fails again it does not ship and 2027 relies on `check-pick-entered` alone, as in 2026. |
| 5 | **The framing follow-up.** The 2026 test stopped after five BLOCK rounds at a design revision. 2026 data is "reserved, not validation" by his own D2 ruling, so even a pass would not approve a change. | Close the 2026 test unrun. Instead record the pregame starting catcher from the first 2027 game and pre-register a 2027 shadow comparison of variant A. | Closing it gives up one more look at 2026 (about 100 CPU-hours of allowance unspent). Not capturing the catcher means 2027 cannot test framing at all. |
| 6 | **Which deferred research items come back?** Calibration map (4a), count model (3), MLB-forecast capture (rank 1 prerequisites). | Day-one captures for all three, because only the captures are time-critical. Studies themselves: 4a first (the live gap was stated 0.765 against realized 0.68 to 0.73), rank 3 only if time remains. Registrations published before the first 2027 outcome. | Captures are cheap. Studies cost review rounds and can be dropped without losing anything as long as the captures ran. |
| 7 | **Field capture in 2027.** The round-board and full-field profile campaigns were designed and never built. They run on his real account. | Decide in February after the 2027 rules are published. Default is the 2026 footprint unchanged. | A ban costs the streak. Deciding late costs nothing. |
| 8 | **Carried items he added on 10/03:** two unrun incident certificates, one recertification (C3, C4), incident register Phase 2, the determinism flag (B3). | Do the GitHub issues sweep (C5) now. Defer C3, C4 and Phase 2 to after the opener: they certify past incidents and do not change 2027 behaviour. Leave the determinism flag as it is; no cross-provider pooling is planned. | Deferring leaves three past incidents without a current-defence certificate. |
| 9 | **Push local `main`** (6 commits, 2 exist only on this Mac). | Yes, today. It deploys nothing. | A disk loss before pushing loses the offseason banner and the archived teardown review. |

---

## 3. Threads

Order within each phase is priority order. "Blocked by" refers to section 2.

### Phase 0: this week (no box, no decisions needed except 9)

- [ ] **0.1 Push `main` to origin.** [inline] Blocked by decision 9. `git push origin main` only; never `main:deploy` while the box is gone (the workflow fails at SSH).
- [ ] **0.2 Record the decision to play.** [inline] One row in `docs/audit/2026-09-22-exposure-register.md` §C with his words verbatim and the date; update the D6 line in `docs/ops/2027-season-start.md` §D. Register rows are read by tests: run the fast suite before pushing.
- [ ] **0.3 Remove finished worktrees.** [inline] `agent-ad620697…`, `agent-afd1ebb2…` and `season-ledger-phase1` under `.claude/worktrees/`. Check each for uncommitted work first (`git -C <path> status --short`); the season-ledger build directory is in no backup set, so confirm `ACCEPTED.json` is not inside that worktree before removing it. Keep `bts-c2f26-code`, `bts-c2f26-code-review`, `bts-c2-framing-results` and `bts-w15-evidence` until decisions 5 and 8 are made.
- [ ] **0.4 GitHub issues and PRs sweep (checklist C5).** [inline] All states, paginated; a sweep table under `docs/audit/`; new incident candidates listed, not certified.
- [ ] **0.5 E107.** Ask Eric for the outcome of his Healthchecks step and record it in the checklist §D.
- [ ] **0.6 Back up what is only on this Mac.** [inline] `~/.local/share/bts-offseason-2026/` (config bundle with cookies and `.env`) and `~/projects/bts-c2-framing-results/` are not in git. Confirm both are inside the Mac backup set (`claude-shared/backup/mac-backup.exclude`), and say so in the offseason memory file.

### Phase 1: November to January (Mac only, box stays deleted)

**A. 2027 readiness of production code** [own plan, production tier]
- [ ] `src/bts/strategy.py:42` has `SEASON_END_DATE = "2026-09-28"`. The tail policy and the phase-aware bins depend on days left. Set the 2027 date (09-26) from one place and add a health check that fails closed when today is past the configured season end during a scheduled slate.
- [ ] `scripts/cron-setup-hetzner.sh:91` builds `--seasons 2026`. Derive the season from the date.
- [ ] Audit the 30 `src/bts` files that mention 2026 for season-bound logic (not comments). Deliverable: a table of file, line, verdict.
- [ ] Rebuild the tail policy if the season length changes its inputs (`scripts/rebuild_tail_policy.py`; the tail is sha-bound to the base policy, checklist A2).
- [ ] Test first, with `today=` pinned (never `date.today()`).

**B. Singleton and moved-game slates** [own plan, production tier]
- [ ] Opening Night 03-24 is a one-game slate, the exact case in `docs/optimization-ideas.md` "Singleton slates get zero pre-start lineup checks" (7/16: one check, game moved up an hour, day locked undelivered). Fix before the opener: an intermediate re-fetch of the schedule, and at least one check that does not depend on the original start time.
- [ ] Regression test replays the 7/16 schedule (19:10 moved to 18:10 after the 10:00 fetch) and asserts a delivered pick.

**C. Watchdog, narrowed** [own plan, production tier] Blocked by decision 4.
- [ ] Start from the W1 evidence map and the r3 findings (`docs/audit/2026-10-06-c1-r2-watchdog-w0-codex-r3.md`). Scope: read `decision.json`, the entry receipt and the reconcile receipt for the day; DM when the day's end state is unknown or contradicts itself. No writes to pick state.
- [ ] Verify in the failure direction on the restored box: withhold a decision file, see the DM, restore.

**D. Day-one captures** [own plan, capture tier] Blocked by decisions 2 and 6. These cannot be repaired after the first 2027 game.
- [ ] Per-fetch receipt log for static captures (rank 1 prerequisites, design frozen in `docs/sota_audit/2026-10-04-prereg-c1-mlb-capture.md`): fetch time, status, hash bound to the response bytes, including unchanged fetches. Reader must list both `.json` and `.json.gz`.
- [ ] Pre-lock plate-appearance count forecast archive (rank 3's capture half only).
- [ ] Pregame starting catcher per game (decision 5). I am assuming the pregame lineup feed carries fielding positions; verify against a stored 2026 feed before designing.
- [ ] Stable user id with each leaderboard batch and an external timestamped hash of each served slate (checklist §D). Both are small and both were named as unrepairable gaps; include them unless decision 7 shrinks field capture.
- [ ] Slate v2 is already deployed (10/06); confirm on the restored box that `game_time` and `status` are persisted.

**E. Longest-streak trade table (4b)** [own plan, research tier] Blocked by decision 3.
- [ ] Resume from the r3 findings B1 to B7 (`docs/audit/2026-10-05-c1-r4b-code-codex-r3.md`); the maths was unchallenged. Narrow the deliverable to the table Eric must approve: chance of 57 given up against expected longest streak gained, by realized-sequence replay on the `estimated_pa` profiles.
- [ ] Inputs: `mdp_estpa_run` profiles are in the restic archive set (R2 keys are in the Mac Keychain). Where it runs is open: the C1 rule was no multi-hour run on the Mac, and the box is gone. Measure the replay cost first on one seed; if it is minutes, run it on the Mac and record that as a ruling.
- [ ] Gate: Eric's approval of the table under D7, and the tail/base pairing check (A2), before any 4b artifact affects picks.

**F. Registrations** [inline per item, research tier] Blocked by decision 6.
- [ ] Publish the exposure rows and the 2027 fit and test windows for each selected study before the first 2027 outcome. The windows need the real calendar, so draft now and freeze once the season start is confirmed.

### Phase 2: January to February (the lockout gate)

- [ ] **2.1 Watch the labor talks.** Check weekly from mid-January. The restore date is the confirmed opener minus 21 days.
- [ ] **2.2 2027 contest rules (checklist A3).** Fetch when MLB publishes them; diff §6 (hit and void clauses, cutoff, double-down, saver) against `docs/audit/2026-09-29-incident-register-evidence/rules/section6.txt`; record season dates and the contest end. Confirm the contest exists in 2027 at all before anything is restored.
- [ ] **2.3 Box choice.** Restore the CPX42 from snapshot `446156433` (known recipe, about an hour) or rebuild on a smaller, cheaper box from the install scripts (a small project with new failure modes). Recommendation: restore the snapshot for 2027; revisit a smaller box next offseason.
- [ ] **2.4 Decision 7** (field capture footprint), once the rules are read.

### Phase 3: opener minus 21 days to the opener (box restored)

Follows `docs/ops/2027-season-start.md`; items in order.
- [ ] Restore the box per the memory recipe (`--start-after-create=false`, attach the firewall, expect regenerated host keys, the resolve timer catches up on first boot). Un-pause both Healthchecks by hand. Remove the offseason banner from `CLAUDE.md`.
- [ ] A1 artifact restore verified; A2 tail and base policy pairing.
- [ ] Deploy Phase 1's production code through a deploy-gating review, inside a scheduler sleep window. The test gate ran 43 minutes on 10/21.
- [ ] A4: Eric re-captures cookies on the Mac; then verify `fetch-contest-streak` and `check-pick-entered` against the live account.
- [ ] A5: `pick_delivery = "dm"`, `entry_intent = "enter"`, `cron-setup-hetzner.sh install`. Verify the crontab before and after.
- [ ] A6: fixture-day rehearsal, no DM, no contest call. A7: written rollback commands.
- [ ] Day-one captures observed running against preseason or first-day data, each with one failure-direction check.
- [ ] **A8: Eric's activation approval.**

### Phase 4: in season

- [ ] First week: daily read of the watchdog, entry receipts and capture receipts. Opening Night is the singleton-slate test.
- [ ] Studies fit on their registered early window and score on later dates. Nothing changes picks without an independent acceptance and Eric's D7 approval.
- [ ] Deferred until after the opener unless he says otherwise: C3, C4, incident register Phase 2, the unverified README benchmark, the swing miss-distance feature idea.

---

## Progress and rulings ledger

**Done (2026-11-12 and 11-13), all on `main`:**
- **Thread A.** Season end follows the year (`29e10bf`); the nightly cron builds the current season (`1137cbe`). Audit of every 2025–2027 literal in `src/bts` code (16 lines): the only season-bound logic was `strategy.SEASON_END_DATE` and the cron line. `leaderboard.scraper.scrape_leaderboard` defaults `season=2026`, but both production callers pass `today.year`. The tail policy needs no rebuild: 2027 has 186 days opener to end, clamped to the same 180-day table as 2026. Still open in A: the fixture-day dry run (checklist A6).
- **Thread B.** Thin slates (one or two games) get checks at start−180 and start−120 as well (`plan_run_times`). Limit: a start moved up by more than about two hours after the morning fetch can still outrun every check.
- **Thread C.** `bts day-outcome` (`src/bts/day_outcome.py`), installed by cron at 23:50 and 08:30 under either intent.
- **Decision 5.** The 2026 framing test ran and came back NEGATIVE (`docs/sota_audit/2026-11-13-result-c2-framing-2026-test.md`). The framing thread is closed; the pregame-catcher capture in thread D is dropped with it.

- **Incident repairs (Eric 2026-11-13: "Repair them").** All five open incidents in `tests/test_incident_register_2026.py` are repaired and the file has no expected failures left: E77 (singleton slate), L01 (a Pass graded as a miss; Official Rules 6A–6C), L02 (same-day reconcile rollback), L03 (postponed cached pick delivered), L04 (a result applied twice across a crash). The frozen 2026 register JSON is left as the record of the season; the tests carry the repaired state.

**Design rulings made under the standing design authority (Eric can overrule any of them):**

| # | Ruling | Why | Cost if wrong |
|---|---|---|---|
| R1 | An unconfigured season year falls back to September 28 and WARNs, instead of raising | A raise in the pick path would idle or crash the day; a wrong end date by a few days is inside the policy's stated tolerance | Days-left is off by up to a few days in a year nobody configured, until the WARN is read |
| R2 | Thin-slate checks are added only for slates of one or two games, at fixed offsets, with no mid-day re-planning of the schedule | The smallest change that covers 7/16 and Opening Night; re-planning the run list mid-loop touches the fallback and coalescing logic | A start moved up by more than about two hours on a thin slate, or any large move on a normal slate whose other checks are all later, is still missed; the day-outcome check then reports it the same night |
| R3 | The watchdog is one stateless read-only command, not the frozen W0 design | W0 failed three review rounds on its own notification state; cron cadence is the dedup | A problem day produces two DMs (23:50 and 08:30), not one; no escalation ladder |
| R4 | "Entered" needs a sealed `observed` receipt with `qualification.all_confirmed`, bound to the decision bytes on disk, the expected account and the date, before the cutoff | The receipt's own consumer rules (`docs/ops/pick-entry-receipt-v1.md`) | A real entry that no check observed in time reads `entry_unverifiable` and DMs: a false alarm, not a miss |
| R5 | The framing test ran on the Mac Studio without the C1 launcher, on a rebuilt `pa_2026` | The box is deleted; X-37 declares each deviation | The result is not byte-comparable with October's box runs; it is a self-contained test |
| R7 | After the checks end with a selected pick undelivered, the loop sleeps to that pick's fallback deadline (not on a standing skip) | The thin-slate checks exposed a day that ended undelivered when later checks produced nothing; the same gap existed on multi-game days | One extra fallback cascade on such days; a skip day with a stale preview is exempt |
| R8 | Grading follows Official Rules 6A–6C: no official at-bat and no sacrifice fly, did not play, or suspended without a hit is a Pass; a suspended game never grades a miss | The pinned rules text; the contest holds the streak in these cases | If MLB's feed shows an empty batting line for a player who did bat, a miss reads as a Pass until the 02:00 or 07:40 reconcile, and the contest-anchored streak overrides the local one either way |
| R9 | A result is applied to the streak at most once per date, by a marker in `streak.json` | The smallest change that makes both scorers idempotent across a crash | Clearing a pick's result by hand to re-score it no longer re-applies; reconcile's replay recomputes from the pick files |
| R10 | The delivery guard delivers when the schedule cannot be read | The read fails in the same outages that make a refresh fail; a postponed pick is a Pass, not a miss | A postponed game's pick can still go out during an MLB API outage |
| R11 | A double with exactly one leg on a postponed or cancelled game is delivered as a single of the other leg (Codex pp1 B2) | Refusing the whole double lost the day at the final fallback, where no later cycle re-picks | A single goes out that the policy chose only as half of a double; both legs had cleared the policy's bar |
| R12 | Reconcile's replay refuses to save while today's pick is unresolved but marked applied (Codex pp1 B1) | Otherwise the replay withdraws a result whose marker stops the scorer's retry from restoring it | The streak is not recomputed by that reconcile run; the next one after the scorer finishes does it |
| R6 | Decision 6–8 defaults stand until Eric says otherwise: day-one captures proceed; field-capture footprint decided in February; C3, C4, register Phase 2 and the determinism flag wait until after the opener | Unanswered on 11/12; all reversible | Captures built that he did not want cost only build time |

**Tier A changes awaiting the one whole-range review before the pre-season deploy:** the season-end lookup, the thin-slate planner, the day-outcome check and its cron lines, the cron season. Added 11/13: the post-loop deadline wait, the Pass grading, the same-day replay, the at-most-once streak marker, the postponed-game delivery guard.

## 4. Critical path

1. Threads A and B are required to play at all on 03-24 and depend on no decision. Start here.
2. Thread D is required to learn anything from 2027 and depends on decisions 2 and 6.
3. Thread C is the largest reliability gain and depends on decision 4.
4. Thread E is the only thread that changes how 2027 is played and depends on decision 3.
5. Everything in Phase 3 depends on the labor outcome.

## 5. Risks

- **Lockout.** The opener may move or the season may shorten. A shorter season changes days-left for the tail policy and every registered fit and test window. Thread A must make the season end a single configured value for this reason.
- **Review rounds consume the offseason again.** See decision 2.
- **The contest changes or is not offered.** Checked in 2.2 before any spend.
- **Restore surprises.** Tailscale re-enrolment, host keys, the Persistent resolve timer and the hardcoded cron season are known; 21 days of margin is for the unknown ones.
- **Live calibration.** The 2026 model's top pick hit 68 to 73 percent against a stated 76.5. Nothing in this roadmap fixes that before the opener; 4a measures it on 2027 data.
