# C1 rank 2 registration: outcome / entry / restart watchdog and restore checks

**Status:** design rev 1, 2026-10-04, for Codex design review (at most 2 rounds, then freeze). Production code follows the deploy rules: reviewed until SIGN, and shipped only with Eric's D7 approval.
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`).
**Plan row (rank 2):**
- Fault fixtures come from the W1.5 incidents; detection and recovery cover the delivery, entry, restart, singleton-slate and private-vs-contest boundaries.
- **Kill if** the watchdog adds state races or cannot tell private picks from entries.
- No P(57) significance is required.

**Exposure:** none needed. This is an ops candidate with no outcome analysis; its gate is fault fixtures, not a 2026 read.

## In plain words
- **What it is.** A separate, read-only checker that runs on a timer, independent of the scheduler. It looks at what the system did each day and alerts Eric when something that should have happened didn't, or something that shouldn't have did:
  - no pick delivered by the cutoff;
  - a pick on a postponed game;
  - an entry it can't confirm;
  - a grade that disagrees with the contest;
  - a streak file that disagrees with the graded history;
  - a delivery setting that contradicts whether Eric is actually entering;
  - a backup that can't be restored.
- **What it never does.** It never fixes anything itself: it writes only its own files, so it cannot race the scheduler.
- **The known defects.** Those it would catch (for example grading a Pass as a miss) need their own fixes, approved separately.

## 1. Rulings on scope (each with its cost if wrong)
- **R1, detect and alert only.**
  - The watchdog never writes pick, decision, streak, scheduler-state or health-state files. It never re-delivers or re-grades.
  - *Cost if wrong:* slower recovery, because a human acts on each alert. An auto-repair would race the in-loop fallback, the 01:00 grader and reconcile, which is the plan's kill condition.
- **R2, the unfixed defects are out of rank 2.** That is I-201, I-202, I-203, I-204, I-077, I-205, I-206's no-games early return, and I-209.
  - Each is a real defect in deployed behaviour, and its repair is a separate D7 change. This design proposes they be listed as checklist item **B4** for Eric.
  - The watchdog detects their failure modes in the meantime.
  - *Cost if wrong:* if Eric plays in 2027 with them unfixed, the watchdog turns silent damage into a same-day alert, but the damage still happens.
- **R3, an independent process.** A `bts watchdog` CLI run by cron. It is not inside the scheduler (the I-206 gap: end-of-day health runs only in the scheduler and is skipped on no-games days). It runs every day, games or not.
- **R4, declared entry intent: the kill condition's answer.**
  - A new required orchestrator key, `entry_intent = "enter" | "research"`.
  - Delivery mode (public, dm or private) says how a pick reaches Eric; intent says whether he enters it.
  - The watchdog refuses to run, and alerts, if the key is missing.
  - Under `research` it never alerts about entries, and alerts on any public or DM delivery: a private-vs-contest breach (the C2 and I-086 class).
  - Under `enter` it alerts on undelivered, unentered or unverifiable picks.
- **R5, writer rules.**
  - It writes only under `data/watchdog/`: its state, a per-date observation log and its alert dedup file, all under its own `flock`.
  - It reads production files read-only.
  - Alerts use the existing DM transport with a `[watchdog]` prefix and their **own** dedup file. They never touch `health_state/health_dm_delivery_status.json`, the scheduler's unlocked read-modify-write file.
- **R6, no new authenticated traffic.** Entry and contest checks read artifacts the existing jobs already produce: `contest_streak.json`, the `check-pick-entered` marker and contest-ledger fetches. The watchdog never calls the contest with Eric's cookies. The only network call it adds is the public, unauthenticated MLB schedule and game-status API.

## 2. Boundaries and the alert each one raises

| Id | Boundary | Fault it must detect | Incident class |
|---|---|---|---|
| W-deliver | delivery | under `enter`, no delivered pick by first pitch − cutoff on a day with games (including a dead or hung scheduler, and singleton or moved-up games) | I-206, I-077, I-084 |
| W-postponed | delivery | a delivered pick whose game is postponed or suspended before first pitch | I-203, I-043 |
| W-entry | entry | under `enter`, entry not verified by the cutoff, or the entry checker unable to read the account through the window (no marker, auth failure). It never says "not entered" unless absence is positively verified | I-061, I-072, I-110; I-071 is the false-alarm guard |
| W-decision | restart / records | a committed pick (pick file shows commit evidence) with no `decision.json` | I-209 |
| W-grade | grading | after 08:00 ET the next day, a local graded result that disagrees with the contest's grade for that slot | I-201, I-089, I-207 |
| W-state | restart | `streak.json` (streak, saver) that disagrees with a replay of the graded pick files | I-202, I-204 |
| W-mode | private vs contest | intent, delivery mode and cron disagree. For example: intent `enter` with delivery `private`; intent `research` with DM/public delivery or the entry cron active; a legacy `shadow_mode`-only config | I-086, checklist C2 |
| W-restore | restore | weekly: restore models and policies from R2 into a scratch directory; any hash differs from live, or the tail/base pairing fails validation | I-113, checklist A1/B2 |
| W-self | liveness | the watchdog's own dead-man ping (a separate Healthchecks check, created by Eric) | — |

**Out of scope:** auto-repair; model quality (`slate_auc` exists); leaderboard scraping; anything needing the contest cookies.

## 3. Schedule (cron, ET)
- Every 15 minutes 10:00–23:45: W-deliver, W-postponed, W-entry and W-mode.
- 01:30, after the grader: W-decision and W-state.
- 08:15, after the 07:40 C-03 reconcile line (checklist B1/A5): W-grade.
- Sunday 04:00: W-restore.
- A `flock -n` singleton per job, like the grading cron.

## 4. Gate: what "done" means
1. **Fault fixtures, red then green.**
   - For each boundary W-deliver … W-restore, a test plants the fault and asserts a **positive alert record**: the alert is written to `data/watchdog/alerts.jsonl` and a transport call is made, with the transport patched.
   - Each test is red before the check exists and green after.
   - Each also has a matching **no-fault control** that must raise no alert.
   - Fixed clocks only: no real elapsed time.
   - Harnesses reused from the W1.5 inventory: `_drive_run_day`, `test_incident_2026_08_30.py`, `test_e2_delivery_idempotency.py`, `tests/health/test_{result_resolution,pick_entry_source,late_delivery,postponed_pick}.py`, `test_scoring_lock.py`, `test_check_results_wait.py`, and the I-077/L03/L04 characterization harnesses.
2. **No-state-race proof (kill condition 1).**
   - The test suite hashes every production file before and after each check runs, including concurrently with a simulated scheduler write loop. Any change made by the watchdog fails the test.
   - A test asserts the watchdog takes only its own lock.
3. **Private vs entry (kill condition 2).**
   - Fixtures run the same day under `enter` and `research` and assert the opposite alert sets.
   - A missing `entry_intent` must refuse to run.
4. **Prerequisite:** checklist C1 (a private-mode test that patches `bts.posting.post_to_bluesky`) lands **before** any watchdog delivery-mode test.
5. **Review:** deploy-gating Codex code review until SIGN, then Eric's D7 approval, then deploy.
   - Activating it on the box needs `cron-setup-hetzner.sh install`, so it rides the same install as checklist A5.
   - If Eric plays, it ships before activation (A8).

**Re-certification:** the watchdog changes no certified defence path (it is read-only), so the W1.5 certificates (tooling frozen at `f453283`) need no rerun. Checklist C3/C4 stay separate.

## 5. Compute and cost
- Unit tests run on the Mac and are not counted toward the cap.
- On the box: the cron jobs (seconds per run) and the weekly restore (an R2 download of about the size of the model and policy artifacts).
- No C1 compute budget is needed beyond a one-off restore rehearsal: declared 0.1 CPU-hours through the launcher.

## 6. Limits
- **Detection, not prevention:** a known defect still does its damage; the watchdog only makes it loud the same day.
- **Grade checks can lag:** W-grade depends on the contest artifacts that existing jobs fetch. If those fail, W-grade reports "unverifiable", never "agrees".
- **Dead-man ping:** W-self needs Eric to create a Healthchecks check (his login). Until he does, watchdog death is silent.
- **Configuration change:** the declared `entry_intent` key is a production configuration change (D7) and must be set in `~/.bts-orchestrator.toml` before the watchdog runs.
