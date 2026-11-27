# Focused read of K1 and K2 at `240324ab`

## Verdict

BLOCKING FINDINGS

- **BLOCK — K1 suppresses distinct, previously delivered alerts after midnight.** On the synthetic June 14 slate, the real `run_day` lineup delivery guard refused game `70000` as postponed at 23:58 ET. It then refused game `70001` as cancelled at 00:24 ET on June 15. Both games reached delivery as scheduled candidates before their status changed. `a5fa7466` sent two CRITICALs with the respective game and state. `240324ab` archived both refusals and sent only the first CRITICAL. `src/bts/scheduler.py:2638` sends the same source for both, and `src/bts/health/alert.py:437` suppresses the second under the June 14 receipt. See `DISTINCT` and `DISTINCT WIRE` in [base](rollover-base.log) and [head](rollover-head.log).
- The same collision can suppress the first live refusal. In a second `run_day` replay, an empty cascade at 23:46 left the end-of-day digest to report a malformed automatic file. A restart at 23:59 kept the June 14 slate. Its cascade returned candidates at 00:01, when the loader found an invalid streak of `-1`. Base sent the live refusal and the updated historical digest. Head sent neither. The receipt came from the earlier end-of-day digest, before any live refusal. The live call is at `src/bts/scheduler.py:2029`. See `DIGEST FIRST` in [base](rollover-base.log) and [head](rollover-head.log).
- The K1 risk statement describes a second alert following an in-day send after midnight and the suppression of a covered digest. Both first sends here preceded midnight. The restart case also reverses the order of the senders. The smallest fix for the demonstrated paths is to key void refusals by the game IDs and states and give live contest refusals an incident key distinct from the generic end-of-day contest-state key, while retaining the slate date.

## Edits

- B8a at `docs/ops/2027-season-start.md:31`. Replace “The refusal names this command.” with “The refusal DM points to the scheduler journal and `bts season-contract status`. Use the dated command above when the timing and prior-entry checks below permit filing.” After the between-runs instruction, add “After the last lineup check, the final fallback needs a saved, undelivered pick. Filing only changes the dated observation.”
- `scripts/audit/season_walk/README.md`. Replace “after the finale every action is a skip; on the finale streak 56 plays” with “The default walk supplies synthetic candidates after the finale. Its raw day count is zero and fresh selection uses one decision day. Streak 56 plays on the scheduled finale and the makeup fixture. An explicit empty fixture yields no decision.” Update the effective-day formula to floor candidate slates at one before the policy horizon clamp.

## Notes

- The clocks, schedules, predictions and contest accounts were synthetic. The December replays extend the stored boundary for their synthetic slate to isolate K1. The off-day probes ran the real schedule and slot filters with fixture responses and stubbed model training. The box path was substituted into diagnostics produced from temporary files for the refusal length checks.
- The host refused the requested `nice` priority change and the process-list query. Direct runs stayed bounded to seconds and ran one at a time. Review artifacts are under `.codex-review/k12-read/`.

## 1. Refusal DM

- Read all live and historical `ContestStateError` branches, automatic authority refusals and saved-pick refusal branches. Historical-only branches feed health checks. A missing required observation is refused by the automatic authority check before the loader's defensive missing-state branch.
- [Refusal probe](refusal.py) ran 60 diagnostics with previous failure counts `1`, `999`, `1000` and `10**300` and the box path `/home/bts/projects/bts/data/picks` in path-bearing reasons. Head delivered all 240. The maximum complete body was 971 code points, including the banner. Every body carried the leading reason and the journal, status and B8a directions. Base delivered 181 and the captured transport rejected 59 for length. Found no input-only send regression. The 97 shipped refusal tests also passed, including empty or broken exception text, Unicode, invalid slate dates, and failed-send retry. [Head log](refusal-head.log) · [Base log](refusal-base.log) · [Tests](test-k1-refusal.log)

## 2. Dedup date

- [Rollover probe](rollover.py) ran both revisions through lineup checks, both fallback refreshes, both contest-state delivery branches, cutoff and void guards, and the missed-pick wait. This covered June 14–15 and December 31–January 1. Head wrote the slate date throughout and sent the next slate's own lineup alert. Base used the wall date and suppressed that next alert when an earlier-slate alert had already claimed it. The known covered end-of-day digest suppression reproduced after a cutoff alert. An end-of-day digest containing a new source still sent after a void refusal. The two additional losses are in the Verdict. [Head log](rollover-head.log) · [Base log](rollover-base.log)

## 3. B8a daily filing

- [B8a probe](b8a.py) expanded both commands from the checklist with the actual Zsh `read` instruction. The simulated service and operator used different working directories. The non-default picks path included a backslash and leading and trailing spaces in its last component. The real CLI wrote and read the same dated row there, with no operator-directory default state or saver flag created.
- At a late start, the 18:15 refusal saved no pick. Filing the verified row at 18:16 between checks let the scheduler deliver one pick on the later game. The control without filing delivered zero. When 18:15 was the last check and no candidate existed, there was no fallback and an idle-time filing did not trigger a pick. For a still-pending earlier slate after midnight, the checklist's same-date condition prevented filing. The old-date CLI negative control exited 2 and the observation bytes stayed identical. Found no new lost pick or wrong write from following the changed steps. [Probe log](b8a.log) · [Four checklist tests](test-k1-b8a.log)

## 4. K2 decision-day floor

- [Policy probe](policy.py) printed `bts.__file__` for the archived base and merged head and loaded the same shipped base and tail artifact hashes. All 59,180 complete decision rows across the 269 positive-day dates of 2027 matched, as did 130 fresh selections across 26 dates. Every changed day-count expression returns the same raw count when the count is already positive. [Head grid](policy-head-grid.log) · [Base grid](policy-base-grid.log)
- On September 27, `select_pick` and `orchestrator.run_and_pick` agreed with either saver state. Streaks 20 with best 20, 55, 57 and 120 doubled. Streak 56 singled. Streak 20 with best 25 skipped. Each used one decision day. The base, the direct resolver and the standalone bar kept their zero-day behavior. Empty, postseason-only and resumed-only inputs produced no pick through the scheduler, `bts run`, `bts preview`, `bts predict-json` and the shadow pipeline. [Head paths](policy-head-paths.log) · [Base paths](policy-base-paths.log) · [45 K2 tests](test-k2.log)

## 5. Other changes

- The makeup and empty season-walk records passed `walk.check`, and 24 standalone season-walk tests passed. The default walk still supplies candidates to every date in its range. Its result covers the decision policy, while the scheduler and prediction probes above cover off-days. The README edit records that distinction. [Walk tests](test-season-walk.log)

ran: refusal and rollover probes on base and head, B8a CLI and scheduler probe, policy grid and path probes on base and head, 170 focused tests; did not run: 46 tests requiring removed review artifacts, 2 frozen-walk tests, the full season walk and fast suite, network, box or contest account runs.
