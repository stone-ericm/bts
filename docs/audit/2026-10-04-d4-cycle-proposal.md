# D4 cycle proposal: name, cap and stop rules for 2027 Cycle 1 (for Eric's approval)

**Status:** **APPROVED by Eric 2026-10-04 as written, with rank 4b added** (exposure register §C, D4). C1 started 10/04. The proposal text below is unchanged except §7, which now records his choice. Not Codex-reviewed: each candidate's registration gets its own Codex review before it runs.

## What Eric has already decided (10/04)
- **Scope:** ranks 2, 4a and 3 first, on the existing box. Rank 1 gets its prerequisites only, with no blend.
- **D1:** optimize for a better expected longest streak, accepting a stated loss in the chance of 57. The measured trade must come to him for approval before any such policy affects picks.
- **D2:** one named change at a time, and only after its own registered gate and an independent acceptance.

## The proposal, in plain words
One cycle, called **C1**. It spends **no new money** and uses at most **100 CPU-hours** on the existing box, with a stop-and-report at 50. It covers the watchdog, the calibration map, the plate-appearance count model and the MLB-forecast prerequisites.
- Each candidate is registered (frozen, Codex-reviewed) before any 2027 result exists.
- Each is tested on 2027 games it was not fitted on.
- None touches picks without an independent acceptance and Eric's D7 approval.
- The cycle stops early on any cost, a compute overrun, any disruption to production, any 403/429 from MLB, or any attempt to widen scope.
- It ends at the close of the 2027 contest; any candidate unfinished by then is recorded as inconclusive and not shipped.

## 1. Name
**C1** ("2027 Cycle 1").
- Files: `docs/sota_audit/<date>-prereg-c1-<candidate>.md` and `…-result-c1-<candidate>.md`.
- Box units: `c1-<candidate>-<step>`, so the compute ledger can find them.

## 2. Money cap: $0 of new spend
- The existing Hetzner box (a fixed monthly cost) and the existing Claude and Codex subscriptions only.
- No new cloud machines, paid APIs or data purchases.
- Anything that would cost money stops the cycle and goes to Eric.

## 3. Compute cap: 100 CPU-hours on bts-hetzner for the whole cycle, with a stop-and-report at 50
- **How it is measured:** each research job runs as a transient `c1-*` systemd unit. The cycle ledger (`data/hetzner_results/c1/compute_ledger.tsv`) sums the "Consumed … CPU time" lines that systemd writes when each unit stops.
- **Why 100 (reasoning, sized to evidence where it exists):**
  - The largest wrap job, the full W1.2 bridge on 10/04, used 5.3 CPU-hours. All of 10/04's wrap runs together used about 7.7.
  - Rank 3 is the only heavy candidate. It may need one fresh walk-forward pass of plate-appearance predictions over 2021–2025. That pass has not been measured on the box; the repo notes say a 5-season backtest takes 2–3 hours of wall time, which on 8 cores is at most about 24 CPU-hours. Count-model fits come on top.
  - Ranks 2 and 4a and the rank-1 prerequisites should each need well under an hour of box CPU.
  - 100 covers rank 3 twice over with a margin. The checkpoint at 50 catches a runaway early.
- **Per job:**
  - one research job at a time on the box;
  - `nice 10`, `MemoryMax=12G` (of 15 GB), `OOMScoreAdjust=1000`;
  - during the 2027 season, only inside the scheduler's sleep window.
- **The Mac:** test suites there are not counted. No multi-hour run goes on the Mac.

## 4. Review cap (attention)
- **Registrations:** each gets at most 2 Codex rounds (the pace rule). One without a SIGN after 2 rounds is **deferred**, not forced.
- **Production code (rank 2):** reviewed until SIGN, under the deploy rules.
- **Results:** each candidate gets one independent Codex acceptance memo, at most 2 rounds.

## 5. Stop rules
**Cycle-level stops.** Any one of these pauses the whole cycle, and resuming needs Eric:
1. **Money:** any cost above $0.
2. **Compute:** 50 CPU-hours means stop and report; 100 means stop.
3. **Production safety:** a research job that delays or disrupts a scheduler run, the dashboard, the nightly backup or a pick is killed at once and not restarted without Eric.
4. **Field capture:** any 403 or 429 during the rank-1 captures stops that capture, with no rerun without a fresh decision (the standing field-capture rule).
5. **Scope:** no candidate joins mid-cycle. Adding 4b, the rank-1 blend, or ranks 5, 6 or 7 needs a recorded scope decision.
6. **Calendar:** the cycle ends on the last day of the 2027 contest, as recorded under checklist A3. A candidate without a disposition then is recorded **inconclusive** and does not ship.

**Per-candidate stops.** These are the plan's kill conditions; each registration fixes its numbers before any 2027 outcome exists.

| Candidate | Stops if | Done when |
|---|---|---|
| 2: outcome / entry / restart watchdog | it adds state races, or cannot tell private picks from contest entries | its declared failure/recovery fixtures go red then green and the deploy-gating review signs. It ships only with Eric's D7 approval, before activation if he plays |
| 4a: calibration map (identity vs one regularized intercept map) | held-out proper scores don't improve by the registered practical threshold | its registered 2027 fit window and later test window are scored and an independent acceptance is recorded. If it passes, a separate policy-boundary check is still needed before it may change what gets played |
| 3: plate-appearance count model | it needs actual-exposure inputs; its hit target changes inconsistently; or it shows nothing beyond the June PA-tilt null | its pre-lock count forecasts are archived from the first 2027 game, scored on later untouched dates, and independently accepted |
| Rank-1 prerequisites (no blend) | the receipt log needs a new authenticated operation outside the field-capture rules, or the all-player field cannot be bound to a game without outcome data | per-attempt receipts bound to response bytes run from the first 2027 capture, and the all-player binding question is answered before any 2027 outcome is read |

## 6. Order
- **Before Opening Day 2027 (box idle):**
  - build and review rank 2;
  - write, Codex-review and publish (exposure rows) the 4a and 3 registrations, with their 2027 fit/test date splits and thresholds;
  - set up rank 3's pre-lock count-forecast archive and the rank-1 receipt log so both run from the first 2027 capture.
- **In season:** 4a and 3 are fitted on their registered early window and scored on later untouched dates. Nothing changes picks before an independent acceptance and Eric's D7 approval.

## 7. Not in this proposal: a policy built on D1 (Eric's choice)
- **What changed.** D1 names the objective (a better expected longest streak), which was rank 4b's missing piece. A full-season policy for it would be a policy-only change (4b), and 4b is outside the approved scope.
- **What already exists.** The 9/03 tail policy already optimizes the expected season best, but only once the chance of 57 is gone.
- **Two options for Eric:**
  - **(a)** add 4b to C1 by a recorded scope decision. Its registration would size its compute and might raise the cap. Its gate is his D1 condition: a measured table of the chance of 57 given up against the expected longest streak gained, on realized-sequence replay, put to him under D7 before it affects picks.
  - **(b)** leave it for a later cycle.
- **Decided (Eric, 10/04): option (a).** Rank 4b, the longest-streak policy serving D1 Option 2, is part of C1. Its gate is his trade table: no 4b change affects picks until he has seen the measured trade between the chance of 57 given up and the expected longest streak gained, and approved it under D7. It shares C1's caps and stop rules; its registration sizes its compute within the 100 CPU-hours.
