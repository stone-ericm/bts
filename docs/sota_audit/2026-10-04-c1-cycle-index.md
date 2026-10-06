# C1 (2027 Cycle 1): cycle index

**Approved by Eric 2026-10-04** (exposure register §C, D4): `docs/audit/2026-10-04-d4-cycle-proposal.md` as written, with rank 4b added. **Started 2026-10-04.**
- **Caps:** $0 of new spend; 100 CPU-hours on bts-hetzner, with a stop-and-report at 50.
- **Reviews:** at most 2 Codex rounds per design.
- **Stop rules:** proposal §5. **Owner gates:** D2 (independent acceptance), D7 (Eric approves every production change), D1's trade table for 4b.

## How C1 jobs run on the box
- **Code** runs from a pinned worktree, `~/projects/bts-c1`. The production tree `~/projects/bts` stays at the deployed commit.
- **Every C1 box job starts through the launcher:**
  `cd ~/projects/bts-c1 && .venv/bin/python -m scripts.audit.c1.launch run --name <candidate-step> --cpu-hours <budget> --max-hours <wall> -- <command>`
  `launch status` reports the ledger total. A job not started this way is outside the cap accounting, and is not allowed.
- **Ledger:** `~/projects/bts/data/hetzner_results/c1/compute_ledger.tsv` (restic-backed). It holds systemd's per-invocation CPU time for every stopped `c1-*` unit.
- **Known undercount:** systemd records a unit's CPU time at info level only above about one CPU-second (measured 10/04: a 0.6 s job left no record, a 1.4 s job did). Each smaller job goes uncounted, by under 1.4 s.
- **Verified on the box 10/04, both directions:** a real job was recorded; a second job while one was running was refused; the 50 CPU-hour checkpoint was refused (planted ledger, scratch data folder); and with a planted 99.999 CPU-hour ledger the per-job limit killed a CPU burner at 3.000 s.
- **Checkpoint:** at 50 CPU-hours the launcher refuses until `CHECKPOINT_50_ACK.json` exists in that directory. It is created only after Eric decides to continue.
- **Box limits, from 2026-10-05** (Eric, row C1-4b-deferral; C1 infrastructure review r1 fixes in `788c3a7`):
  - **Guard:** every job runs under `scripts/audit/c1/guard.py` in a `Delegate=yes` unit. The guard sits in a
    `guard/` leaf and the job in `payload/`, niced 10. The cap is the unit's **cumulative** CPU, at min(declared
    budget, remaining cycle budget). The guard acts at budget − cores × 5 s, kills the payload cgroup before any
    I/O, then writes a durable `TERMINAL_<unit>.json` (exit / overrun / oom / terminated / guard-error, with CPU).
  - **What the bound assumes:** the guard (nice 0) is scheduled within 3 s of its 2 s poll. That is not proved.
    Any total over budget is still recorded as an overrun.
  - **Escaping the unit:** a job could spawn work outside its cgroup by asking the user manager for another unit.
    That work would not be capped. C1 commands must not do this; it is a stated precondition, not something the
    guard enforces.
  - **Launcher:**
    - one canonical root (no alternate-root option) and one lock;
    - an active unit refuses the launch before any accounting;
    - a durable `PENDING_<unit>.json` precedes each start and is reconciled against the guard's receipt (its CPU
      enters the ledger even without a journal record);
    - a missing receipt reserves the full budget and pauses C1;
    - any non-exit result, or a systemd timeout/oom-kill, writes `OVERRUN_<unit>.json` and pauses **all** C1;
    - a release needs `RESUME_<unit>.json` bound to that marker's sha256, plus register row `C1-resume-<unit>`
      recording exactly "**RULED <date>: RESUME `<unit>` overrun `<sha prefix>`**" from Eric.
  - **Ledger:** non-finite or negative values are refused, and the rename is directory-fsynced.
  - **Known limit (preexisting, not addressed):** the season guard reads the latest realtime journal timestamp, so
    a clock rollback could select an older sleep line.
- **Verified on the box 2026-10-05/06, failure direction** (each witness shows what it says, not more):
  - **First guard (`66e1b68`, superseded):** `c1-guardtest-20261006T000354Z` killed 3 children at 24.0 s of a
    36 s budget; a second job was refused on that scratch root.
  - **Current guard, run directly** (`c1-guardtest-*` units; receipts to a scratch folder, so the real cycle was
    not paused; their CPU is in the real ledger):
    - **T1 (overrun):** 3 burning children, one in its own session, under a 120 s budget gave `overrun` at
      84.0 s (action point 80 s). The payload cgroup was empty after the kill, and no burner survived.
    - **T2 (guard SIGKILLed):** systemd stopped the unit and every burner died. No receipt was written, which
      the launcher treats as unreconciled: a pause.
    - **T3 (`systemctl stop`):** a `terminated` receipt (signal 15), with the payload empty.
    - **T4 (leftover descendant):** a job leaving a detached `sleep 300` gave `exit` 0, with the leftover pid
      recorded and killed.
    - **T5 (no delegated unit, run in an ssh session scope):** `guard-error` (permission denied on the cgroup
      split), and the job never started.
  - **Real launcher smoke** (`c1-infra-smoke-20261006T002211Z`, a 3 s sleep): PENDING then TERMINAL `exit` 0,
    reconciled at `status` into `jobs/` with no pause. Its 0.05 s entered the ledger as a guard row (the journal
    skips jobs under about 1 s).
  - **Not witnessed on the box:** an OOM kill, a real `RuntimeMaxSec` timeout, a concurrent second launcher and
    a lost journal. Those are covered only by unit tests with systemd replaced.

## Candidates

| Candidate | Registration | Exposure row | Gate before picks change | Status |
|---|---|---|---|---|
| 2: outcome / entry / restart watchdog | `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` | — (ops: no outcome read) | failure/recovery fixtures red→green; deploy-gating review SIGN on the watchdog and its producer prerequisites (bound entry receipt, I-207 reconcile receipt, consistent-capture mechanism); D7; operational acceptance after commissioning | **design FROZEN** (r1 BLOCK → r2 SIGN WITH EDITS) |
| 4a: calibration map | `docs/sota_audit/2026-10-04-prereg-c1-calibration.md` | X-32 (fit and test), published before the first 2027 read | held-out proper-score improvement on the registered 2027 split; independent acceptance; D7; a separate boundary check before it changes play | **design FROZEN** (trio r1 SIGN WITH EDITS) |
| 3: plate-appearance count model | `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` | X-34 (historical fit and prospective evaluation), published before those reads | earlier-fit / later-untouched proper scores; independent acceptance; D7; the June-null kill condition stays open unless a separate downstream test closes it | **design FROZEN**. Feeds re-acquired 10/04–05: 12,148 games, 0 failures, no 403/429; all 12,148 stored files bound to their receipts (`--verify`, 10/05).<br>**Limitations of this evidence (reconciled again 10/05 with the code review r2 N7 `--verify`; unit `c1-r3-verify-n7-20261005T180629Z`, no requests):**<ul><li>**Feeds:** 12,148 receipted, 12,148 bound, none mismatched, missing or unreceipted.</li><li>**Per-game receipts only:** the run predates per-attempt receipts. It wrote one intent and one `stored` receipt per game, so any retries and any response-level byte retention are unrecorded.</li><li>**Schedules:** the five schedules carry no receipts at all. Their bytes are pinned only by sha256 (the calendar-coverage note and 4b's `SCHEDULE_PINS`), so the new `--verify` reports them unbound (exit 1), and a resume of this acquisition would refuse them until an explicit, reviewed reconciliation.</li><li>**No synthetic receipts were created.**</li></ul> |
| 4b: longest-streak policy (D1 Option 2) | `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md` | to publish before any outcome-bearing replay | Eric's trade table (chance of 57 given up against expected longest streak gained, realized-sequence replay) approved under D7; tail/base pairing (checklist A2) | **design FROZEN**; gate amended 10/04 (trade table). **Code built** (`b2e7005`; Codex code r1 BLOCK → all 11 findings fixed; r2 pending). 2023-10-02: **ruled 10/05: NO PLAY** (replaces the backfill ruling; `3b9739b`, pre-registered; applied in `9bcb13b`: 2023 calendar ends 10-01, 182 days). Generator commit: Eric ruled (a) check first; **CONFIRMED BY REPRODUCTION** 10/05 (`8522412`, seed 42 2023-09-25..10-01, 70/70 rows exact, 5.6 CPU-min; register row C1-4b-generator-commit). Code review: r1 BLOCK, r2 BLOCK (`docs/audit/2026-10-05-c1-r4b-code-codex-r{1,2}.md`). Eric then allowed a third round (row C1-4b-review-r3). The r2 fixes are `b56d870`..`3648120`, and the record is corrected for N2 (8 LightGBM threads, not 16; `ad29869`). **r3 (a fresh session, whole range) = BLOCK** (`docs/audit/2026-10-05-c1-r4b-code-codex-r3.md`): the maths is unchallenged, but the execution and evidence controls still fail (B1–B7). **4b DEFERRED this cycle (Eric 10/05, Choice A; row C1-4b-deferral): no trade table; the r3 findings are the deferral record; nothing ran.** The r3 launcher findings B4/B5 (per-process LimitCPU, no cycle-wide pause after an overrun, no launch lock) apply to every C1 box job, not just 4b |
| Rank-1 prerequisites (no blend) | `docs/sota_audit/2026-10-04-prereg-c1-mlb-capture.md` | none for receipt-only capture; X-33 for the outcome-free 2026 concordance diagnostic, published before it runs | receipts bound to retained bytes from the first 2027 capture; all-player binding reported as **not established** unless independent evidence exists | **design FROZEN** (r1 BLOCK → r2 SIGN WITH EDITS) |

## Compute ledger
| Date | CPU-hours used (cumulative) | Note |
|---|---|---|
| 2026-10-04 | 0.0012 | cycle started; launcher verification jobs only (worktree `~/projects/bts-c1` at `4a18416`) |
| 2026-10-05 | 0.2223 | rank-3 feed re-acquisition (13.3 CPU-min) plus verification jobs |
