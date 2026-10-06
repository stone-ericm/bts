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
- **Box limits, from 2026-10-05** (Eric, row C1-4b-deferral). The infrastructure reviews are archived:
  `docs/audit/2026-10-05-c1-infra-codex-r1.md` (BLOCK) and `docs/audit/2026-10-06-c1-infra-codex-r2.md` (final round,
  BLOCK). Fixes: `788c3a7` for r1, `1fa872a` for r2.
  **Status: SIGNED 2026-10-06.**
  - **Rounds:** r1 and r2 BLOCK, then closure round r3 BLOCK (row C1-infra-review-r3).
  - **Round 4** (row C1-infra-review-r4) gave **SIGN** on `3d6ae3f`
    (`docs/audit/2026-10-06-c1-infra-codex-r4.md`).
  - The reviewer's conditional bound and the confinement and witness limits stand as stated.
  - Box jobs now need only their own candidate gates.
  - **Guard:** every job runs under `scripts/audit/c1/guard.py` in a `Delegate=yes` unit.
    - The guard sits in a `guard/` leaf and the job in `payload/`, niced 10.
    - The cap is the unit's **cumulative** CPU, at min(declared budget, remaining cycle budget). The guard acts at
      budget − cores × 5 s and kills the payload cgroup before any I/O.
    - It confirms the payload subtree empty through recursive `cgroup.events`. If emptiness is not confirmed,
      that is an enforcement failure.
    - It then writes a durable `TERMINAL_<unit>.json` (exit / overrun / oom / terminated / guard-error, with CPU).
    - `prepare` checks CPU readability and that `cgroup.kill` exists with write access; it makes no test write.
  - **Conditions (stated, not enforced):**
    - **Scheduling:** the guard (nice 0) is scheduled within 3 s of its 2 s poll. Any total over budget is still
      recorded as an overrun.
    - **Confinement:** the job's work stays in the `payload/` subtree. It must not move processes elsewhere in the
      unit, nor ask the user manager for another unit.
    - **Guard overhead:** the terminal CPU does not include the guard's own receipt and logging after it.
  - **Launcher:**
    - one canonical root and lock; unit names carry a random suffix and are never reused;
    - an active unit refuses the launch before any accounting;
    - a durable `PENDING_<unit>.json` precedes each start, and a failed `systemd-run` keeps it;
    - reconciliation moves nothing: the pause marker, then the ledger, then a durable `RECONCILED_<unit>.json`;
    - a receipt is trusted only if it is this invocation's and well formed. A missing or invalid receipt reserves
      the full budget and pauses C1; any non-exit result or a systemd timeout/oom-kill pauses **all** C1;
    - an ordinary failure (exit rc ≠ 0) blocks a same-name relaunch even after its journal entry is gone;
    - a release needs `RESUME_<unit>.json` bound to that marker's sha256, plus row `C1-resume-<unit>` recording
      exactly "**RULED <date>: RESUME `<unit>` overrun `<sha prefix>`**", with the source token Eric.
  - **Downloader:** `--verify` scans, decides and reports inside one writer-lock interval.
  - **Ledger:** non-finite or negative values are refused, and the rename is directory-fsynced.
  - **Known limit (preexisting, not addressed):** the season guard reads the latest realtime journal timestamp, so
    a clock rollback could select an older sleep line.
- **Verified on the box, failure direction** (each witness shows what it says, not more):
  - **First guard (`66e1b68`, superseded):** `c1-guardtest-20261006T000354Z` killed 3 children at 24.0 s of a
    36 s budget.
  - **Guard at `788c3a7`, run directly** (`c1-guardtest-*` units; receipts to a scratch folder; their CPU is in
    the real ledger):
    - **T1:** 3 burning children (one in its own session) under a 120 s budget gave `overrun` at 84.0 s, with the
      payload empty.
    - **T2:** the guard SIGKILLed; systemd stopped the unit and every burner died, with no receipt.
    - **T3:** `systemctl stop` gave `terminated` (signal 15).
    - **T4:** a detached `sleep 300` was recorded and killed.
    - **T5:** with no delegated unit, `guard-error`, and the job never started.
  - **Guard at `1fa872a`:**
    - **T6:** the job moved a `sleep 300` into a nested `payload/workers/` cgroup and exited. Result: `exit`,
      with `leftover: true`, the descendant killed, and nothing left.
    - **T1b:** the T1 overrun was repeated, giving `overrun` at 84.0 s of 120 s, with the payload confirmed empty.
  - **Real launcher smoke:**
    - `c1-infra-smoke-20261006T002211Z` (at `788c3a7`): reconciled with no pause.
    - `c1-infra-smoke2-20261006T004058Z-911349cb` (at `1fa872a`): PENDING, then TERMINAL `exit` 0, then a durable
      RECONCILED at `status`, with no pause and a 0.05 s guard row in the ledger.
  - Local tests exercise selected decision branches with systemd or cgroup state replaced. They do not establish
    Linux crash recovery, collected-unit failure handling, concurrent-launch behavior or real OOM/RuntimeMaxSec
    enforcement. At review pin `85adb83`, the permitted Mac suite reports 89 passed and two guard preexec failures.
    T1–T5 and the launcher smoke are separately supplied box observations. (At `1fa872a` the preexec cause is
    written to stderr, and those two tests pass outside Codex's sandbox.)

## Candidates

| Candidate | Registration | Exposure row | Gate before picks change | Status |
|---|---|---|---|---|
| 2: outcome / entry / restart watchdog | `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` | — (ops: no outcome read) | failure/recovery fixtures red→green; deploy-gating review SIGN on the watchdog and its producer prerequisites (bound entry receipt, I-207 reconcile receipt, consistent-capture mechanism); D7; operational acceptance after commissioning | **design FROZEN** (r1 BLOCK → r2 SIGN WITH EDITS). Build plan `docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md` (production; reviewed until SIGN; D7). **Hazard found 10/06:** `cron-setup-hetzner.sh` installs `check-pick-entered` unconditionally, so a plain `install` re-enables entry nags whatever the intent. The intent-aware cron (P4) fixes it; until P4 is deployed, re-check that line after any install. **P4 built 2026-10-06 (`1c0e759`), with P5/C1 and C2 (`04cde0a`); P1 entry receipt built 2026-10-05 (`5f5217e`, schema `docs/ops/pick-entry-receipt-v1.md`)** (dates per C-06: all 10/05 ET); each is under review until SIGN, then D7; not deployed |
| 4a: calibration map | `docs/sota_audit/2026-10-04-prereg-c1-calibration.md` | X-32 (fit and test), published before the first 2027 read | held-out proper-score improvement on the registered 2027 split; independent acceptance; D7; a separate boundary check before it changes play | **design FROZEN** (trio r1 SIGN WITH EDITS). Build plan `docs/superpowers/plans/2026-10-06-c1-r4a-build.md`. **Production prerequisite found 10/06:** the served-slate archive (`bts_slate_v1`) does not persist each row's `game_time` or schedule `status`, so §2's eligibility would exclude every 2027 row. The fix is to persist both (slate v2): production code, reviewed until SIGN, then D7, deployed before 2027 date 1 |
| 3: plate-appearance count model | `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` | X-34 (historical fit and prospective evaluation), published before those reads | earlier-fit / later-untouched proper scores; independent acceptance; D7; the June-null kill condition stays open unless a separate downstream test closes it | **design FROZEN**. Feeds re-acquired 10/04–05: 12,148 games, 0 failures, no 403/429; all 12,148 stored files bound to their receipts (`--verify`, 10/05).<br>**Limitations of this evidence (reconciled again 10/05 with the code review r2 N7 `--verify`; unit `c1-r3-verify-n7-20261005T180629Z`, no requests):**<ul><li>**Feeds:** 12,148 receipted, 12,148 bound, none mismatched, missing or unreceipted.</li><li>**Per-game receipts only:** the run predates per-attempt receipts. It wrote one intent and one `stored` receipt per game, so any retries and any response-level byte retention are unrecorded.</li><li>**Schedules:** the five schedules carry no receipts at all. Their bytes are pinned only by sha256 (the calendar-coverage note and 4b's `SCHEDULE_PINS`), so the new `--verify` reports them unbound (exit 1), and a resume of this acquisition would refuse them until an explicit, reviewed reconciliation.</li><li>**No synthetic receipts were created.**</li></ul><br>**Historical count build:** code review r1 BLOCK (F1–F9) and r2 FINAL BLOCK (N1–N8) (`docs/audit/2026-10-06-c1-r3-build-codex-r{1,2}.md`). Eric ruled one more round (register row C1-r3-build-review-r3); fixes `a1f673b`, `adec2bd`. **r3 BLOCK** (`docs/audit/2026-10-06-c1-r3-build-codex-r3.md`): N8 closed, N1–N7 partially closed, new findings R3-1 to R3-6. The round says no repair, run, hash capture or X-34 on its strength. **Eric ruled 2026-10-05 (row C1-r3-build-review-r4): fix all six, then one fourth round;** a plain SIGN leads to the hash capture, X-34 and the build; anything else defers rank 3 this cycle, with no fifth round. Fixes `b233585` (R3-1, R3-6), `da1143c` (R3-3, R3-4; MLB event-type snapshot), `5cca66e` (R3-2, R3-5); mutant red check 18/18 killed. **Review r4 = BLOCK** (`docs/audit/2026-10-05-c1-r3-build-codex-r4.md`): R3-1 to R3-4 and R3-6 closed, R3-5 partially closed, new R4-1 (receipt witnesses validated after duplicate collapse and per-outcome partitioning). **Rank 3 DEFERRED for this cycle** (register row C1-r3-deferral); no fifth round, no hash capture, X-34 or build. Nothing has read the inputs. The re-acquired feeds and receipts stay stored for a later cycle; R4-1's required changes are where it would resume. |
| 4b: longest-streak policy (D1 Option 2) | `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md` | to publish before any outcome-bearing replay | Eric's trade table (chance of 57 given up against expected longest streak gained, realized-sequence replay) approved under D7; tail/base pairing (checklist A2) | **design FROZEN**; gate amended 10/04 (trade table). **Code built** (`b2e7005`; Codex code r1 BLOCK → all 11 findings fixed; r2 pending). 2023-10-02: **ruled 10/05: NO PLAY** (replaces the backfill ruling; `3b9739b`, pre-registered; applied in `9bcb13b`: 2023 calendar ends 10-01, 182 days). Generator commit: Eric ruled (a) check first; **CONFIRMED BY REPRODUCTION** 10/05 (`8522412`, seed 42 2023-09-25..10-01, 70/70 rows exact, 5.6 CPU-min; register row C1-4b-generator-commit). Code review: r1 BLOCK, r2 BLOCK (`docs/audit/2026-10-05-c1-r4b-code-codex-r{1,2}.md`). Eric then allowed a third round (row C1-4b-review-r3). The r2 fixes are `b56d870`..`3648120`, and the record is corrected for N2 (8 LightGBM threads, not 16; `ad29869`). **r3 (a fresh session, whole range) = BLOCK** (`docs/audit/2026-10-05-c1-r4b-code-codex-r3.md`): the maths is unchallenged, but the execution and evidence controls still fail (B1–B7). **4b DEFERRED this cycle (Eric 10/05, Choice A; row C1-4b-deferral): no trade table; the r3 findings are the deferral record; nothing ran.** The r3 launcher findings B4/B5 (per-process LimitCPU, no cycle-wide pause after an overrun, no launch lock) apply to every C1 box job, not just 4b |
| Rank-1 prerequisites (no blend) | `docs/sota_audit/2026-10-04-prereg-c1-mlb-capture.md` | none for receipt-only capture; X-33 for the outcome-free 2026 concordance diagnostic, published before it runs | receipts bound to retained bytes from the first 2027 capture; all-player binding reported as **not established** unless independent evidence exists | **design FROZEN** (r1 BLOCK → r2 SIGN WITH EDITS) |

## Compute ledger
| Date | CPU-hours used (cumulative) | Note |
|---|---|---|
| 2026-10-04 | 0.0012 | cycle started; launcher verification jobs only (worktree `~/projects/bts-c1` at `4a18416`) |
| 2026-10-05 | 0.2223 | rank-3 feed re-acquisition (13.3 CPU-min) plus verification jobs |
