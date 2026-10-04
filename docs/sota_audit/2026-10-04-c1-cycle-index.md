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

## Candidates

| Candidate | Registration | Exposure row | Gate before picks change | Status |
|---|---|---|---|---|
| 2: outcome / entry / restart watchdog | `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` | — (ops: no outcome read) | failure/recovery fixtures red→green; deploy-gating review SIGN on the watchdog and its producer prerequisites (bound entry receipt, I-207 reconcile receipt, consistent-capture mechanism); D7; operational acceptance after commissioning | **design FROZEN** (r1 BLOCK → r2 SIGN WITH EDITS) |
| 4a: calibration map | `docs/sota_audit/2026-10-04-prereg-c1-calibration.md` | X-32 (fit and test), published before the first 2027 read | held-out proper-score improvement on the registered 2027 split; independent acceptance; D7; a separate boundary check before it changes play | **design FROZEN** (trio r1 SIGN WITH EDITS) |
| 3: plate-appearance count model | `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` | X-34 (historical fit and prospective evaluation), published before those reads | earlier-fit / later-untouched proper scores; independent acceptance; D7; the June-null kill condition stays open unless a separate downstream test closes it | design rev 2 in Codex round 2 (r1 BLOCK) |
| 4b: longest-streak policy (D1 Option 2) | `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md` | to publish before any outcome-bearing replay | Eric's trade table (chance of 57 given up against expected longest streak gained, realized-sequence replay) approved under D7; tail/base pairing (checklist A2) | **design FROZEN** (r1 BLOCK → r2 SIGN WITH EDITS). Open owner question: a separate 2027 validation contract, or a recorded amendment making the historical screen + trade table the gate |
| Rank-1 prerequisites (no blend) | `docs/sota_audit/2026-10-04-prereg-c1-mlb-capture.md` | none for receipt-only capture; X-33 for the outcome-free 2026 concordance diagnostic, published before it runs | receipts bound to retained bytes from the first 2027 capture; all-player binding reported as **not established** unless independent evidence exists | design rev 2 in Codex round 2 (r1 BLOCK) |

## Compute ledger
| Date | CPU-hours used (cumulative) | Note |
|---|---|---|
| 2026-10-04 | 0.0012 | cycle started; launcher verification jobs only (worktree `~/projects/bts-c1` at `4a18416`) |
