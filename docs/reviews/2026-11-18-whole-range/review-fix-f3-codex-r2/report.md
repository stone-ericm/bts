# Verdict

**SIGN.** Amendment 3 resolves all round-1 findings. The checks found no defect introduced by `dee14603` and `490c5381` on `5e0167b5`.

## Blocking findings

None.

## Edits

None.

## Notes

- The base and F3 both return `no_decision` for a list-valued decision schema and both raise `TypeError` for an object-valued decision action. The original round-1 input script expected a traceback for both corruptions. The follow-up completed the remaining checks. [Base](inputs-base.log) · [F3](inputs-trial.log) · [Follow-up](evidence-edges-trial.log)
- The read limit is per file. With no valid installed observation, the round-1 archive check opened 81 archives and read 160,577 bytes in total. The 8 MiB file contributed 65,537 bytes. [Archive checks](archives-trial.log)
- A next-day dated observation can replace the only dated difference, and a later revoke can replace the same-day revocation timestamp. A matching archive can suppress the revoke reason. These are documented limits. [Carbon replay](carbon-trial-replay.log) · [Scheduler replay](scheduler-trial-replay.log)
- The sandbox rejected the `nice` priority adjustment. An initial launcher started subsequent test commands before the F1 run finished. I stopped the launcher and reran its test files serially. Victoria 3 was absent at the process checks.

## 1. Round-1 findings

- **B1 and K2c.** Replayed dated streak 1 at 09:00 ET, revoke at 09:01, enable and fetch automatic streak 30 at 09:02, and the 16:10 skip on that automatic 30 with saver unavailable. At 08:30 the base was clean. F3 returned exactly K2e, `skip_state_review`, exit 1 and one DM with the archived-observation reason only. The control with no differing archive was clean. Amendment 3’s K2c now matches both results. [Paired base](scheduler-base.log) · [F3](scheduler-trial-replay.log) · [Committed K2e test](round1-fixes.log)
- **E1.** Both `0001-01-01T00:00:00Z` and `9999-12-31T23:59:59-12:00` were ignored as installed and archived evidence. With a valid differing archive, an overflowing matching archive, and a valid same-day revocation, F3 returned both valid reasons and one DM. Adding an overflowing installed observation preserved that result. [Committed cases](round1-fixes.log) · [Independent CLI check](evidence-edges-trial.log)
- **E2.** Both `CLAUDE.md:66` and the module docstring match Amendment 3’s skip and optional-evidence wording verbatim after normalizing line wraps. [Documentation tests](round1-fixes.log)
- **Bytes read.** The four selected checks passed on the unchanged scratch copy. After changing it to read to EOF and then enforce the same size limit, the three oversized-file result checks still passed and `test_evidence_read_byte_limit` failed at the byte count, 262,432 against a limit of 65,537. [Control](byte-control.log) · [Read-to-EOF change](byte-mutant.log)

## 2. Introduced defects

- **Base parity.** Archived `5e0167b5 src` outside the checkout. Both runs printed and verified `bts.__file__`. All 18 paired cases covering every base outcome outside `deliberate_skip` matched byte for byte in stdout JSON, stderr, `message()`, exits and captured sends. Evidence files were present and none was opened or listed for these paths. [Base](parity-base.log) · [F3](parity-trial.log)
- **Scheduler and report.** Replayed 26 paired scheduler cases and both historical Carbon cases. M1–M4 each alerted once with the expected reasons. The same-day and next-year checks used the processed ET date. The base stayed clean in both Carbon cases and F3 raised the expected review. The 26 scheduler replays verified identical directory entries, modes and file bytes before and after the morning command. [Base scheduler](scheduler-base.log) · [F3 scheduler](scheduler-trial-replay.log) · [Base Carbon](carbon-base-replay.log) · [F3 Carbon](carbon-trial-replay.log)
- **Optional evidence.** Checked 45 input combinations covering the installed file, archive and switch. FIFO opens were nonblocking and read no bytes. A partial read followed by `EIO` closed the file and preserved the archive alert. Short reads of an oversized file stopped at 65,537 bytes. Partial archive enumeration retained evidence already found. No evidence fault raised an unexpected exception, exited 2 or blocked. The injected interrupt closed the descriptor and left files identical. Restart and failed or unconfirmed send behavior held, and invalid TOML still exited 2 with the existing could-not-run DM on both versions. [Base inputs](inputs-base.log) · [F3 inputs](inputs-trial.log) · [Fault and restart follow-up](evidence-edges-trial.log)
- **R64.** Checked the actual installed-only, archive-only, revoke-only and combined text in the scheduler and CLI results, including a wrong filed correction, the B8a restore, and the year boundary. The report identifies the skip’s saved inputs and what the files record. The instruction asks the operator to check MLB for unsettled entries and verify the current streak, season best and saver. In those sequences, following the instruction requires only reading the account and leaves picks, state and alerts intact. [Rendered reports and DMs](scheduler-trial-replay.log)

## 3. Tests

- F3 had 92 passes. The 12 existing pytest files found by the requested search had 487 passes and four skips. Each completed under two minutes. The initial cap for `test_wr_f1_wrong_game.py` was too short for its two deliberate 20-second FIFO waits. The serial rerun passed all 70 tests in 40.60 seconds. [F3](pytest-f3.log) · [F1 rerun](test_wr_f1_wrong_game-55s.log)
- The search also matched three scripts. Cron syntax and stubbed cron behavior passed, the leaderboard CLI printed help, and the PVR script ran on 21 synthetic days. [Cron](script-cron-serial.log) · [Leaderboard](script-leaderboard-serial.log) · [PVR](script-pvr-serial.log)
- The diff from `5e0167b5` to `490c5381` changed `CLAUDE.md` and `src/bts/day_outcome.py` and added the F3 tests. No base test changed. The reviewed paths at `82c04543` match `490c5381`, and tracked files in the checkout are untouched.

ran: paired archived-base and F3 parity, scheduler and Carbon checks, input and archive checks, R64 text checks, 13 pytest files, three offline script checks and the read-to-EOF scratch check; did not run: full suite, other round-1 scratch changes, live services, live cron or an authenticated leaderboard grab
