# Whole-range fix set B

Branch `fix/wr-b` from `9e0ec28a`. Commit `2dce391c4ad8a04c22af1963c80851e47829cf71`. The fast suite matched the saved host baseline exactly.

## B1 · Saver warning

`src/bts/health/contest_state.py:178` changes the flag-based `not_earned` contradiction at streak 10–15. Before, with the opening probe's values:

> Streak Saver flag not_earned for season 2027 contradicts streak 11; counting the saver as spent. Check the saver in MLB and run ONE of these. Earned and unused: `bts saver-state --init active --force --season 2027`. Already used: `bts saver-state --use --season 2027`. If MLB shows the saver not earned yet, streak 11 is a stale contest reading (dated 2026-09-27): leave the flag as it is and correct the contest streak with bts set-contest-streak

After, with `<quoted picks dir>` standing in for the invocation's shell-quoted path:

> Streak Saver flag not_earned for season 2027 contradicts streak 11; counting the saver as spent. Check the saver in MLB and run ONE of these. Earned and unused: `bts saver-state --init active --force --season 2027`. Already used: `bts saver-state --use --season 2027`. If MLB shows the saver not earned yet, streak 11 is a stale contest reading (dated 2026-09-27): leave the flag as it is and correct the contest streak with `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir <quoted picks dir>`. Replace N, B and S with MLB's active streak, best streak and saver state (one of not_earned, active or used).

The AST inventory assembles command expressions containing `bts set-contest-streak`, including split f-strings and concatenation. Click registration, comments and identifiers are outside this search. Audited `health/contest_state.py:144` (dated contradiction's active, used and not-earned remedies), `scheduler.py:88` and `:201` (post-delivery and morning stale advice), and `season_contract.py:98` (shared dated template, used by live refusal and revoke). Those templates retain their bytes. The named RED `test_saver_flag_warning_command_repairs_live_state` failed to find a quoted dated command on base. It substitutes MLB values into the actual rendered command and checks live dated authority and unchanged saver flag.

## B2 · Entry watchdog for unconfirmed delivery

`src/bts/cli.py:2120` uses the same-day scoreable decision already read by the watchdog. The names come from the checked `DailyPick`. Before, at the initial tier:

> ⚠️ BTS pick NOT entered in MLB app: {names} — first pitch {pitch} ({minutes} min to submit). Fix it now!

For initial mismatch the lead was `BTS entry does NOT match the recommended pick`. Before, at both repeat tiers:

> ⏰ STILL NOT entered ({reason}): {names} — {minutes} min left to submit!

After, at initial, T-30 and T-15:

> ⚠️ Delivery of this pick was NOT confirmed. {names} is the pick the scheduler tried to send. No entry found in MLB. First pitch {pitch} ({minutes} min to submit). Check your DMs and MLB entry before entering anything, and run `bts season-contract status --picks-dir {quoted picks dir}`.

For mismatch, `The MLB entry differs.` replaces `No entry found in MLB.` The original text remains in use for delivered, private and legacy fallback picks. The receipt, marker, escalation and all-clear paths retain their existing behavior.

`test_unconfirmed_nags_verify_delivery_at_each_tier[locked_unconfirmed]` failed on the base's initial `Fix it now!` DM. The test checks all tiers and their receipts and compares delivered, private and legacy fallback DMs byte-for-byte. `test_crash_then_corrected_real_scheduler_replay` runs the slice's crash before transport acceptance, a dated correction and real scheduler restart to `locked_unconfirmed`. The watchdog runs at three simulated cron times and EOD still reports `not_entered`.

## B3 · Executed names in day-outcome

`src/bts/day_outcome.py:46` supplies the shared JSON and DM value. Before, a single with a populated runner-up printed JSON `"pick": "Batter 101 + Batter 102"` and appended `Pick: Batter 101 + Batter 102.` to the problem DM. After, it prints `"pick": "Batter 101"` and appends `Pick: Batter 101.` An actual double uses both usable names. Empty, non-string or malformed executed summaries are omitted. The single's unused runner-up is never examined.

`test_single_outcome_names_only_the_executed_batter` on base showed the runner-up beside the selection. After the change it exercises real CLI output with and without a published, sealed, decision-bound positive receipt. It checks exit code, classification, decision bytes, problem DM and no DM on confirmation. Controls cover a true double and malformed summaries.

## B4 · Automatic authority behind today's dated observation

`src/bts/cli.py:2178`, `src/bts/season_contract.py:65` and `:186`, `src/bts/contest_state.py:307` and `:533`, `src/bts/saver_state.py:45`, and `src/bts/util.py:13`. Status supplies the live decision loader with the captured clock, active-year interpretation of its switch snapshot and a regular-file reader. It reuses the live result unless it selected today's dated observation. In that case, it calls the same decision logic with only that chosen manual record omitted. The second call uses the normal automatic parsing, activation, freshness and saver checks. Switch, manual, automatic, model, saver, root pick and ledger reads made by status all avoid blocking on non-regular files. The status-only reader verifies the opened descriptor.

Before, after the existing `live decision now: …` output, status went directly to `ledger: …`. After, between them it prints either `automatic path alone: accepted (streak {n}, recorded {time})` or `automatic path alone: refused: {first refusal line}`. Timestamp text is `unavailable` when unusable. Unexpected decision read errors also get a one-line refusal diagnostic. The final line when today's dated observation masks an automatic refusal is:

> Today's dated observation expires at midnight ET. Tomorrow's live pick, if games are scheduled, requires a usable automatic reading or a new dated observation.

`test_status_exposes_automatic_refusal_masked_by_dated_observation` failed on base because the automatic verdict was absent. It covers four dated/automatic combinations. Edge tests cover a winning legacy override, corrupt manual/auto/switch, wrong activation, missing/naive/stale/future timestamps, special files, an expired dated observation, the active-year verdict while inspecting another year, interrupted status reads and read-only disk state. The actual cutover commands also show the warning clearing after a matching fetch.

## B5 · Fetch status

`src/bts/cli.py:1719`, `:1745`, `:1753` and `:2454` use the shared regular-file decoder at both status reads and atomically replace the status file. Unusable content is treated as empty. Malformed timestamps have no cooldown power. A category map stays authoritative even if entries were discarded. A valid legacy timestamp applies to all categories only when the map is missing or malformed. Valid legacy and category fields survive success. A DM with no acknowledgement ID leaves no cooldown stamp.

Normal failure DM text stays `BTS fetch-contest-streak failed: {msg}`. Before, unusable status could abort during the read with an uncaught exception or block on a FIFO. After, a writable target is replaced and fetch proceeds through its existing checkpoints. Before, failure to replace produced an uncaught `OSError` (exit 1). After, it reports `fetch-contest-streak: cannot replace {path}: {error}` and exits 2. An empty DM acknowledgement now prints `(DM delivery was not confirmed)` with no cooldown stamp. The existing `(DM failed: {exc})` reports raised transport errors.

`test_fetch_repairs_bad_status_on_success_and_failure` failed on base on both real CLI lanes for array/null and invalid UTF-8, and on the failure lane for an integer stamp. Tests require a full switched success through the ledger and saver, and a failure DM and exit 2 with prior observation and ledger effects intact. Bounded FIFO and directory cases exercise atomic replacement and refusal. Existing prefix controls now assert exit 2 for status replacement errors. The switch-read count spy observes the nonblocking descriptor open. Two existing successful DM fakes return a confirmation ID to keep their cooldown controls representative.

Lead amendment 1 addresses two additional expectations exposed by the first fast run.

- `tests/test_cli_integration.py:953`, `test_auth_failure_dm_throttled`. Before, the transport fake was `lambda h, m: dm_calls.append(m)` and `assert len(dm_calls) == 1` expected the first send to start cooldown. After, the fake returns `dm_calls.append(m) or "sent-id"` and the assertion remains `assert len(dm_calls) == 1`. The test now supplies an acknowledged delivery. It follows B5's “Failed/unconfirmed DM or no recipient cannot add a cooldown stamp.”
- `tests/test_season_contract_cs4b2.py:480`, `test_restore_reenable_fetch_sequence`. Before, `assert result.exit_code == (0 if failure is None else 1)`. After, `assert result.exit_code == {None: 0, "before": 1, "after": 2}[failure]`. The `after` case fails at status replacement after installing the observation. It follows B5's “If the target cannot be replaced, for example a directory or unwritable parent, diagnose the path and exit 2 with the prior completed effects intact.”

## B6 · Cutover checklist

`docs/ops/2027-season-start.md:31`. Before, B8a used `live decision now: accepted` after enable and required an unconditional saver init. After, the post-enable fetch must succeed, the active ET season must report `automatic path alone: accepted`, and the operator reads `bts saver-state`. Only an `uninitialized` flag calls for initialization using MLB's actual state and the current ET year. The row covers interruption, rereading after concurrent initialization and resuming scheduling on a usable dated observation while automatic cutover is pending.

`test_b8a_requires_successful_fetch_automatic_readiness_and_saver_read` failed on base's missing readiness and conditional saver gates. The base checklist probe reproduced the unconditional init's exit 1 after cold initialization. `test_cutover_rehearses_masked_and_matching_authority` runs enable, failed fetch, status, successful fetch, status and saver read against settled and unsettled fixtures.

## Standing hostile inputs

| Row | Interrupt, failed or uncertain send, and restart | Malformed or special input | ET midnight and New Year |
|---|---|---|---|
| B1 | Interrupt and lost response in real health dispatch, retry, acknowledgement, dedup on restart, then a bundled CRITICAL and saver WARN. No new warning reads or writes. | Existing malformed contest and uninitialized saver controls run through the same check. | Warning date from the check instant despite prior-season coverage and a prior-day dispatch key. Old command refused after midnight. New date accepted after replacing values from MLB. |
| B2 | Simulated scheduler crash before acceptance and later correction, three real watchdog nags, plus interrupt and lost-response injection on the watchdog. Retry uses the old day and a checkpointed nag permits a confirmed all-clear. Failed all-clear DM retains confirmation. | Absent decision, rejected JSON shape and non-regular decision directory exercise delivered-pick fallback. Existing decision, pick, receipt and marker read paths are preserved. | CLI begins before midnight, auth advances the clock into Jan 1, and marker and receipt are bound to the invocation's original date. |
| B3 | DM raises either interrupt or timeout. The real stateless command can run again against the same decision bytes. | Valid and malformed leg summaries, zero receipts, and a sealed bound positive receipt. Existing schedule and receipt error controls apply. | Default captured before schedule read crosses New Year. Explicit replay uses the same prior-year slate. |
| B4 | Both interrupts injected during status reads, with state checked before restart. No send in status. | Bounded FIFOs in switch, manual, auto, saver, ledger, dated-pick history and model lanes. Corrupt authority files keep refusals and diagnostics. | Dec 31 current-year activation tested with `--season` pointing to another year, then Jan 1 refusal after expiry without new activation. |
| B5 | Both interrupts after observation, status, ledger and failure send. Retry resumes from disk. Lost/unconfirmed DMs have no stamp. | Array, null, invalid UTF-8, malformed stamp, FIFO and directory. Status JSON remains atomic. | Existing captured-clock tests advance time during auth or persistence, checking season-bound observation, ledger and saver writes and UTC timestamps. |
| B6 | Procedure requires verification again after interruption and rereading an existing flag after a concurrent fetch. The procedure adds no send. | Failed fetch or readiness refusal keeps the accepted dated path in use. | Readiness and saver steps refer to the active ET season and require rechecking day/year before acting on old output. |

## Scope and verification

No edits to `strategy.py`, `scheduler.py`, `orchestrator.py`, `bts run`, the schedule producer, deployment workflow, shared fixtures, or 4c/6b-1 exact-text tests. B7 combined checks apply after the lead merges fix sets A and B. The B-only rehearsal confirms the existing crash recovery and watchdog behaviour. Post-merge checks still need to exercise A2 refusal before any attempt, A5 malformed schedule flowing through the outcome DM, valid off-days and dated identity after B1 correction. A3/A4 are separate change sets per R45 and remain outside this build.

- Opening rehearsal first on `9e0ec28a`: all 14 opening and restart scenarios, 42 slate outcomes. Artifacts in `.codex-review/wr-b/opening/`.
- Targeted base playback in `.codex-review/wr-b/base-reproductions.log`: W6–W10 and W19 reproduced with synthetic state and captured transports. The report driver needed its slice path corrected at runtime. The legacy fixture used the local Git archive of `619abda7`.
- RED `.codex-review/wr-b/red.log`: 18 failed, 4 passed. Every B1–B6 test showed its targeted failure. Two extra errors in the initial controls came from fixture setup (legacy delivery ID and writing the malformed summary), since corrected.
- Initial sandboxed focus `.codex-review/wr-b/focused-first.log`: 112 passed, 8 failed. Six prior assertions still expected exit 1 for a status replacement error. One saver audit test's filename-only read spy rejected a descriptor, and the web saver loopback bind hit sandbox policy. The next run included the status assertion and descriptor-reader corrections.
- Expanded sandboxed focus `.codex-review/wr-b/focused.log`: 308 passed. The sole failure was the web saver test's loopback bind, denied by the sandbox. It was included in the requested same-process run with the LightGBM import test and the new contract and edge tests.
- Replay check `.codex-review/wr-b/replay.log`: 124 passed, 1 failed. The imported scheduler fixture initialized under the surrounding pinned clock. Initializing it with the real datetime type restored its scoped simulated clock.
- Follow-up `.codex-review/wr-b/replay-verified.log`: 268 passed, including the adopted scheduler crash fixture and the unmodified 4c/6b-1 pinned-text tests.
- Committed-tree fast suite on `2318ac1b` in `.codex-review/wr-b/fast.log`: 9,304 passed, 39 failed, 16 errors, 25 skipped, 6 deselected, 1 xfailed. LightGBM optional and all 12 web saver tests passed. Exact comparison found all 53 expected IDs and two additional failures: `test_auth_failure_dm_throttled` mocked successful delivery without an ID, and `test_restore_reenable_fetch_sequence[after]` still expected exit 1 for a status write error.
- Approved committed-tree rerun on `2dce391c` in `.codex-review/wr-b/fast-amended.log`: 9,306 passed, 37 failed, 16 errors, 25 skipped, 6 deselected, 1 xfailed. Both amended expectations passed, as did all whole-range B, LightGBM optional, and web saver tests in the same process. The failed/errored ID set equals the 53 saved IDs in `~/bts-runs/gate-failed.txt`, with zero extra and zero missing. No baseline rerun. Wrapper exit 1 reflects those expected failures.
