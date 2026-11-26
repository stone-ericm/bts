# Season rollover audit F4–F8 handover

Branch `sb-audit-f4f8`, based on `3d49308d`. Five original local Tier B commits and two follow-ups for F4 and F5. Selection, delivery and scoring remain as before. No push, external network, box or contest account was used.

## F4 · Opening-night contest coverage

- Original commit `af0255cb`. Follow-up `4257641f`. `src/bts/health/contest_state.py:73–99`. `tests/test_contest_dated_cs4a.py:71–137`.
- Reproduced on the automatic path on March 24 and 25, 2027. `test_opening_eod_automatic_coverage` was red twice on the base with `CRITICAL` in place of `INFO`. Both pass after the change. March 26 and a March 25 check with a March 23 reading remained critical in both runs.
- The dated path was green on the base on both days. `test_green_control_opening_eod_dated_observation` uses the real `set-contest-streak --date` CLI and the expected health check. The dated observation supplies its own effective date as `source_date`.
- Limit. Only the missing-source-date alert for a selected automatic file is lowered. Its timestamp must parse, be in the current ET year and be from the last 36 hours through now. Exactly one or two current-year root-level picks must be settled and the latest settlement must be on or before today ET. The existing final live refusal, loader errors, saver warnings and other freshness checks still apply.
- Follow-up reproductions. The April 20–21, 2027 moved opener and the 2028 case were red before the follow-up. The third-settlement, stale timestamp, prior ET year, future timestamp and future-settlement cases stayed critical. Existing test expectations were retained.

## F5 · Park drag

- Original commit `46e16b67`. Follow-up `153ff8c2`. `src/bts/health/park_drag_freshness.py:75–103`. `src/bts/strategy.py:46–64`. `src/bts/features/park_drag_producer.py:307–360`. `docs/ops/2027-season-start.md:32`. Tests at `tests/health/test_park_drag_freshness.py:34–64` and `tests/test_park_drag_producer.py:188–201`.
- Both paths failed on the base. The 2026-09-27 table paged during the March carryover gap, and the header-only store failed with a NaN-to-integer conversion. The tests pass after the change, including the opening-date fixture and first 2027 game.
- Limit. A source date in any prior year counts as carryover. When today's year has a configured `SEASON_OPENER_DATES` entry, carryover lag is zero through the opener and accrues from that date. For the default March 24, 2027 opener, the refresh first reaches it on March 25, a still-carried table first warns on March 27, and the first critical is March 30. Unconfigured years and current-year data use the full existing gap. Producer-error, missing-artifact and cron-liveness alerts still apply. The ops checklist calls for re-checking the opener table alongside the end dates after a labor stoppage.
- Follow-up reproductions. The moved 2027 opener and configured 2028 opener were red before the follow-up. The existing 2027 fixture and the full-gap alarm in unconfigured 2028 stayed green. The producer was untouched by the follow-up.
- Limit. Empty pitch stores start a March 1 window for the current year and adopt the fetched data's types on first ingest. A refresh with empty current and static data records a successful zero-row refresh without emitting an export. Nonempty stores retain the prior window, stale-store refusal, physics and builder. Existing nonempty end-to-end and idempotency fixtures and the independent numerical as-of assertion passed. The optional reference-analysis-repository comparison was skipped because that repository is absent. The existing fake HTTP fixture was parameterized by date with its original default and assertions intact.

## F6 · Contest fetch cron

- Commit `ecc977ba`. `scripts/cron-setup-hetzner.sh:56–72,92–95`. `tests/test_entry_intent.py:160–179`. `tests/test_contest_fetch.py:578–635`.
- The real cron install under the throwaway home reproduced the unwanted recipient on the base. The updated comparison allows exactly one new difference between the two intents: the fetch recipient on the four existing schedules. The enter lines retain their previous arguments and log redirection.
- Limit. Only the four fetch lines lose the recipient in research. Existing fetch code keeps exit 2, stderr and the status file. Twelve no-recipient controls covered cookie, transient auth, rate-limited auth, auth network, both identity checks, profile 429/5xx/4xx/network, malformed profile and corrupt switch. Each wrote the expected error category, left the DM timestamp empty, and made zero calls to the DM sender.
- Owner question. Which handle should `deploy/systemd/bts-leaderboard.service:15` use, `stoneericm.bsky.social` (the service today) or `stonehengee.bsky.social` (the fetch cron)? The service was left as written.

## F7 · Thin live-forward capture

- Commit `5277dd70`. `src/bts/experiment/artifacts.py:473–476,521–522,1354–1366`. `scripts/live_forward_capture_once.py:440–466,506–508,567–718`. `src/bts/health/analytics_artifacts_missing.py:22–27`.
- `tests/experiment/test_artifacts.py:413–462` first failed on the base with two scored rows and three requested, then passed with nine scored and ten requested as well as at full count. A separate direct probe ran the base materializer and verifier on exactly nine Opening Night rows. Both variants failed their per-date row-count checks against ten. `tests/scripts/test_live_forward_capture_once.py:157–187` reproduced the no-retry wedge and now passes, including the two green controls for a settled pick and processed PA outcomes.
- Limit. The live manifest keeps requested `top_n` and adds the effective scored count. It must be a positive integer within the requested cap and both variants must match it exactly. Resolved copies inherit it. Legacy artifacts keep the prior requested-count checks. Schema, provenance, snapshot and pre-outcome checks still apply.
- Limit. Only a recorded failed verification for the same locked pick is eligible for this retry. Previously verified captures are excluded from this retry, even after a failed recheck. The existing staged export verifies before replacing the prior capture and backs it up. The retry checks the pick and processed PA before export and rechecks both before publication, rejecting an intervening pick change. The original drift handling and its authorization still apply. This uses the existing pick and processed-PA sources for outcome visibility. No pick logic changed.

## F8 · Shadow monitoring samples

- Commit `86a5a58f`. `src/bts/shadow_eval.py:497–522`. `src/bts/skip_policy_shadow.py:231–241,417–439`. `src/bts/cli.py:2664,3054–3069,3100–3144`.
- The two mixed-year tests failed on the base and pass with the default ET year, including an instant already in 2028 UTC but still in 2027 ET. A separate `--season 2026` read still loads the earlier sample. Existing 2026 tests were pinned explicitly: `test_build_shadow_cycle_status_tracks_recorded_monitoring_state`, `test_build_shadow_cycle_status_counts_void_as_resolved_but_not_evaluable`, `test_shadow_status_cli_writes_status_artifact`, and `test_end_to_end_decision_to_status`. Their original result expectations were retained.
- Limit. `shadow-status` and the status output of `skip-policy-shadow-update` default to one season and accept `--season`. Where a decision date is passed, `check-results --date` scopes the shadow status it writes and `skip-policy-shadow-update --date` scopes its status unless overridden. Recording, pruning, reconciliation, `shadow-report` and shadow backfill keep their existing scope. Checkpoint and shadow-quality calculations are unchanged.
- The existing operational 2026 pick corpus was unavailable locally (`data/picks` is absent), so the CLI comparison used 29 dated 2026 fixture pairs and one 2027 pair. Under a 2027 wall clock both commands returned identical status JSON with the mixed and 2027-only directories after normalizing timestamps and directory names. Both reported one day and the policy verdict stayed `insufficient_n`. Explicit 2026 output matched the 2026-only directory at 29 days. For 2026-only input, the 2027 defaults reported zero. Pooling all 30 fixture records with the same policy calculation yielded `above_breakeven`.

## Verification

- Base focused reproductions via the wrapper, before production edits: 9 expected failures, 7 green controls, 882 deselected. No fast-suite baseline was run.
- Initial sandboxed integration run: 961 passed, 1 skipped, 3 failed. Two failures were corrected (empty-store dtype and a staged-export test double). The third was the sandbox's loopback bind restriction in `test_do_post_saver_transition_socket_wiring`.
- Completed focused wrapper runs with loopback access under the real 2026 wall clock and process-local 2027 and 2028 wall clocks: each 1,016 passed, 1 skipped. All changed test files, `tests/test_lightgbm_optional.py` and `tests/test_web_saver.py` ran together in each process. The shifted runs verified both Python `datetime.now()` and `date.today()` in ET before invoking pytest.
- The original unsandboxed fast suite ran by itself on committed HEAD `86a5a58f`. Pytest exited 1 with 9,218 passed, 37 failed, 16 errors, 23 skipped, 6 deselected and 1 xfailed in 25m47s. The 53 unique failed/error IDs exactly matched `~/bts-runs/gate-failed.txt`, with an empty diff and comparison exit 0.
- Follow-up tests before the edits: 5 expected failures, 13 passing controls, 697 deselected. After the edits, the same focused script ran unsandboxed under each of the 2026, 2027 and 2028 clocks. Each run passed 1,027 tests with 1 skip, including the web saver and optional LightGBM checks in the same process.
- The follow-up unsandboxed fast suite ran by itself on committed HEAD `153ff8c2`. Pytest exited 1 with 9,229 passed, 37 failed, 16 errors, 23 skipped, 6 deselected and 1 xfailed in 26m17s. The 53 unique failed/error IDs exactly matched `~/bts-runs/gate-failed.txt`. The diff was empty and the comparison exited 0.
- Logs, scripts and fixture probes: `.codex-review/audit-f4f8/`.

## Fixes after the batch review

Five local commits follow `153ff8c2`. The limits below supersede the earlier F4, F5, F7 and F8 descriptions where the batch review changed them. F6 needed no fix. The committed changes cover health, evidence capture and reporting. No push, external network, production box or contest account was used.

### F4 · Contest coverage

- `9619563e` · `src/bts/health/contest_state.py:73–98` and `tests/test_contest_dated_cs4a.py:125–170`.
- A missing source date can be informational only for a selected automatic reading from the current ET year, at most 36 hours old, with one or two current-year root-level settlements and no future settlement. The earliest current-year settlement must be today or yesterday ET. The current count and calendar bounds both apply.
- RED on `153ff8c2` and GREEN on the fix for a July check with only the January test pick or only Opening Night, both with and without a fresh July settlement. Controls cover a legacy manual file, earlier-year history, no eligible settlement, future and third settlements, and ignored unresolved and shadow records. Existing moved-opener, 2028 and dated observation controls passed.

### F5 · Park drag

- `a3ce1aac` · `src/bts/health/park_drag_freshness.py:75–103`, `tests/health/test_park_drag_freshness.py:46–82` and `tests/test_park_drag_producer.py:203–220`.
- The configured opener can replace the start of the lag calculation only for a source from the immediately preceding season dated on or within the configured critical threshold before that season's end. Other tables use the original calendar gap. With the default thresholds and the configured March 24 opener, valid carryover is quiet through March 26, warns March 27 and becomes critical March 30.
- RED on `153ff8c2` and GREEN on the fix for the June 15 outage, a two-season-old table and a closing gap above an overridden threshold. A source exactly at the threshold passes. The moved-opener fixture now uses September 22, six days before the 2026 close. The both-empty producer control was green before and after, with a successful zero-row status and no export or manifest. The producer implementation was untouched.

### F7 · Live-forward capture

- `043dbd02` · `scripts/live_forward_capture_once.py:379–475,536–670`, `src/bts/health/analytics_artifacts_missing.py:22–27,302–313`, `tests/scripts/test_live_forward_capture_once.py:157–187,215–305` and `tests/experiment/test_artifacts.py:413–493,608–673`.
- Same-pick verification failures keep the existing capture and raise the capture-health alarm. A later tick can recheck those files. Restored both the wrapper and the health status set byte for byte to `3d49308d`, including the original separately authorized snapshot-drift behavior. The live manifest still records the effective scored count and the verifier uses it for both variants. Legacy files still use the requested count.
- RED on `153ff8c2` and GREEN on the fix for the outcome-arrival probes at exporter entry and swap, and for a formerly verified capture with an unreadable profile and an intervening pick change. The regression asserts one export, frozen evidence and a health alarm for failed verification. The nine-batter Opening Night fixture passes initial verification and a later tick against the frozen capture. The effective-count controls already passed on `153ff8c2` and cover non-integer, zero, over-cap, variant disagreement and propagation to a verified resolved copy.

### F8 · Aged shadow contradictions

- `1446cda8` · `src/bts/skip_policy_shadow.py:417–438`, `src/bts/cli.py:3142–3154` (existing warning output), `tests/test_skip_policy_shadow.py:534–570` and `tests/test_shadow_eval.py:420–463`.
- Season selection applies to the statistical sample. The status JSON and default nightly command report aged contradictions from every season, using the original warning logic.
- RED on `153ff8c2` and GREEN on the fix for a September 27, 2026 decision rewritten to a single and checked in March 2027, both in the default run and with an explicit 2027 season. Date assertions cover `skip-policy-shadow-update --date`, explicit season precedence, `check-results --date`, and the status builder's ET-year default and historical override.

### P-03 · Frozen snapshot

- `8cac32ea` · `scripts/audit/p03_context_shadow_closeout.py:36–60` and `tests/test_p03_context_shadow_closeout.py:7–37`.
- The closeout requests the frozen 2026 season when reading recorded status. The script fixture ran with a March 2027 clock and locally cached feeds. It was RED with an empty recorded sample on `153ff8c2` and GREEN with the expected September coverage, evaluable pair and discordant date.

### Verification after the review

- Replayed the regression tests against archived `153ff8c2` source and scripts. Logs are `fixes-final-red.log`, `fixes-scripts-red.log` and `fixes-capture-races-red.log`. The worktree run before production edits is in `fixes-red.log`.
- The final focused wrapper runs, each in one process with all changed test files, `tests/test_lightgbm_optional.py` and `tests/test_web_saver.py`, finished with 1,054 passed and one skip under each of the 2026, 2027 and 2028 ET clocks. The skipped park producer oracle requires the absent external analysis fixture. Logs are `fixes-focused-{2026,2027,2028}.log`.
- `git diff --check 153ff8c2 HEAD` passed. The F7 wrapper and health status diffs against `3d49308d` were empty. No new baseline fast suite was run.
- The final fast suite ran alone on committed HEAD `8cac32ea`. Pytest exited 1 after 25m25s with 9,251 passed, 37 failed, 16 errors, 23 skipped, six deselected and one xfailed. The 53 unique failed and errored IDs exactly matched `~/bts-runs/gate-failed.txt`. The comparison exited 0 and `fixes-fast-failed.diff` is empty. The suite log is `fixes-fast.log`.
