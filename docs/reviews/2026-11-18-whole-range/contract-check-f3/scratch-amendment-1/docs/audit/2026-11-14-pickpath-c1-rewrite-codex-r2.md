# Verdict

**SIGN** — tier A, round 2 of 2, under R13's replacement contract. Reviewed the uncommitted diff and new `tests/test_pickpath_review_pp2.py` over HEAD `148ec3407f5b267d1f2eb590436d97415abf8e36`. Read R13 from `main` at `7bc1dd6`; its scheduler, picks and late-delivery code are identical to this worktree's HEAD.

The reduction and its extra writes/mutation are gone. Partial-void doubles follow the original delivery path. No new blocking production failure established; the remaining notes do not prevent acceptance under R13.

# Round-1 B1 against the reference

I adapted my own `/tmp/test_ppc1_review_r1.py` scenarios into `/tmp/test_ppc1_review_r2.py`, retaining real `run_day`, `run_single_check`, cache loading, stale archival, fallback, delivery and commit control flow. Network/cascade/clock/transport leaves are fixtures. Both leg orientations and DM/public/private were covered where applicable.

**All 44 executed observation tuples match `4199360` exactly:** final file presence, slot IDs, delivery/attempt flags, decision action/candidates/status/scoreability and successful transport calls/text. Twenty-four cases commit and twenty lose the day in both versions.

| Scenario | Cases | Reference → change |
| --- | ---: | --- |
| Equal 19:10 starts: pick-save death, state-save death, contest-state error or caught send failure | 20 | Committed double → same |
| Earlier void 13:40 start: death before first normal pick save; restart 13:36 | 6 | No pick/decision/send → same |
| Earlier void start: caught contest-state error or caught transport failure | 8 | No pick/decision/successful send → same |
| Earlier void start: normal state-save death, DM/public | 4 | Delivered double retained, no duplicate send → same |
| Earlier void start: normal state-save death, private | 2 | No pick/decision → same |
| Corrected status and 19:40 schedule start after pre-send error | 4 | Old cached time causes cutoff refusal; no delivery → same |

The deleted reduction's pre-send state-save boundary no longer exists. A state-save injection now reaches the normal lock save: after the successful pick save/send in DM/public, before the private decision record in private mode. The author's handover correctly distinguishes these boundaries.

Against HEAD, some earlier-cutoff/error cases and all four corrected-status cases previously recovered a separately saved single and now lose the day. Those differences result from removing reduction and retaining the original earliest cutoff, the costs R13 explicitly accepts. Conversely, the six equal-start pick-save deaths that lost the file on HEAD now deliver as the reference does. No executed case is worse than both HEAD and the reference; I found no loss of HEAD-correct behavior outside the removed reduction.

**Reference method checked independently:** I copied the application package into scratch, replaced scheduler/late-delivery source with `git show` output and imported it in a fresh process. The reference initially has no `_void_committed_games` or `_delivery_game_statuses`; it does not inherit globals from the new scheduler. Other application dependencies remain current. The author's execute-into-existing-module method leaves newer helper names behind, but that did not contaminate these outcomes: the fresh import reproduces them. This is a scheduler comparison, not a reconstruction of every historical dependency.

The adopted B2, restart matrix and corrected-status tests also pass against the fresh reference: **43 passed, 8 skipped**. Inert test seams were added only to permit current conftest/guard-control patches; the reference source never calls them.

# Blocking findings

None established for the replacement contract. The earlier-cutoff and corrected-time losses remain, but are measured reference behavior rather than newly claimed survivor recovery.

# Notes

- **Refusal condition is sound for recognized game IDs.** `_void_committed_games` constructs a subset of `_committed_pick_game_pks`; that helper deduplicates same-game doubles. Thus equal lengths at `scheduler.py:1070` means every distinct committed game was explicitly identified as void. My 105-case matrix exercised all modes, singles, different-game/same-game doubles, mixed Postponed/Cancelled/Canceled, case/whitespace normalization, unrelated games, missing answers and unreadable schedules. Recognized all-void picks refuse without a transport send; partial-void doubles retain both slots, with no reduction record, archive or alert dispatch.
- **Unknown is not proof of playability.** Missing games and string-keyed answers are treated as unverified, hence delivered under the existing fail-open policy. A response whose `gamePk` values are strings (`"1"`, `"2"`) while the pick IDs are integers delivers even if both returned states say postponed/cancelled. The parser preserves those key types; lookup misses them. HEAD does the same. This is a lookup/schema limit, not a new length-comparison defect. I did not establish an actual producer of mismatched IDs in normal operation.
- **Grading/readers verified on produced files.** Forty-eight independent cases exercised both real scorers, either void leg, DM/private commits, Preview or Final postponed feeds with no roster, active hits, ordinary misses and saver-absorbed misses. Hit+void adds exactly one; miss+void follows the single-pick saver rule. Scorer retries and complete-history reconcile replay preserve the result. The daemon was invoked with the primary game, matching `run_day`. The real live-forward snapshot loader, called at capture time, retains both slots; cached shadow evaluation retains two slots and grades the void correctly. `decision.json` records the original double and both candidates. No scoring exception or double-count was observed.
- **Health distinction:** an ample-margin DM partial-void delivery yields no `late_delivery` or `postponed_pick` alert. A successful private partial-void commit yields no late alert but still produces `postponed_pick` CRITICAL saying the undelivered candidate is stale and should be regenerated (`health/postponed_pick.py:33–45`). That check ignores the scoreable private decision and sees no human-delivery flag. Executed against the reference: identical alert; HEAD's reduction avoided it. This is an additional operational consequence of retaining the void leg, not a new automatic pick/scoring failure.
- **Legacy reduction records:** deleting their health branch makes an old `leg_dropped_game_not_playable` record fall into the late-refusal default (`health/late_delivery.py:57–61`), producing CRITICAL `None min past the None cutoff`. Reproduced with a planted old-format record. No new code emits that reason, and the check reads only the requested day's state. Historical/restart compatibility of such diagnostic records is now limited; this does not affect the new 2027 pick contract.

# False greens

- **Adopted scorer tests do not establish schedule-based void resolution.** `test_delivered_hit_plus_void_double_adds_exactly_one` uses `H._route_http`, whose schedule reports every supplied game as Final; its postponed feed is also Final and contains explicit zero ABs. I disabled only `picks._slot_is_voided` in memory: **all four adopted cases still passed**, because no-official-AB grading independently supplies the Pass. An independent Preview/Postponed/no-roster case then fails unresolved under that same mutation, and passes on unmodified production. Add that fixture to pin the dependency the retained double needs. This is a test-quality NOTE; the production path was independently covered and works.
- The adopted daemon scorer calls `run_result_polling(kept.game_pk, ...)` (`tests/test_pickpath_review_pp2.py:262`). When the primary is void, that is the DD game, unlike real `run_day`, which passes the primary. My independent cases use the production argument and pass, including Preview void feeds. Prefer that argument in the durable regression.
- The adopted day-level tests still execute the original failure paths: real restart/archival/fallback, failed prediction cascades, pre-send contest-state errors and corrected schedules. Their hardcoded reference-loss expectations are now explicitly true of the fresh reference. A green result means parity with the accepted original cutoff behavior, not preservation of every enterable surviving leg.
- The unchanged-double matrix's identity/value/save-count, original decision-source, no-refusal/no-alert and transport assertions pin the new contract. I found no assertion weakened to hide an unrelated delivery failure. The old reduction-specific assertions were removed because R13 supersedes them.

# What I ran

- **691 passed, 8 skipped:** the exact focused set in kickoff 2, including all three changed test files, scheduler/decision/streak/incident/thin-slate tests and `tests/health`. Log: `/tmp/ppc1-review-r2-focused.log`. No whole suite.
- **199 passed, 8 skipped:** my independent probes, `/tmp/test_ppc1_review_r2.py`; log `/tmp/ppc1-r2-independent-final.log`. Includes the 44 day-level cases, 105 refusal cases, 48 grading/reader cases and two diagnostic probes. Private contest-state/send boundaries account for the eight skips.
- Own day-level cases in fresh reference and HEAD processes: **44 passed, 8 skipped each**; HEAD is observed without requiring the new double contract. Exact reference/current tuple comparison passed for all 44. Logs/observations: `/tmp/ppc1-r2-own-{reference,head}.log`, corresponding `.jsonl`, and `/tmp/ppc1-r2-own-current-final.jsonl`. Fresh adopted-reference run: `/tmp/ppc1-r2-adopted-reference.log`.
- Reproduce current probes: `PPR2_VERSION=current TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/ppc1_r2_run_version.py current /tmp/test_ppc1_review_r2.py -q -p no:cacheprovider --tb=short`. For the `own_b1` subset, change both `PPR2_VERSION` and the runner version argument to `reference` or `head`. Scratch packages are `/tmp/ppc1-r2-review-versions/`.
- In-memory void-resolution mutation: `/tmp/ppc1_r2_disable_void_resolution.py`; adopted false green **4 passed**, independent Preview control **1 failed as expected**. Logs `/tmp/ppc1-r2-scorer-false-green.log` and `/tmp/ppc1-r2-scorer-control.log`. No source mutation.
- Read both contracts/handovers, R13 on main, the full diff against HEAD and `4199360`, cache/caller/commit/scorer/health paths, test fixtures and artifact readers. Tests used synthetic files/feeds/clocks and mocked transports. No real account, network send, historical outcome read, prohibited directory read, model training, stash, commit, push or deployment. Source/tests unchanged; this report is the only intended worktree write. `git diff --check` passed.
