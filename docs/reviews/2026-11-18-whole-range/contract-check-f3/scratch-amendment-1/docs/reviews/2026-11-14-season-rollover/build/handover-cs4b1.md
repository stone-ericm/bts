# CS4b-1 handover

Base: `8421e3cd6de42e83fe3b671d38309bbf550203a2`, branch `sb-cs4b1`.
Implementation commit: `28ae3aa995f04ba6745389fc22a54abdd8f45e4a` — Add operator season switch and conditional contest fetch persistence. No push, deployment, contest account or box access.
Contract: kickoff version 2, incorporating F1–F6; no design deviation identified.

## Contract implementation and tests

`C` below is `tests/test_season_contract_cs4b1.py`; `E` is `tests/test_season_contract_cs4b1_edges.py`.

| Row | Implementation | Tests |
|---|---|---|
| 1 | `src/bts/cli.py:2114`; `src/bts/season_contract.py:104`: independent offline sections; actual date/year versus inspection season; lstat presence; parser calls with require_dated=False; schema and identifier comparison; saver diagnostics; last ten object ledger lines and bad-line count. Only corrupt switch exits 1. | C `test_status_independent_sections_readonly_and_inspection_season`; E `test_status_schema_dating_and_unavailable_comparison`, `test_status_bad_neighbor_does_not_hide_later_sections`, `test_cold_status_creates_nothing`, `test_status_complete_valid_snapshot_and_identifier_comparison`, `test_status_shows_inactive_manual_and_current_saver_and_last_ten_rows`. |
| 2 | `src/bts/cli.py:2131`: entry clock, year usage check before state I/O, no-op from snapshot, UUID on replacement, valid same-year revoked_at carry, single atomic_write_text, failure and post-enable messages. Switch validation is solely `src/bts/season_contract.py:36`; `switch_state` at :67 validates before season comparison. | C ordered-check, paired-check, nullable-timestamp, identifier, unreadable-switch tests; `test_enable_snapshot_carry_and_noop`, `test_wrong_enable_season_precedes_state_reads`, `test_enable_write_failure_preserves_prior`; E entry-clock and enable-message/before-or-after-install tests. |
| 3 | `src/bts/cli.py:2247` preserves profile .get before rounds, then predictions presence/list/objects and nonempty rounds before date derivation/builder. `src/bts/contest_fetch.py:191` preserves old streak checks then caps best at 57. Builder remains standalone auto_v1. | C response validation and shape precedence tests; E builder/bound control; changed CS1 writer test retains raw bad-file reader refusal. |
| 4 | Fetch entry clock at `src/bts/cli.py:2196`; switch snapshot at :2279; automatic-only observation replacement, status, ledger, conditional saver in order. Unswitched status+ledger only, including cold directories. Corrupt switch uses existing _fail. | C switch-path order/open/saver spies, entry-clock/year-boundary tests, full activation sequence; E prior-identity tests; 1,728-case comparison. |
| 5 | Existing writer/alert semantics retained; dry-run branches at `src/bts/cli.py:2292`; completed persistence prefixes stand. No rollback. | C every switch state × dry-run, response/identity failures with recipient; E every stage × switch state, partial append, saver audit failure, enable-message beside pre/post-observation failure. |
| 6 | `src/bts/cli.py:2314`: original five ledger keys plus season, switched, activation_id, row partition by mapped year/unmapped/unsettled. | C mixed-year row-count partition including duplicate rows, missing/unknown results; consumer checks described below. |
| 7 | `src/bts/contest_state.py:40` adds optional metadata; :258 validates season then identifier after legacy checks only at automatic filename; :191 expands exact-schema profile best trust. | C invalid v2 fields, season error precedence, legacy manual-v2 metadata None and best trust; E builder v1 control; reader matrix. |
| 8 | Decision loader, health, scheduler, orchestrator, web and scripts retain their source. Existing consumers do not read the switch; shared parser/trust effects are intentional. | E switch-nonread spy with corrupt/valid switch, manual present/absent and both settings; reader matrix; existing-test regression run. |

## Recorded verification

Every wrapper ran alone. Baseline fast ran once, before any test file was created, and completed before any other wrapper started. Logs and helpers are only in `.codex-review/cs4b1/`, uncommitted. No shell heredoc or pattern kill was used.

| Run | Result |
|---|---|
| `baseline-fast.log` | 6,316 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed; 1,480.33s. Exact failing/error set equals `~/bts-runs/gate-failed.txt`: 53 IDs, no additions or omissions. |
| `baseline-relevant.log` | 2,168 passed, 8 skipped. Includes requested fetch, ledger, CLI, contest-state, health, backup, season-ledger and discovered related consumers. |
| `red-contract.log`, `red-edges.log` | Initial unchanged-source runs: 98 failed/4 passed; 65 failed/11 passed. They exposed a test-fixture isolation mistake: some tests shared their parent health directory and therefore DM cooldown. Fixed by a per-test picks parent, retaining every assertion. |
| `red-corrected.log` | Against `git archive 8421e3cd src`, after correcting fixture isolation: 149 failed, 29 passed. |
| `red-final.log`, `red-assertions.json` | Final 187 new cases on archived base: 158 failed, 29 passed. JSON records every test ID, outcome and reached failure, including added status/message/partial-append checks. |
| `frozen-final.log` | All seven added/touched test files: 551 passed with wall clock 2027-07-15; 551 passed with wall clock 2028-07-15. Plugin imports pandas before substituting datetime types. Earlier 542-case run also passed in both years. |
| `comparison.log`, `comparison-summary.json` | 1,728 fetch comparisons, 192 reader combinations (each calls decision and health): passed. |
| `consumers.log`, `consumer-results.json` | All direct-reader and typed-projection compatibility checks passed; raw/hash/derived-ID changes expected. |
| `final-relevant.log` | 2,355 passed, 8 skipped; 7.46s. Same existing-test inventory as baseline plus the 187 added cases. |
| `final-fast.log`, on committed `28ae3aa9`, alone | 6,503 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed; 1,477.19s. Exact 53-ID failing/error set matches both baseline and `~/bts-runs/gate-failed.txt`; no added or missing failures. Passing count increased by exactly the 187 added cases. |

### RED versus GREEN evidence

Missing-command/import failures are availability evidence only, not proof that the base exercises a new command's inner assertions. The base has neither command nor module. For those tests, `red-assertions.json` names the unavailable API/command; the branch run exercises the ordered validation, messages, reads/writes, carry-over, no-op and error assertions. An unknown command exiting 2 is explicitly not counted as evidence for the wrong-year usage behavior: that test additionally requires the current-year diagnostic and no state reads.

Concrete base behavioral failures in existing APIs: switch-path spies see unconditional observation/saver work; malformed missing/nonlist/empty-round/best58 fetches succeed or emit different shape diagnostics; unswitched fetch changes prior observation bytes; ledger lacks the added partition fields; v2 parser does not reject its fields or expose metadata; v2 best is untrusted; entry time is captured after authentication/at later persistence stages; partial append leaves v1 rather than stamped v2. The full sequence fails on its first supposedly unswitched observation-byte comparison, before it reaches a missing command.

Already-GREEN controls (29 cases): response validation for null/string/nonobject predictions on both real/dry paths (6), successful empty-list dry-run (1), null-profile versus null-predictions rounds-failure precedence (2), automatic observation-write failure prefix (1), prior identity guard across five switch states and both dry settings (10), decision/health never reading valid/corrupt switches with manual present/absent and both settings (8), saver audit best-effort/helper-invoked-with-no-transition (1). Mixed tests retain their builder-v1 and legacy-reader assertions even where a later new assertion is RED. The pure builder still does not require predictions or a switch.

## Base/branch comparison

`comparison.py` archives base source into scratch, runs isolated base and branch processes with network-denying audit hooks, captures auth/profile/round/DM transports, and fixes clocks. Every case records input bytes, all application-file bytes before/after (base64), exit, output and DMs in `comparison-base.json` and `comparison-branch.json`. Paths in output are normalized to `<case>`. Each process uses distinct case directories; source never overlays tracked files.

Dimensions: 12 switch states (absent, each of eight invalid checks, other season, revoked, automatic) × 8 automatic files (absent, v1, matching v2, other identifier, other season, invalid season, invalid identifier, CS1-refused) × 9 readings (valid, missing predictions, null predictions, nonlist, nonobject row, explicit empty list, empty rounds, best57, best58) × both dry settings = 1,728 fetch cases. Reader comparison uses the same 12 × 8 inputs × both require_contest_state/expected settings = 192 combinations, 384 calls per package.

| Difference group | Fetch cases | Required rows |
|---|---:|---|
| Shape/bound rejection or specific shape diagnostic | 1,152 | 3 (failure status/DM, no ledger or observation) |
| Corrupt switch after valid reading | 384 | 4/5 (including failing dry-run) |
| Valid reading and noncorrupt switch: conditional writes, v2 stamps, ledger metadata or dry/success output | 192 | 4/5/6 |
| Reader combinations with v2 field refusal or best-trust change | 120 | 7 |
| Remaining reader combinations | 72 unchanged | 8 preservation |

Every differing case and differing file/exit/output/DM field is listed in the summary JSON. Reader differences for valid v2 inputs are precisely best_status untrusted→trusted with health unchanged; invalid v2 season/identifier inputs now raise domain errors and health diagnoses them. Other-season and different-identifier valid v2 files remain readable: there is no live switch enforcement in this slice.

## Existing-test migration inventory

No assertions were deleted or loosened. Changes are limited to:

- `tests/test_contest_fetch.py::test_fetch_cli_persists_current_activestreak_despite_lag` and `::test_fetch_cli_persists_snapshot_when_no_settled_predictions`: valid 2027 switch and pinned operation clock; original active/source assertions retained. Added helper `_automatic_fetch` centralizes those fixtures.
- `tests/test_contest_fetch.py::test_fetch_auto_earn_cold_file_at_10_stays_uninitialized` and `::test_fetch_auto_earn_below_10_inits_not_earned_then_promotes_at_10`: fixed 2027 clock/year, switch, real helper spy proving one/two calls. Original saver assertions retained. Ledger-only and cooldown-success tests remain unswitched; early failure and standalone builder controls unchanged.
- `tests/test_cli_integration.py::TestFetchContestStreak::test_happy_path_writes_auto`: pinned switch; intentionally change literal auto_v1 to auto_v2, additionally assert season and identifier. `::test_genuinely_stale_profile_still_writes_snapshot` and `::test_expected_settlement_lag_writes_snapshot_no_alert`: same automatic fixture; original counter/source/no-DM assertions retained. Other class tests unchanged.
- `tests/test_saver_cs3.py::_observe`: on fetch branch only, install switch for captured 2027; no-source fixture keeps predictions=[] but now has nonempty rounds. `::test_fetch_without_settled_source_never_creates_a_flag` retains null-source/flag assertions through that helper. `::test_other_year_observation_does_not_touch_flag`: pin actual operation to 2027, seed/source offsets relative to 2027, add helper-not-called proof. `::test_current_year_observation_initializes_then_earns`: fixed/pinned year, original initialization and promotion assertions retained. Other helper callers retain their assertions, including manual branch behavior.
- `tests/test_contest_dated_cs4a_review.py::test_all_three_row11_warning_commands_and_fetch_auto_earn_isolation`: add valid 2027 automatic switch beside its existing pinned 2027 clock, so fetch truly promotes the flag and the dated observation still says used.
- `tests/test_contest_state_cs1_review.py::test_fetch_writer_out_of_bounds_best_rejected_and_manual_override_recovers`: intentionally expect ContestFetchError from validate_fetch and build_observation for 58; write the bad v1 payload directly to retain reader refusal and manual recovery coverage.

## Reader inventory

Ledger readers found under `src` and `scripts`:

| Reader | Added-key behavior |
|---|---|
| `src/bts/contest_ledger.py::parse_latest_ledger` | Projects predictions from last two lines; same rounds, stability and save evidence. |
| `src/bts/web.py::saver_dashboard_context` | Calls that projection/likely_save; same dashboard nudge. Still may see unswitched successful readings. |
| `src/bts/data/backup.py::verify_restored_ops_tree` | Parses last nonblank line for corruption; extra keys do not change validation. Backup/archive bytes naturally retain them. |
| `scripts/audit/boundary_shadow_census.py::load_ledger` | Projects recorded_at/active_streak/source_date; same sorted observations. |
| `scripts/audit/p05_tail_policy_audit.py::main` | Projects latest predictions by roundId; same map. Checked by executing its actual AST reader statements, not a rewritten imitation. |
| `scripts/audit/season_ledger/sources/contest_ledger.py::parse_contest_ledger` | Same typed line/round/slot projection and quarantine; preserves added keys in line record_raw_json. Content SHA and derived observation IDs change. |
| `scripts/audit/season_ledger/reconcile.py` | Uses that parser and counts structural locators; same locators/dispositions, with content-derived identifiers/raw evidence propagated. |
| `scripts/audit/season_ledger/compile.py` via `contest.py` helpers | Same entered rounds, line-round streak evidence and slot history apart from last_obs_id; source hashes/raw evidence/derived IDs propagate intentionally. |

Automatic-file readers and consumers:

- `src/bts/contest_state.py::_parse_state_file` and `load_contest_streak_state`: same selection/fallback, now ordered v2 metadata validation at automatic filename. Manual-v2 remains legacy with metadata None. Best trust expands to both exact automatic schemas with profile source.
- `load_decision_streak_state`: same decision/saver/fallback logic; no switch read. Shared parser/trust changes may alter v2 outcomes. Both required/optional settings tested.
- `src/bts/health/contest_state.py::check`: same selection, date/coverage/saver diagnostics; no switch read, even with corrupt switch. Both expected settings tested.
- `src/bts/cli.py` run reads (lines 1161,1206,1247), preview (:1321), `src/bts/web.py` decision context (:125) and saver dashboard (:150): unchanged loader use; can still display/use a frozen observation.
- `src/bts/orchestrator.py` decision reads (:283,:362,:386,:429), `src/bts/scheduler.py` delivery/selection reads (:1098,:1723): unchanged required/optional behavior and fallback order. No new refusal, lock, wakeup or delivery guarantee.
- Fetch's prior-observation identity read remains before profile and switch in all switch states; prior nonobject/non-UTF8 behavior is not broadened into a handled failure.
- New `season-contract status` independently parses the actual manual and automatic paths for display without selecting or health checks. Identifier equality with any valid switch snapshot is informational only.

## Accepted limits and promise scope

No live protection yet: decisions still accept on-disk automatic observations stamped or unstamped, matching or mismatching. CS4b-2 adds refusal/revoke. Nothing from CS4a onward may deploy before CS6; this commit is not deployable alone.

Enable is the operator's statement after inspecting the endpoints; the code cannot establish what MLB counters mean across a boundary. While unswitched, the automatic observation is frozen. Existing source-date/coverage health checks can see it age when no dated observation applies; fetch success does not mean accepted live state and does not promise system-wide silence.

No enable/fetch lock: each operation reports its snapshot; a racing fetch stamps the identifier it read, and last replacement wins. A fetch can replace the observation then fail later; prior completed writes are not rolled back, and ledger append may be partial. A rejected reading has no ledger row. Saver audit failure stays best-effort.

All “never”/“always” claims concern JSON/file contents and command-line text, captured normal transport payloads and the stated filesystem-failure exits. They do not cover hostile in-process Python objects or arbitrary failures beyond the specified persistence semantics.

Unmatched rows or implementation concerns: none identified. Availability-only base failures are disclosed above rather than presented as per-assertion behavioral RED evidence.
