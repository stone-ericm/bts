# C2b confirmation review of `6663045e`

## Verdict

**SIGN**

## Blocking findings

None found. No other repository writer produced an acknowledgement that suppressed a genuinely unacknowledged slate.

## Notes

- The path-only replay of the original identity probe reaches its old five-field writer assertion immediately on HEAD. The correction probe already expects six fields and passed unchanged. The separate receipt matrix retested the identity cases with the new format. [Original replay](identity-archive-head.log) · [Correction](correction-archive-head.log) · [Matrix](receipt-matrix.log)
- The adopted multi-source test adds the correct digest to its deliberately unsorted fixture at `tests/test_wr_c2_r2_carbon.py:123`. Amendment 2 literally names genuine fixtures for that addition. This extra fixture edit keeps the unsorted list as the sole reason for rejection, as required by amendment 2a. The older Carbon test name still refers to deduping swapped inputs.
- The symlinked-directory test at `tests/test_wr_c2_carbon_v2.py:145` seeds an old five-field record. Its populated-target variant would miss a regression that followed the directory link for reads alone. The independent probe put a genuine same-slate acknowledgement behind a directory symlink, file symlink and hardlink. Both routes paged and preserved the alias and target bytes in all six cases. [Alias probe](canonical-alias.log)

## 1. Suppressing receipt bytes and writer audit

For an unaliased regular file under the dedicated directory, `src/bts/health/alert.py:247–283,317–344` accepts exactly one UTF-8 JSON object with these six decoded members. Their types and values are the schema string shown below, string `"sent"`, the exact slate date string, the one-element string array `["CRITICAL:late_delivery"]`, a string with truthy `.strip()`, and the exact lowercase SHA-256 hex string of the full UTF-8 U7 DM, including its CRITICAL heading and source prefix. For `2027-03-25`, the real writer emitted these bytes, including the final LF. [Captured bytes](receipt-matrix.log)

```json
{
  "body_sha256": "9f08c9013aec667f9a4208b4e930043efefe9a21d2c9c09cb35afaf1edc2cef2",
  "message_id": "seed-id",
  "schema_version": "bts_health_dm_delivery_status_v1",
  "sent_day": "2027-03-25",
  "sent_sources": [
    "CRITICAL:late_delivery"
  ],
  "status": "sent"
}
```

| Byte pattern | Writer that can produce it at the slate path | Result on both routes |
| --- | --- | --- |
| Exact six values above with any nonblank string ID | The C2 opt-in writer after acknowledgement. The ops restore drill can only reproduce already backed-up bytes under its empty restore target. | Suppresses. A genuine same-slate copy also suppresses by contract. |
| Same decoded members in any key order, with JSON whitespace or equivalent JSON escapes. Literal UTF-8 and escaped IDs are equivalent when they decode to the same string. Even escaped NUL or lone surrogates in a nonblank ID pass. | C2 emits sorted, indented JSON and escapes non-ASCII. Alternate serializations were seeded by the probe. No other production writer emits the six matching values. | Suppresses. Whitespace allowed outside tokens is space, tab, CR and LF. |
| Leading BOM, UTF-16, invalid UTF-8, non-JSON whitespace, malformed/deep JSON, duplicate decoded names at any depth, any extra or missing member, wrong type or value | Probe fixtures. No Unicode normalization after JSON decoding. `NaN` and infinite exponents parse as numbers and cannot match. The 4,300-digit integer failed the type check. The 5,000-digit integer failed Python's default conversion limit. | Rejected. NFKC-equivalent status, date and source also failed. Unicode whitespace-only IDs failed `.strip()`. |
| Five-member sent record, failed/skipped record, another slate's genuine record, different opt-in body with the same source and slate | Prior C2 writers or current C2 on a different slate or outcome. No second production opt-in caller. The probe ran the real opt-in dispatcher on cutoff text. | One page, valid replacement, then dedup. |
| Legacy health DM receipts, including cutoff and void, contest failure, missed pick, morning refusal and stale reading, and EOD | Default dispatcher writes `health_dm_delivery_status.json`, `contest_stale_reading_dm_delivery_status.json`, and `contest_morning_check_dm_delivery_status.json`. EOD can target the C2 path via a status-path override. Its schema has extra fields and no `message_id`, even with the exact body. | Copies rejected, including cutoff with an ID inserted. The real EOD runner's exact-body override also could not suppress. |
| `pick_entry_check.json`, `pick_entry_receipts/`, `reconcile_receipts/`, `contest_streak_fetch_status.json` | Entry watchdog and receipt publisher, reconcile receipt publisher, contest-fetch cooldown. All have separate paths and different top-level fields. | Actual outputs copied to the slate path could not suppress. |
| `warn_attention_state.json`, `memory_growth_history.jsonl`, `cascade_stage_durations.jsonl`, `slate_auc_status.json`, `scheduler_churn.json`, `backup_status.json` and its lock | WARN attention, RSS history, scheduler watchdog, slate AUC, heartbeat cron, and restic backup bookkeeping. The WARN/RSS/heartbeat paths can be overridden, with incompatible schemas. | Actual outputs copied to the slate path could not suppress. |

Searched `src/`, `scripts/`, `deploy/` and workflow files for receipt fields, path construction and writer calls. Checked the Hetzner cron table and every systemd unit. Only `bts-scheduler.service` reaches the C2 writer. The dashboard reads the shared status. Cron's ops backup copies existing bytes out and its status writer uses `ops` or `archive` keys. The other jobs use their own state paths. The 28-output copy probe found zero suppressions over 56 route cases. [Writer output](state-writers.log) · [Legacy replay](replay-codex-probe_legacy_dispatch-head.log)

## 2. B1, B2 and reverse deduplication

The path-only identity replay on `e897c766` reproduced six false suppressions per route across 126 cases. On HEAD the unchanged correction probe completed ten cases. Extra members, duplicates, a foreign acknowledgement and the five-member record each sent once across two encounters. The genuine receipt sent zero more. The expanded replay completed 180 cases, including both routes for the original counterexamples and exact-body and encoding controls. Four rollover runs used the same slate from 23:58 through 00:08 ET, including Dec 31 to Jan 1 and a switch to the other route. Each produced one C2 DM over four encounters with stable receipt bytes and no recommendation resend. [Prior head](identity-archive-e897c766.log) · [Correction](correction-archive-head.log) · [Matrix](receipt-matrix.log)

## 3. Base parity, damaged paths and interruption

- `replay_round1.py` and the direct restart comparison matched `386aa846` on all 14 legacy cases for transport bodies, status-file bytes, return values, stderr and log records with relocated traceback coordinates normalized. Both omitted and explicit `require_message_id=False` matched. The three real crash phases matched base pick/decision hashes, lock state and polling, with 0/1/0 total recommendation sends and one C2 DM per slate. Six more classification cases included scoreable records, terminal picks and a decision writer returning no record. Three direct-branch cases matched base pick and decision bytes, durable lock state and journal. [Legacy](replay-codex-probe_legacy_dispatch-head.log) · [Restart](restart-head-direct.log) · [Direct branch](direct-head.log)
- The replay kept cutoff/EOD isolation, explicit versus undated slate selection and both ET rollovers. Carbon acknowledgement and damage replays passed after refreshing only their prior five-field expectations. They covered 13 reply shapes per route, 32 damaged/special/racing paths per route, and 12 interrupt windows. `KeyboardInterrupt` and `SystemExit` propagated after the durable lock. Before acknowledgement publication the next encounter repaged, and after publication it deduped. Diagnostic tests used a short fixture root and separately reproduced the ledgered long-path truncation. [Isolation](replay-carbon-probe_2c_isolation_and_dedup-head.log) · [Acknowledgement](ack-head.log) · [Storage](storage-head.log) · [Diagnostics](diagnostics-head.log)

## 4. Tests and scratch mutations

The earlier C2 files change only exact sent-receipt assertions and the genuine read-only-directory fixture to include the digest. The adopted Carbon file updates one exact receipt assertion and exercises the amendment 2a send/rehash/repeat sequence while preserving the unsorted-list check. The new file covers both routes with valid digests on damaged fixtures, and directly checks nested duplicates and the digest of the captured transport bytes.

All **469 tests passed** in one pytest process in 13.07 seconds, including the four C2 modules and seven requested compatibility modules. The web-saver fixture used loopback only. All seven scratch mutants were killed. [Combined run](pytest-combined.log)

| Mutation | Assertion that killed it | Failures |
| --- | --- | ---: |
| Drop the key-set check | `test_rejects_invalid_receipt`, `extra-error` on both routes | 2 [Log](pytest-drop-keys.log) |
| Accept a key superset | `test_rejects_invalid_receipt`, `extra-unknown` on both routes | 2 [Log](pytest-accept-superset.log) |
| Allow duplicate members | `test_rejects_invalid_receipt`, `duplicate-status` on both routes, and `test_reader_rejects_duplicate_members_inside_nested_objects` | 3 [Log](pytest-allow-duplicates.log) |
| Drop the digest comparison | `test_foreign_dispatcher_receipts_do_not_suppress_c2`, `foreign-body` on both routes | 2 [Log](pytest-drop-hash-compare.log) |
| Hash a constant | `test_receipt_hash_matches_exact_transport_body` on both routes | 2 [Log](pytest-hash-constant.log) |
| Hash only the alert message | `test_receipt_hash_matches_exact_transport_body` on both routes | 2 [Log](pytest-hash-message.log) |
| Compare only `sent_day` | `test_foreign_dispatcher_receipts_do_not_suppress_c2`, `foreign-body` on both routes | 2 [Log](pytest-compare-day-only.log) |

## 5. Remaining diff and ledger

`git diff --check 386aa846 HEAD` was clean. The only pre-C2 test edit concerns the two authorized unconfirmed cases. U7 matched on both routes with the kickoff's ASCII apostrophe ruling, including the status command's read-only behavior and the separately ledgered delivered-label watchdog message. `CLAUDE.md:92–93` matches the receipt rule, old records, same-slate copying, retry limits and backstops. No tracked file was changed for this review. [Operator replay](replay-carbon-probe_5g_operator_text_and_watchdog-head.log) · [Diagnostics](diagnostics-head.log)

The author's completed full gate reported 10,396 passes. Its 53 failure and error IDs match `~/bts-runs/gate-failed.txt` exactly. The reviewer ran only the focused suite and scratch mutation selections.

## What I ran

- Ran the archived identity and correction probes, the restart/base and legacy parity replays, the applicable Codex and Carbon controls, and the new receipt matrix and state-writer probes with `.venv/bin/python`, bounded offline fixtures, pinned ET clocks and captured transports. Prepared isolated `git archive` trees for `386aa846` and `e897c766` and copied HEAD source for each mutant.
- Ran `.venv/bin/python -m pytest -p no:cacheprovider -q --tb=short` through `~/bts-runs/yield_run.sh`, once for the combined suite and once per mutation. Checked `pgrep -f yield_run.sh` before each start. The first wrapper started at 00:50:46 ET, after the author's wrapper exited. The test driver allowed the web fixture's local socket and rejected external Python network connections. [Driver](run_review_pytest.py) · [Wrapper timing and exits](pytest-review-wrapper.log.watch)
