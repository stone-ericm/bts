# Adversarial Review Report, Round 2 — Whole-Range Change Set C2 (`e897c766` vs `386aa846`, W22)

## Verdict
**SIGN**

## Blocking findings
None.

## Notes
- **N1 (Extra keys on an otherwise complete `sent` receipt, `src/bts/health/alert.py:324-327`):** `_dispatch_with_acknowledgement` validates the five required acknowledgement fields (`schema_version`, `status`, `sent_day`, `sent_sources`, `message_id`) via `.get(...)` without restricting `prior.keys()`. Seeding all five valid `sent` fields plus an extra key (including `"error": "stale"`) still deduplicates (`item1_receipt_hunt.json`). Not a blocker: no code path in `src/` emits `status == "sent"` with `message_id` unless `send_dm` returned a nonblank `str`, and any `failed` or `skipped_no_recipient` receipt tampered with `message_id`/`sent_day`/`sent_sources` still sends (`status != "sent"`).
- **N2 (Vacuous base pass on `stderr-classification`, `tests/test_wr_c2_recovery.py:438-446, 471`):** `test_unknown_alert_failure_is_contained_after_commit[stderr-classification]` asserts `0` stderr lines without checking that the injected `fail` callback ran, accounting for the 11th pass on `386aa846` alongside the 10 green lifecycle controls (`90 failed, 11 passed`). Covered on HEAD by `codex/probe_failures.py`.
- **N3 (Supplemental tests in `.codex-review/wr-c2-r2/adopt/test_wr_c2_r2_carbon.py`):** 18 self-contained tests covering same-day cutoff shared status copied into the own receipt path, copied prior-slate C2 receipts, `failed` status carrying `sent` keys, superset/duplicate/unsorted `sent_sources`, NBSP `message_id`, boolean `schema_version`, double-down unconfirmed locks, `config["bluesky"] = None`, and the ledgered `dm.send_dm -> None` vs watchdog divergence.

## 1. B1 and every other way a receipt could wrongly suppress the alert
Ran `.codex-review/wr-c2-r2/probe_item1_receipt_suppression_hunt.py` (`item1_receipt_hunt.json`) across `dfe774df` and `e897c766`:
- **B1 (`schema_version`):** Across 16 non-conforming `schema_version` values (missing, `null`, `""`, `"v0"`, `"v2"`, trailing space, uppercase, `["damaged"]`, `[valid_str]`, `{…}`, `0`, `1`, `true`, `false`, `NaN`, `Infinity`) x 2 routes (`direct` and `classification` via real `run_day`), all 32 runs falsely suppressed (`0` DMs) on `dfe774df`, and **all 32 sent 1 DM, atomically replaced the file with a valid `sent` receipt, and deduplicated the next `run_day` on `e897c766`**.
- **Wrong field types/values, other-slate copies, foreign code paths, and source order:** Tested 63 further receipt payloads x 2 routes (`126` runs): wrong/missing/blank/non-`str` values on `status` (12), `sent_day` (12, plus a real `2027-03-24` C2 receipt copied into `2027-03-25.json`), `sent_sources` (12, including supersets `["CRITICAL:late_delivery","CRITICAL:pick_delivery"]` and duplicates), `message_id` (15, including `"\u00a0"`/`"\u2000\u3000"`, booleans, numbers, `NaN`), and real outputs from other writers copied into `delivery_outcome_unknown/<slate>.json` (same-day cutoff refusal `health_dm_delivery_status.json` sharing `schema_version`, `status="sent"`, `sent_day`, and `["CRITICAL:late_delivery"]` but lacking `message_id`; EOD `run_all_checks`; 4c stale warning; own `failed` and `skipped_no_recipient`; `failed` tampered with `sent` fields; `decision.json`; delivered pick JSON with `notification_id`; `warn_attention_state.json`; entry/reconcile receipt shapes). **Zero false suppressions (`158/158` runs sent 1 DM and replaced the file).** On multi-alert opt-in calls, unsorted `sent_sources` on disk did not suppress (`attempted=True`, rewrote sorted `sent_sources`), while swapped `Alert` input order deduplicated against the canonical sorted receipt (`attempted=False`).
- **Reverse check:** A genuinely acknowledged receipt (`"  exact-ack-id  "`) deduplicated across 4 subsequent encounters (`classification` -> `direct` -> post-midnight `run_day(DAY)` -> no-recipient `run_day`), sending `0` extra DMs and preserving exact receipt bytes.

## 2. Round 1 probe re-runs and legacy-caller parity
Ran `.codex-review/wr-c2-r2/probe_item2b_rerun_round1_probes.py` (`item2_round1_rerun.json`), staging Round 1's 6 Codex probes and 5 Carbon final probes at `/tmp` temp roots against `e897c766` and archived `386aa846`:
- **Codex (`probe_restart`, `probe_isolation`, `probe_legacy_dispatch`, `probe_receipts`, `probe_failures`, `probe_operator`):** All 6 exited `0`. `probe_receipts` reported `wrong_schema: sends=1 (contract 1), diagnostics=0.` with zero `CONTRACT MISMATCH` lines; `probe_restart` matched base pick/decision hashes across real restarts; `probe_legacy_dispatch` confirmed omitted and explicit `require_message_id=False` are **byte-identical to `386aa846`** across all 14 live legacy callers/cases (bodies, booleans, status bytes, stderr, log records).
- **Carbon (`probe_1c`, `probe_2c`, `probe_3b`, `probe_4b`, `probe_5g`):** All 5 exited `0` (real restarts, midnight/New Year rollovers, cutoff/EOD isolation, 13 reply shapes x 2 routes, 14 legacy callers vs `386aa846`, 32 damaged/special/TOCTOU receipt cases x 2 routes, 12 `KeyboardInterrupt`/`SystemExit` windows, U7 scenarios).
- Note on `probe_item2_rerun_round1_probes.py` (first staging): placing `TemporaryDirectory` inside the 84-char nested `.codex-review/wr-c2-r2/r1_staged/codex` path made `classification-mkdir`'s `detail` string 203 chars so `[:200]` clipped the last 3 chars of `.json`; staging tempdirs in `/tmp` (`probe_item2b`) kept path lengths realistic (`<100` chars) with every assertion untouched.

## 3. Operator text (row U7) and `CLAUDE.md`
Ran `.codex-review/wr-c2-r2/probe_item3_5_operator_and_edges.py`:
- Confirmed `_alert_delivery_outcome_unknown` in `src/bts/scheduler.py:1160-1183` is **byte-identical** between `dfe774df` and `e897c766` and matches `contract-check.md` Replacement U7 (with the ASCII `scheduler's` apostrophe) across both routes and retries.
- Verified `CLAUDE.md:92` matches live behaviour, including the new `**Known and not fixed:**` sentence: driving `_deliver_and_lock_pick` with `dm.send_dm -> None` wrote scoreable `decision.json` (`delivery_status="delivered"`) while `delivery_attempted=True` and `pick_was_delivered=False`; restarting `run_day` paged C2 (`"Do not enter or change a pick based on this alert."`) while `bts check-pick-entered` hit its untouched generic branch (`"Fix it now!"`).

## 4. Round 2 test changes, scratch mutations, and one-process suite
Inspected `git diff dfe774df e897c766 -- tests/` and ran `.codex-review/wr-c2-r2/run_item4_mutants_and_base.py` (`item4_mutants_and_base.json`) plus combined suites:
- All callback assertions in `tests/test_wr_c2_recovery.py` (`test_direct_unconfirmed_uses_own_receipt_after_commit`, `test_classified_restart_preserves_existing_commit_and_terminal_result`, `test_unknown_alert_operator_copy`) were moved outside `send_dm` and paired with `receipt["status"] == "sent"` assertions; `test_wr_c2_carbon_v2.py` matches Carbon's Round 1 `adopt/` file byte for byte; `test_unknown_receipt_ignores_unusable_path` now runs all 17 injuries across both routes (`34` items).
- **Base `386aa846` RED check:** `test_wr_c2_recovery.py` (`90 failed, 11 passed`), `test_wr_c2_carbon_v2.py` (`21 failed, 0 passed`), `test_alert_status_callers_cs6b2.py` (`4 failed, 18 passed`), `adopt/test_wr_c2_r2_carbon.py` (`18 failed, 0 passed`).
- **12 scratch mutants (`prepare_mutants_r2.py`) vs committed C2 tests (`recovery` + `carbon_v2`) — `0` survivors (`12/12` killed):**
  - `m01_drop_schema_check`: **killed** (`6` fails in `recovery`; `+2` in `r2_adopt`).
  - `m02_early_classification_before_decision`: **killed by both files** (`5` fails in `recovery`, `4` in `carbon_v2` — fixing the Round 1 callback-swallowing survivor in `recovery` too).
  - `m03_follow_receipt_dir_symlink`: **killed** (`4` fails in `carbon_v2`; `0` in `recovery` alone).
  - `m04_drop_post_send_entry_recheck`: **killed** (`3` fails in `carbon_v2`; `0` in `recovery` alone).
  - `m05_drop_resurface_guard`: **killed** (`4` fails in `carbon_v2`; `0` in `recovery` alone).
  - `m06_accept_none_id`: **killed** (`13` fails in `recovery`, `5` in `carbon_v2`; `+2` in `r2_adopt`).
  - `m07_wall_date_key` (`2+2`), `m08_wall_date_filename` (`4+2`), `m09_shared_status_file` (`70+20`), `m10_drop_classification_call` (`40+12`), `m11_follow_file_symlinks` (`2+3`), `m12_allow_hardlinks` (`2+1`): all **killed**.
- **One-process combined run:** In the 9-file run (`pytest-combined-9.log`, `308` items), `307 passed`; only `test_web_saver.py::test_do_post_saver_transition_socket_wiring` failed on `sock.connect(('127.0.0.1', …))` (`PermissionError`, reproduced identically alone in `pytest-web-saver-control.log` due to Seatbelt loopback restriction). With `adopt/test_wr_c2_r2_carbon.py` added (`10` files, `socket_wiring` deselected, `pytest-combined-10.log`), **`325 passed` in one process (`10.33s`)**.

## 5. Anything else in `git diff 386aa846 HEAD`
Verified in `probe_item3_5_operator_and_edges.py` and `git diff --check 386aa846 HEAD`:
- Cross-route retries/dedup (`direct <-> classification`), malformed `config` (`{}`, `{"bluesky": None}`, `{"bluesky": "str"}`, empty recipient), Unicode line separators (`\u2028`, `\u2029`, `\x85`, `\v`, `\f`) and lone UTF-16 surrogates (`\ud800`) in errors and `message_id`, double-down unconfirmed picks (neither primary nor double-down batter named), and INFO-only/empty alerts with a FIFO at `status_path` all behaved safely.
- Only the 6 expected files changed; C1 overlap files (`runner.py`, `health/contest_state.py`, `season_contract.py`, `contest_state.py`) and `cli.py` are untouched; no hardcoded dates/years in `src/`; `git status --short` has zero tracked modifications.

## What I ran
- Pre-21:45 ET direct probes (`.venv/bin/python -B`): `probe_item1_receipt_suppression_hunt.py`, `probe_item3_5_operator_and_edges.py`, `prepare_mutants_r2.py`, `probe_item2_rerun_round1_probes.py`, and `probe_item2b_rerun_round1_probes.py`.
- Post-21:45 ET serial pytest runs (Seatbelt refused `/Users/eric/bts-runs/yield_run.sh` and `pgrep`, so all pytest runs executed sequentially one at a time after 21:45 ET via `/usr/bin/nice -n 19 .venv/bin/python -m pytest -p no:cacheprovider …`):
  - `pytest-combined-9.log` (`307 passed, 1 failed` sandbox loopback socket), `pytest-web-saver-control.log` (isolated socket control), `pytest-combined-10.log` (`325 passed, 1 deselected`).
  - `run_item4_mutants_and_base.py` (`mutant_logs/*.log`, `item4_mutants_and_base.json`): `base_386aa846` RED check + 12 sequential scratch mutants (`0` survivors).
