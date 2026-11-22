# Review Report, Round 2 — Change Set W3 (`c0917aa7` vs `59b72c04` and `ceab9854`)

## Verdict
**SIGN** — All four round-1 blocking findings across both reviewers (R1–R4) plus R5 and the reuse journal placement are fixed on their original inputs without regression on the round-1 contract matrix or non-reuse paths.

## Blocking findings
None.

## Notes
- **N1 (`src/bts/scheduler.py:3531-3553` — R2×R4 defer-to-deliver flip discards `defer_void_games`):** When `polled_for_defer` is `True` at `18:42:59 ET`, `_replan_after_reuse_poll` records `defer_void_games = {101: "Postponed"}` while advancing the clock to `18:43:01 ET`, and the second planner call flips `plan.action` from `"defer"` to `"deliver"`. Because line `3542` gates `defer_void_games` on `plan.action == "defer"` (matching Amendment 1b R4 verbatim), `_deliver_and_lock_pick` runs a second unbudgeted `_delivery_game_statuses` request: if consistent, it refuses `[postponed_pick]` after printing `"FALLBACK — delivering..."`; if that second request fails transiently (`OSError`), `_void_committed_games` fails open under R10 and delivers a game just observed as `Postponed` (`r2_probe_2_results.json` `r2_flip_void_*`).
- **N2 (`src/bts/scheduler.py:3373,3510` — shadow log name on mismatched scoreable commit):** On a scoreable `decision.json` whose `primary` batter differs from `daily.pick`, `finish_fallback_lock(daily)` passes `daily.pick.batter_name` to `_trigger_shadow_prediction_on_lock` (affecting only the stderr `AGREES`/`DISAGREES` tag, matching existing line `3312`; `{date}.shadow.json` is unaffected).
- **N3 (`tests/test_wr_w3.py` — heuristic fixture & 10 surviving scratch mutants):** `offline` still omits `module.ROOT` (running under the `0.80` heuristic; real shipped-MDP controls verified in `r2_probe_1c`), and 10 of 29 targeted mutants survived (`r2_probe_3_mutations.json`; Section 5).

## 1. Round-1 blocking findings of both reviewers
Replayed all original round-1 blocking inputs against `c0917aa7` (`r2_probe_1c_r1_blockers_and_table.py` -> `r2_probe_1_results.json`):
- **Codex 1 (R1 — post-poll unknown-outcome alert & untrusted-record attempt recovery):** Replayed `probe_durable.py` (`locked-unconfirmed`, `attempt-invalid-action`, `attempt-invalid-record`, plus 5 other poll cases and 12 decision-reader cases). `locked-unconfirmed` retains exact `decision.json` bytes and sends one receipt-deduplicated `DELIVERY OUTCOME UNKNOWN` DM; `attempt-invalid-action` (`action: []`) and `attempt-invalid-record` (`{`) both recover `locked_unconfirmed` immediately on run 1 with one unknown-outcome alert and no `TypeError`.
- **Codex 2 (R2 — stale gap feasibility after status poll):** Replayed `probe_gap_window.py` (`correction=False` and `True`). After the 2s status read advances `18:42:59 -> 18:43:01 ET`, line `3532` replans `gap_no_feasible_window` (`deliver`), delivering Game `101` when valid and emitting both `[contest_state]` and `[missed_pick]` under a `best` correction.
- **Codex 3 (R3 — transient `EIO` clearing known-unsent marker):** Replayed `probe_delivery.py` `cutoff_clear_error` (`dm`, `public`) and `test_wr_w3.py` midnight `clear_error`. `allow_reused_transport()` retries `save_pick` once outside the transport `try/except` and completes the `[late_delivery]` / `[contest_state]` refusal.
- **Carbon B1 (R4 — void-before-cutoff on reused candidate):** Replayed both states (`deliver_past_cutoff` at `19:05:30 ET` and `defer_save_crosses_cutoff` at `19:04:56 -> 19:05:01 ET` on `Postponed` Game `101` under shipped MDP). Both record `reason="game_not_playable"` and send `[postponed_pick]` without naming the batter (`late_alerts=0`).

## 2. Round-1 required probes on `c0917aa7`

| Row / requirement | Evidence on `c0917aa7` | Result |
| --- | --- | --- |
| **M1** Overrun check reuses candidate before cutoff (`duration_sec=0`, 1 cascade) | `r2_probe_1c` (`m1_overrun`, `codex_bind`) | **PASS** |
| **M2** Single (`runner_up`), double, genuine MDP skip (Warmup cache + pick flip; skip emits no key) | `r2_probe_1c` (`m2_*`, `m4_*`, `codex_pol`) | **PASS** |
| **M3** Pick swapped, audit-replaced, `save_pick`-wrapped, whitespace, absent, dangling, bad JSON/shape/time, dir, FIFO | `r2_probe_1c` (`m2_*`, `m9_*`, `codex_pf`, `codex_cap`) | **PASS** |
| **M4** Slept fallback (`now < fallback_deadline`) runs fresh cascade (`reused_check_cascade=False`) | `r2_probe_1c` (`m3_sleep`, `codex_bind`) | **PASS** |
| **M5** Cutoff at read, reload, or marker write (`dm`, `public`, including transient clear `EIO`) -> `[late_delivery]` | `r2_probe_1c` (`m5_*`, `r3_delivery_clear_error_and_guards`) | **PASS** |
| **M6 (Check 1)** Contest-state drift (`dated` best/saver, `automatic`->`dated`, `save_pick` hook) -> `[contest_state]` | `r2_probe_1c` (`m6_*`, `codex_ws`), `pytest_focused_r2.log` | **PASS** |
| **M7 + Check 2 LW3-1/LW3-2** Midnight (`06-18`) & New Year (`12-31`) rollover before/during read & marker (`dated`/`automatic`) | `r2_probe_1c` (`lw3_2_*`, `r3_delivery_*`) | **PASS** |
| **M8 (with R4)** Void game: timely & past-cutoff all-void (`deliver`/`defer`) -> `[postponed_pick]`; 1-leg void double delivers | `r2_probe_1c` (`m8_*`, `r4_carbon_void_at_cutoff_blocker`) | **PASS** |
| **M9 + Check 2 M9 wording** Pending window replans; reuse defer never coalesces overdue checks; filters prior `coalesced` | `r2_probe_1c` (`m3_defer_no_coalesce_no_inherit`, `codex_sched`) | **PASS** |
| **M10** Warmup/cached selection mismatch -> `_reuse_commit_selection` passes `None`, commits saved pick rows | `r2_probe_1c` (`m2_warmup_cache`, `m2_nonmatching_selection`) | **PASS** |
| **M11 + Check 2 4c** Stale `automatic` stamped before marker; 4c warning only after confirmed `msg_id`/`uri` | `r2_probe_1c` (`m7_*`, `r3_delivery_*`) | **PASS** |
| **M12 + Rulings 1 & 2** Keyword `reused_check_cascade: bool = False`; 14/14 non-reuse callers & EOD/polling byte-identical | `r2_probe_1c` (`ruling1_signature`, `probe_3_checked=14`, `codex_leg`) | **PASS** |
| **Check 2: Decision read, post-poll I/O & 2s budget** `O_NONBLOCK`+`S_ISREG`; 2.0s shared deadline; `<=3s` skip; late worker isolated | `r2_probe_1c` (`c2_dec_*`, `shared_budget_*`, `codex_look`) | **PASS** |
| **Standing: Interrupts, failed/unconfirmed send & restart** `KeyboardInterrupt`/`SystemExit` across poll/marker/clear/commit; durable restart | `r2_probe_1c` (`r3_delivery_*`), `r2_probe_2d` (`r3_interrupts_*`) | **PASS** |

## 3. What the round-1 fixes changed and edge analysis
Probed in `r2_probe_2d_changes_and_optext.py` -> `r2_probe_2_results.json`:
- **Second planner run (`R2`, `scheduler.py:3531-3532`):** Monotonic `now` and `confirmed_sides` cannot flip `deliver -> defer`; `defer -> deliver` occurs only on `gap` when an overdue contender window expires before `cutoff`. When `now >= cutoff` after status polling (`r2_poll_crosses_cutoff_keeps_defer_void_order`), `now < cutoff` prevents flipping to `deliver`, preserving `defer_void_games` and the 1-call void-before-late refusal. See **N1** for the `defer -> deliver` flip on a void game.
- **Refusal-path retry (`R3`, `scheduler.py:1443-1446,1480,1517`):** When `save_pick` fails **twice** (`r3_double_fail_*` across `dm`/`public` and cutoff/midnight), `OSError` propagates out of `run_day` before transport and before the refusal alert, leaving `delivery_attempted=True` on disk; on restart, C2 locks `locked_unconfirmed` and pages `DELIVERY OUTCOME UNKNOWN` (blocking later-game re-pick on a 2-game slate, matching `59b72c04` on persistent write failure). For `KeyboardInterrupt`/`SystemExit` (`r3_interrupts_in_marker_clear`, 6/6 cases), `except OSError:` never catches `BaseException`: an interrupt before `os.replace` on either attempt leaves `delivery_attempted=True` (the standard E2 post-marker window), while an interrupt after `os.replace` inside `append_lineup_evolution` leaves `delivery_attempted=False` and `prior_reading` restored.
- **Unknown-outcome alert & receipt (`R1`, `scheduler.py:3170-3171`):** Uses `health_state/delivery_outcome_unknown/<date>.json` (`now_et_date=slate`, `require_message_id=True`). Verified 1 total alert across reuse -> same-day/next-day C2 restart and C2 restart -> reuse, plus retry on restart when reuse's send returned `""` (`status="failed"` -> `"sent"`).
- **Attempt recovery with untrusted record (`R1`, `scheduler.py:3520-3526`):** Tested `delivery_attempted=True` with all 7 untrusted `decision.json` kinds (`malformed_json`, `invalid_action`, `wrong_date`, `dangling`, `fifo`, `unreadable`, `directory`); all 7 lock and send 1 unknown-outcome alert without blocking (`os.replace` heals `fifo`/`dangling`; `directory` cannot be replaced so `committed_pick_written` stays `False` and `[missed_pick]` also fires).
- **Reordered void/late refusals (`R4`) & shadow trigger (`R5`):** Void precedes cutoff on both `defer` (`3543-3552`) and `deliver` (`1368-1382`), while a 1-void-leg double or `<=3s` unpolled expired defer still takes `[late_delivery]`. All 4 early reuse breaks (`3373,3377,3510,3525`) invoke `finish_fallback_lock` (`r5_shadow_all_four_breaks_and_optext`).

## 4. Operator text

| New/changed sentence (`src/bts/scheduler.py`) | Channel | Worst state if followed literally |
| --- | --- | --- |
| `"ET date changed before contest state could be validated"` (`1417`) | stderr + CRITICAL `[contest_state]` DM | Both `read_live_state()` attempts straddled ET midnight; asks for no manual action; pick stays unattempted so Step 5 or a later check can run. |
| `"ET date changed after the final contest-state read"` (`1450`) | stderr + CRITICAL `[contest_state]` DM | Midnight crossed after validation; clears known-unsent marker (with 1 retry); on valid `automatic` state Step 5 may deliver seconds later; asks for no manual action. |
| `"  FALLBACK: reusing this check's fresh cascade decision (no second refresh)."` (`3533`) | stderr journal | Now printed only after post-poll validation succeeds (never on abandoned reuse); still precedes `3542`/`3574` void/late/contest refusals, which immediately log `DELIVERY REFUSED` or `CONTEST STATE ERROR`. |
| `"  Pick locked. Stopping lineup checks."` (`3176`) | stderr journal | Now also printed on the 4 early reuse breaks; when `preserve_reuse_commit` locks a `locked_unconfirmed` record (`3372,3509`), stderr prints only this line without a `DELIVERY OUTCOME UNKNOWN` stderr line, though the CRITICAL DM alert is sent and this line gives no entry instruction. |

## 5. Tests and scratch mutations
- **Focused suite (`pytest_focused_r2.log` via `yield_run.sh`):** 11 required files in one process: `342 passed, 4 skipped in 23.81s`; save-hook regressions (`test_wr_a_pick.py`, `test_wr_a_review_carbon.py`): `13 passed in 2.23s`.
- **Mutations (`r2_probe_3_mutations.py` -> `r2_probe_3_mutations.json`, 29 mutants, `163.1s`):**
  - **Killed (19):** `m01`–`m03`, `m04b`, `m05b`, `m06a` (split 2s budget), `m06b` (late worker mutates `confirmed_sides`), `m07a`, `m08`, `m14a`, plus **R2 mutants** `r2_m01` (drop second planner run), `r2_m02` (drop marker-clear retry), `r2_m03` (drop `locked_unconfirmed` alert), `r2_m04` (late before void refusal), `r2_m05a`/`r2_m05c`/`r2_m05d` (drop shadow on 3 break paths), `r2_m06` (require trusted record for post-poll attempt), `r2_m07` (print reuse journal early).
  - **Survived (10):** `r2_m05b` (drop `finish_fallback_lock` on pre-poll `delivery_attempted` break at `3377` — unreachable without a `save_pick` hook mutating `daily` during `run_single_check`), `m04a` (skip post-read ET-day check in `read_live_state`, masked by `allow_reused_transport` + Step 5), `m05a` (drop pre-poll scoreable check, masked by `3508`), `m07b`/`m11` (ignore `double_down` slot mismatch in `_reuse_commit_selection`/`_pick_matches_reuse_commit`), `m09` (blocking read on post-poll pick reload), `m10` (drop `prior_reading` restore), `m12` (drop `coalesced` filter), `m13` (`>3`s guard relaxed to `>0`s), `m14b` (drop `daily is candidate` identity check).

## What I ran / What I did not run
- **Ran:** `r2_probe_1c_r1_blockers_and_table.py` (all 4 R1 blockers, all 11 Codex R1 probe modules, 57 Carbon R1 rows, 14 parity cases vs `ceab9854`), `r2_probe_2d_changes_and_optext.py` (10 R1–R5 adversarial & operator-text suites), `r2_run_focused_pytest.py` (342+13 passed via `yield_run.sh`), `r2_probe_3_mutations.py` (29 targeted mutants via `yield_run.sh`).
- **Did not run:** Full repository fast-suite gate (run by author); live network/account calls (prohibited).
