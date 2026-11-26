# Adversarial review, round 1: WR-C3 (`c70effe9` vs `c2039541`)

## Verdict
**BLOCK** (`c70effe9` against `c2039541`). One contract-violating defect in the new `_fold_season` directory scan / decision-parent check can silently omit a committed day and overwrite `streak.json`.

## Blocking findings
1. **`_fold_season` swallows `PermissionError` in `Path.glob` and leaves the `<date>` decision directory component unvalidated (`src/bts/picks.py:921-937`, `:839-845`).**
   - **Concrete state & inputs:** (a) `picks/2027-04-01.json` is a resolved hit; `picks/2027-04-16/decision.json` is a scoreable commit (`scoreable: true`), `picks/2027-04-16.json` is absent, `picks/2027-04-16` is `chmod 000`, and `streak.json` is `(9, False, 2027)`. (b) `picks/2027-04-01.json` is a hit, `picks/2027-04-16.json` is a delivered pending pick (`result=None`), `picks/2027-04-16` is a symlink to a directory holding a `scoreable: false` skip decision (or `picks/2027-04-16` is a FIFO with `2027-04-16.json` absent), and `streak.json` is `(9, False, 2027)`. (c) `picks_dir` holds 10 hits + 1 saver miss (`streak.json` `(10, False, 2027)`) with mode `0o300` (`wx`, unreadable listing), called via `picks.reconcile_results(picks_dir, ...)`.
   - **Wrong production outcome:** In (a) and (b), `bts reconcile` exits `0` with empty `stderr`, prints `No scoring changes detected. Streak: 1`, and overwrites `streak.json` from `(9, False)` to `(1, True)`. In (c), `scoring_lock` and `_read_reconcile_streak` open `.scoring.lock` and `streak.json` by exact name while `picks_dir.glob(...)` silently returns `[]`, overwriting `streak.json` from `(10, False)` to `(0, True)`.
   - **Smallest fix:** In `src/bts/picks.py:921-935`, replace both `picks_dir.glob(...)` calls with `os.scandir(picks_dir)` (converting `OSError` to `NonRegularReplayFile`), add every `f"{season}-*"` entry whose `.json`-stripped or raw name matches `_ISO_DATE_RE` to `dates`, and check `stat.S_ISDIR((picks_dir / day).lstat().st_mode)` before `_read_reconcile_file(decision_path(day, picks_dir))` (raising `NonRegularReplayFile` otherwise).
   - **How demonstrated:** `probe_blocker_glob_and_parent.py` (`blocker_glob_results.json`) reproduced all four wrong `streak.json` overwrites on `c70effe9` and verified the in-memory `_fixed_fold_season` control blocks (a)–(b) with `Season replay unavailable for 2027-04-16 (unreadable or invalid decision record)` keeping `(9, False)`, raises `NonRegularReplayFile` on (c) keeping `(10, False)`, and preserves healthy skip controls `(1, True)`. RED tests in `adopt/test_wr_c3_blocker_directory_scan.py`.

## Notes
- **Untouched special-file paths (`src/bts/picks.py:767`, `:498`):** Bounded subprocess probes showed `bts reconcile` hangs (>1.5s timeout) if untouched `.scoring.lock` is a FIFO or (during a correction save) `lineup_evolution_<date>.jsonl` is a FIFO, and exits 1 with raw `IsADirectoryError`/`FileExistsError` if `.scoring.lock` is a directory or `--picks-dir` is a file/FIFO.
- **Untouched concurrent receipt mkdir race (`src/bts/receipt_io.py:49`):** `ensure_dir` calls `d.mkdir()` without `exist_ok=True`; two simultaneous `bts reconcile` runs on an uncreated `reconcile_receipts/<date>` dir serialize `streak.json` properly under `scoring_lock` while one logs `reconcile receipt unavailable (FileExistsError)` and drops its receipt (`e2e_tail_results.json`).
- **Unmocked atomicity test gap (`tests/test_wr_c3_reconcile.py:328`):** Mocking `picks.atomic_write_text` allowed mutant `8b` (in-place `Path.write_text` when unmocked) to survive `test_wr_c3_reconcile.py` alone (killed by `test_streak_season_cs2b_review.py:582`). Added unmocked `os.replace` failure + `chmod 000` test in `adopt/test_wr_c3_review_adopted.py`.
- **Receipt schema doc (`docs/ops/reconcile-receipt-v1.md:32,41`):** Not yet updated for day state `invalid_pick_file` or `replay` keys `day`, `blocked_date`, `reason`, `retained_streak`.

## 1. Compatibility corpus (Base `c2039541` vs HEAD `c70effe9`)
Built **435 season histories** (`probe_corpus_and_fold.py`, `probe_corpus_v2.py`, `corpus_results.json`) spanning every repo writer (`save_pick` from `6798410`'s extra `"streak"` key onward, pre-`pitcher_team`, `2026-03-29/30` `game_pk: null`, pre-DM public picks, `regenerate._record_to_pick_file` single + teamless DD, `regenerate._hist_to_pick_file` None-backfills, `export_initial_state` round-trips, `v1`–`v3` decisions, partial/whole voids, legacy DD without `slot_results`) plus malformed/hostile shapes:
- **279/435 identical** (`same`), including **all 210 seeded valid legacy/modern season histories**, UTF-8 BOM (both reject via `json.loads`), extra top-level keys, `game_pk: null`, `{}` `double_down`/`slot_results`, AppleDouble/shadow/archive/future files.
- **27/435 category (a)** (required by contract): uncommitted past `None`/`suspended`/`unresolved` picks and skip leftovers skipped; foreign-season markers ignored.
- **129/435 category (b)** (contract blockers): Base folded any `miss`/`void` dict without calling `load_pick` and ignored `decision.json`; HEAD blocks on JSON arrays/primitives, `{"result":"miss"}`, missing `runner_up`/`run_time`/`pitcher_id`, extra slot keys (`p_hit_pa`), `date != stem`, unknown `result`, non-terminal `slot_results`, non-bool delivery flags (incl. explicit `null`), non-str `run_time`/`notification_id`/`batter_name`/`team`, non-exact-int `batter_id` or non-null `game_pk` (numeric strings/floats/bools), malformed `today` pick, invalid/mismatched `decision.json` beside terminal picks, and missing root pick with scoreable decision.
- **0/435 category (c) on real-writer shapes:** No repo writer emits the rejected scalar shapes, and `docs/audit/2026-09-28-season-ledger.md` compiled all 156 real 2026 pick files and 92 decisions with `0` quarantined pick slots (`batter_id` exact int, `game_pk` exact int or `null`) and `0` type-mismatch rows. **Uncheckable without box `data/picks`:** explicit JSON `null` in `run_time`/`batter_name`/`team`/delivery flags, omitted `runner_up` on hand-created `miss`/`void` files, non-terminal strings in `slot_results`, or `date != stem` (all checked by the dry-run command below).

## 2. Independent from-scratch fold
`independent_fold` (`probe_corpus_and_fold.py`) matched HEAD on **435/435** histories (`indep_mismatches: 0`), covering hit/miss/void, legacy DD (`+2`), partial-void DD (`+1`), saver consumption on the first miss in `[10, 15]` (tested at 9, 10, 15, 16) and never again, ascending in-season dates through `today`, terminal `today` included, unmarked unresolved `today` excluded, decision-only, pick-only, and combined days.

## 3. Skipped vs blocking states and contest-entry safety
`probe_reconcile_e2e_v4.py` verified every CLI outcome and stderr line `Season replay unavailable for <date> (<reason>)`:
- **Skipped (0 contribution, empty stderr):** refused preview (`chosen_on.required=False`); `scoreable=False` skip with or without leftover pick; past/today unresolved uncommitted pick with foreign-season marker; unmarked unresolved `today` pick/commit.
- **Blocked (retains `streak.json`):** private pending, `locked_unconfirmed` pending, delivered pending/`suspended`/`unresolved` (`committed pick awaiting a result`); bare `delivery_attempted` (`delivery attempt without a decision or result`); scoreable decision without root pick (`committed pick file missing`); same-season or truly tagless marker on unresolved past/`today` day (`applied-date marker without a result`, retaining `0` when tagless); invalid/non-regular pick (`unreadable or invalid pick file`) or decision (`unreadable or invalid decision record`).
- **Contest-entry trace:** Apart from the symlinked decision-parent case in Blocking Finding 1 and the pre-existing lost-DM-response race (`W11`, where `send_dm` raises after remote acceptance on final fallback, clearing `delivery_attempted` so neither `check-results` nor replay scores it), every delivered/attempted/private/crash-recovered commit sets `scoreable=True` or `delivery_attempted=True` and blocks replay until graded.

## 4. `bts reconcile` end to end
Verified (`probe_reconcile_e2e_v4.py`, `probe_reconcile_e2e_v5.py`): pre-08:00 ET `hit->miss` and `miss->hit` corrections vs >=08:00 ET cutoff; `"Streak retained"` iff `state == "unavailable"` and `"Streak recalculated"` iff replay saved with corrections; 5 corrupt `streak.json` payloads (`{bad`, `[]`, string/bool streak, 5000-digit int) exit `1` without traceback and keep bytes when blocked, and rebuild atomically when complete; bounded subprocess probes (FIFO/dir/symlink) on recent/old picks, `decision.json`, `streak.json`, `picks_dir`, and `receipts_root` never hung; two concurrent reconciles serialized cleanly; `KeyboardInterrupt` after correction save preserved the correction and `streak.json`, and retry refolded `(2, True, 2027)` with 0 new corrections.

## 5. Other callers
`probe_reconcile_e2e_v5.py` ran Base (`c2039541`) and HEAD in isolated subprocesses across `update_streak`/`save_streak` (14 steps with pinned UTC clock: **byte-identical** `streak.json`), `parse_daily_pick`/`load_pick_bytes`, `bts check-results`, `scheduler.run_result_polling`, `web.load_streak`/`load_all_picks`, `health.streak_validation`, and `health.contest_state`: **100% identical**.

## 6. Edited existing tests
Verified `git diff c2039541 HEAD -- tests/`: only the three authorized edits (`tests/test_streak_replay.py:71` delivered pending fixture per Amendment 1; `tests/test_streak_season_cs2.py:385` blocked-replay mock + `Streak retained: 0` per Amendment 2; `tests/test_streak_season_cs2b_review.py:602` explicit `notification_id` per `contract-check.md`), with every fail-closed/season assertion intact.

## 7. New tests, wall clocks, and mutation sweep
`probe_item7_v2.py` (`item7_results.json`):
- **On Base (`c2039541`):** **25 failed, 1 passed** (`test_legacy_2026_terminal_history_keeps_its_base_fold` control green; 2 failed on unskipped preview, 2 hung on pick/`streak.json` FIFOs, 3 crashed on `[]`/recent invalid JSON, 17 lacked dated stderr/season-marker behavior, 1 skipped `save_streak` on blocked replay).
- **Focused one-process run:** `585 passed, 8 skipped, 1 failed` (`test_web_saver.py::test_do_post_saver_transition_socket_wiring` sandbox loopback `PermissionError` only). Under pinned wall clocks in **2027** (Jul 15, Dec 31) and **2028** (Jan 1, Jul 15): **442 passed, 0 failed** each.
- **Mutants (`1`–`8`):** Mutants 1–7 and `8a` (direct `Path.write_text`) were **KILLED** by `test_wr_c3_reconcile.py`; mutant `8b` (unmocked non-atomic write) **SURVIVED** `test_wr_c3_reconcile.py` and was **KILLED** by `test_streak_season_cs2b_review.py` (and `adopt/test_wr_c3_review_adopted.py`).

## 8. Other diff checks
Checked `ReconcileReceipt.replayed` degradation safety, `effective_daily_result` slot-only updates, and UTC/ET New Year lock rollover; nothing else in the diff was wrong beyond Blocking Finding 1 and the Notes above.

## For the restored box
Run read-only from the repo root **before** the first `bts reconcile`:
```bash
.venv/bin/python -c '
import hashlib, json, subprocess, sys, tarfile, tempfile
from pathlib import Path
picks = Path("data/picks"); sp = picks / "streak.json"
before = (sp.stat().st_mtime_ns, hashlib.sha256(sp.read_bytes()).hexdigest()) if sp.exists() else None
probe = """import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import bts.picks as p
d = Path(sys.argv[2]); out = {"r2026": p._replay_season_streak(d, 2026, "2026-12-31"), "bad": []}
if hasattr(p, "_fold_season"):
    st = p._read_reconcile_streak(d / "streak.json")
    out["fold2026"] = p._fold_season(d, 2026, "2026-12-31", p._replay_markers(st or {}, 2026))
    for f in sorted(d.glob("2026-*.json")):
        if p._ISO_DATE_RE.fullmatch(f.stem):
            try: p._read_reconcile_pick(d, f.stem)
            except Exception as e: out["bad"].append([f.name, type(e).__name__, str(e)])
print(json.dumps(out))"""
with tempfile.TemporaryDirectory() as tmp:
    with subprocess.Popen(["git", "archive", "c2039541", "src"], stdout=subprocess.PIPE) as a:
        with tarfile.open(fileobj=a.stdout, mode="r|") as tf: tf.extractall(tmp, filter="data")
        assert a.wait() == 0
    old = json.loads(subprocess.check_output([sys.executable, "-c", probe, f"{tmp}/src", str(picks)], text=True))
    new = json.loads(subprocess.check_output([sys.executable, "-c", probe, "src", str(picks)], text=True))
after = (sp.stat().st_mtime_ns, hashlib.sha256(sp.read_bytes()).hexdigest()) if sp.exists() else None
print("OLD:", old, "\nNEW:", new)
assert before == after and old["r2026"] == new["r2026"] and old["r2026"] is not None and not new["bad"], "MISMATCH"
print("DRY COMPARISON OK")
'
```

## What I ran
`~/bts-runs/yield_run.sh` was denied by the sandbox (`operation not permitted`), so per the brief's fallback instruction all pytest suites ran sequentially one at a time after 03:34 ET under `/usr/bin/nice -n 19`:
- `.venv/bin/python .codex-review/wr-c3-review/probe_corpus_v2.py` (435 histories, Base vs HEAD vs independent fold)
- `.venv/bin/python .codex-review/wr-c3-review/probe_reconcile_e2e_v4.py` & `probe_reconcile_e2e_v5.py` (CLI matrix, bounded special-file subprocesses, concurrency, interrupt+retry, caller parity)
- `.venv/bin/python .codex-review/wr-c3-review/probe_blocker_glob_and_parent.py` (Blocking Finding 1 reproduction + fix control)
- `/usr/bin/nice -n 19 .venv/bin/python -m pytest` on the 594-item focused suite, `.codex-review/wr-c3-review/probe_item7_v2.py` (Base RED, 4 pinned 2027/2028 wall clocks, 9 mutants), and `.codex-review/wr-c3-review/adopt/test_wr_c3_review_adopted.py` (`30 passed`)
