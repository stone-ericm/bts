# Verdict

**BLOCK** `aa61bb43` against `c2039541`. All four round-1 Codex counterexamples and all 35 category-(c) corpus regressions are fixed, but eight real-producer entered-pick states (`N1`–`N8`) and one symlinked decision-parent state (`b2`) still cause `bts reconcile` to skip a day that `c2039541` refused to skip and overwrite `streak.json` with `(0, True)` (even when `streak.json` already held the true `(10, False)`).

## Blocking findings

1. **Non-scoreable EOD `decision.json` written after an unmarked entered send skips the entered pick (`src/bts/picks.py:936-961`).**
   - **Concrete state & producer sequence:** 9 hits (`2027-04-01..09`), entered hit on `2027-04-18`, `2027-04-19` miss (true fold `(10, False)`). (`N1`/`N2`) Live DM `bts schedule` (`contest_state_required=true`, `entry_intent="enter"`): `run_single_check` saves `2027-04-18.json` (`chosen_on.required=true`); `_deliver_and_lock_pick` (`src/bts/scheduler.py:1314-1335`) stamps `automatic_reading` (`N1`), saves `delivery_attempted=true`, sends DM (accepted + entered), catches `TimeoutError`, and clears `delivery_attempted=false` with `pick_locked=false`. Minutes later in the same `run_day`, `_refresh_pick_at_fallback_decision` (`:2303-2325, :3236-3242`) sees updated predictions below the MDP bar (`action="skip", source="mdp"`), sets `final_skip_candidate` without removing `2027-04-18.json`, and `_write_endofday_skip` (`:986-1010, :3505`) writes `2027-04-18/decision.json` (`scoreable=false`). (`N3`) Morning MDP skip in `final_skip_candidate`, operator `bts run` (`src/bts/cli.py:1257, 1280`) posts to Bluesky (accepted + entered) and crashes before saving `bluesky_posted=true`, then daemon EOD writes `decision.json` (`scoreable=false`).
   - **Wrong saved streak:** Base (`c2039541`) returns `None`, keeping `(10, True)` on overturn and keeping an already-true `(10, False)`. `aa61bb43` skips `2027-04-18` with empty `stderr` and saves `(0, True)`.
   - **Smallest fix & demonstration:** In `src/bts/picks.py:936-961`, include `raw_data.get("automatic_reading") is not None` in `delivery_signal` and refuse to skip when `daily.chosen_on` has `required == True` beside a non-scoreable decision (`"live-selected pick beside a skip record"`), or (if producer edits are allowed) retain a durable attempt marker in `scheduler.py:1332,:1363` and pre-post in `cli.py:1279`. Demonstrated in `probe_1_and_2_v5.py`, `probe_5_7_and_fixes_v3.py`, and `adopt/test_wr_c3_r2_entered_pick_blockers.py`.

2. **`chosen_on.required == false` is emitted/preserved on entered picks while `scheduler_state.json` is reset on restart, quarantined, or absent (`src/bts/picks.py:948-961`).**
   - **Concrete state & producer sequence:** Same season history; `2027-04-18.json` has `result: null`, no delivery flags, and valid `chosen_on` with `required: false`; `decision.json` is absent; `scheduler_state.json` is `pick_locked: false` (`N4`, `N5`), quarantined to `scheduler_state.json.corrupt-<ts>` (`N6`), or absent (`N7`, `N8`). Amendment 3's premise that *only* `bts preview` writes `chosen_on.required == false` is false: (`N4`, `N6`) `03:00` `bts preview` (`cli.py:1371`) writes `required=false`; live private `bts schedule` (`contest_state_required=true`, `entry_intent="research"`) reuses cached `daily` on fallback cascade failure (`scheduler.py:2316,:2325` or Warmup `strategy.py:549`), `_deliver_and_lock_pick` (`scheduler.py:1262-1267`) runs *before* `saved_pick_refusal` (`:1284`), saves `pick_locked=true` (dashboard `LOCKED`, entered) keeping `chosen_on.required=false`, and crashes before `_record_commit`; then (`N4`) same-day `run_day` restart (`scheduler.py:2935-2957` + `carry_forward_skip_state` `:1768`) overwrites `scheduler_state.json` with `pick_locked=false`, or (`N6`) `load_state` (`:1053-1055`) renames torn state to `scheduler_state.json.corrupt-<ts>` (ignored by `picks.py:948`). (`N5`, `N7`) `orchestrator.run_and_pick` (`src/bts/orchestrator.py:283, 308`) stamps `chosen_on.required=false` on *every* fresh pick when `_contest_state_required(config)` is `False` (allowed by `entry_intent.py:51` for `private` in `N5`, and via `orchestrate()` post crash in `N7`). (`N8`) `bts preview` (`cli.py:1319` checks only `result or bluesky_posted`) overwrites an unmarked DM/live `2027-04-18.json` with `required=false`.
   - **Wrong saved streak:** Base preserves `(10, True)` / `(10, False)`; `aa61bb43` skips `2027-04-18` silently and writes `(0, True)`.
   - **Smallest fix & demonstration:** In `src/bts/picks.py:948-960`, block when any `scheduler_state.json.corrupt-*` exists in `picks_dir / day`, and require an explicit `"preview_only": true` stamp when `decision is None` (plus, if producer edits are allowed: stamp `"preview_only": true` and guard `existing is not None` in `cli.py:1319, 1371`, clear `daily.chosen_on` on private lock in `scheduler.py:1263`, and copy `pick_locked`/`pick_locked_at` in `scheduler.py:1774`). Verified in `probe_5_7_and_fixes_v3.py` and `adopt/test_wr_c3_r2_entered_pick_blockers.py`.

3. **Symlinked `<date>` directory lets an external skip `decision.json` drop a day Base blocked (`src/bts/picks.py:841, 852, 995`).**
   - **Concrete state:** `2027-04-01.json` hit; `2027-04-16.json` unresolved live pick (`chosen_on.required=true`, no delivery flag, e.g. `lost_ack`); `picks/2027-04-16` symlinked to a dir with `decision.json` (`scoreable=false`); `streak.json` `(9, False)`. Base keeps `(9, False)`; `aa61bb43` checks `lstat`/`O_NOFOLLOW` only on the leaf `decision.json`, reads `scoreable=false` through the symlink, and writes `(1, True)`.
   - **Smallest fix & demonstration:** Check `stat.S_ISDIR((picks_dir / day).lstat().st_mode)` before reading `<day>/decision.json` or `scheduler_state.json` (else raise `NonRegularReplayFile`). Verified in `probe_5_7_and_fixes_v3.py`.

## Notes

- **Pre-existing directory-scan drops (`src/bts/picks.py:967-968`):** Both Base and `aa61bb43` silently drop `chmod 000` `<date>/` with no root pick (`a`) and `chmod 0300` `picks_dir` in `reconcile_results` (`c`).
- **Blocker reason ordering (`src/bts/picks.py:943-945`):** Real `scheduler.py:1315-1324, 1347-1355` leaves `delivery_attempted=True` on confirmed deliveries; checking `if daily.delivery_attempted:` first labels a confirmed pick without `decision.json` `"delivery attempt without a decision or result"` instead of `"committed pick awaiting a result"`.
- **Surviving type-guard placement (`src/bts/picks.py:932-935`):** Non-string `bluesky_uri`/`delivered_at` is checked only in `_past_pick_blocker` (missed on `today` and correction reads; mutant `M6` survived the full suite).

## 1. Worst-case entered-pick hunt

Re-ran Codex's four round-1 counterexamples (`codex_lost_ack`, `codex_skip_conflict`, `codex_private_crash`, `codex_manual_run_crash`) on `c2039541`, `c70effe9`, and `aa61bb43`: all four now block on `aa61bb43` and preserve `streak.json` byte-for-byte. Extending those real writers (`scheduler.py`, `cli.py` `run`/`preview`/`check-results`, `orchestrator.py`) to end-of-day, same-day restart, and state quarantine exposed `N1`–`N8` (Blocking Findings 1–2 above); `check-results` (`applied_dates` written before `save_pick` under `scoring_lock`) and `web.py` (`/saver/transition` touches only `saver_state.json`) had no entered-pick skip window.

## 2. Carbon's round-1 directory-scan finding

- **(a) `chmod 000` `picks/2027-04-16/` (`scoreable: true`, no root pick):** Both Base and `aa61bb43` drop the day (`(9, False) -> (1, True)`) — pre-existing NOTE, though the contract sentence *"a scoreable decision without a readable root pick blocks replay"* is unmet when `<date>/` is `000` because `Path.glob` swallows `PermissionError`.
- **(b) Symlink / FIFO `<date>` dir:** Delivered pending pick + symlinked skip dir (`b1`) now blocks on `aa61bb43` (`"skip record contradicts a delivery"`). FIFO `<date>` dir returns promptly without hanging with (`b3b`, blocks) or without (`b3`, folds `(1, True)`) a root pick. Unmarked live pick + symlinked skip dir (`b2`) blocks on Base `(9, False)` and drops to `(1, True)` on `aa61bb43` (**BLOCK**, Finding 3).
- **(c) `chmod 0300` `picks_dir`:** CLI `bts reconcile` fails closed on both (`(10, False)` kept); `picks.reconcile_results` drops to `(0, True)` on both — pre-existing NOTE.
- **Should the date scan fail loudly?** Yes: replacing the two `picks_dir.glob(...)` calls with one `os.scandir(picks_dir)` + `lstat`/`scandir` on `{season}-*` date dirs (~20 lines in `_fold_season`) converts unreadable listings into `NonRegularReplayFile` and unreadable `<date>` dirs into dated blockers, and benchmarked **faster** on a healthy 180-day season (`4.61 ms` vs `7.79 ms`, `-3.18 ms`, single directory pass).

## 3. Rule 1 on the corpora (`c2039541` vs `aa61bb43` + independent fold)

- **Codex 630 histories:** `560` identical, `70` diffs (`0` category-(c); all 35 regenerated-miss and numeric-string-ID cases match Base). Where Base returned a value, `aa61bb43` matched on every terminal history and differed only on `committed_without_root` (`7`, unresolved past commit missing root pick) and `today_unknown_result` (`7`, nonterminal `"result": "mystery"` on `today`, blocked per Replacement C3). Reverse (`21`: `refused_preview`, `skip_leftover`, `today_marker_foreign` × 7) + all `364` folded histories matched our clean-room `scratch_fold_season` (`0` mismatches).
- **Carbon 435 histories:** `386` identical, `49` diffs (`0` category-(c)). Base-value -> `None` diffs were only `21` scoreable-decision-missing-pick histories and `1` `blocker_today_malformed_minimal` (`{"result": null}` on `today`). Reverse (`14` uncommitted skips/previews and foreign markers) + all folded histories matched `scratch_fold_season` (`0` mismatches).

## 4. Rule 2 truth table (`5 × 7 × 4 × 4 × 4 × 2 = 4,480` cells)

`probe_3_4_corpus_and_table.py` evaluated `_fold_season` across all **4,480** cells: **0 mismatches** against Amendment 3 (`932` skipped; `1,792` `applied-date marker without a result`; `896` `unreadable or invalid decision record`; `384` `committed pick awaiting a result`; `384` `skip record contradicts a delivery`; `32` `delivery attempt without a decision or result`; `24` `scheduler state marks the pick locked`; `24` `unreadable or invalid scheduler state`; `12` `no decision or preview stamp`). Existing `tests/test_wr_c3_*.py` directly cover **34 / 4,480** cells, leaving untested: `non_scoreable_other` + delivery signals (`0/6`), `decision=None` + `bluesky_posted`/`bluesky_uri`/`notification_sent`/`delivered_at` alone (`0/4`), past `other_season` marker beside a valid preview/skip (`0`), and `today` with unreadable `decision.json` (blocks) or `skip` + delivery / corrupt `scheduler_state.json` (skipped on `today`) — added in `adopt/test_wr_c3_r2_coverage_and_mutants.py`.

## 5. `bts reconcile` end to end

Verified (`probe_5_7_and_fixes_v3.py`): `02:00` and `07:59:59` ET `hit<->miss` corrections (`Streak recalculated`), `08:00:00` ET cutoff hold, blocked replay (`Season replay unavailable for <date> (<reason>)` on stderr, `Streak retained`, bytes intact), 4 corrupt `streak.json` shapes exiting `1` without traceback when blocked and atomically replaced (new inode, open fd retains old bytes) when complete, and 5 FIFO targets returning in `<13 ms` under a `2 s` alarm.

## 6. Tests, wall clocks, and mutations

`~/bts-runs/yield_run.sh` was denied by the sandbox (`operation not permitted`), so all pytest suites ran sequentially one at a time via `/usr/bin/nice -n 19`:
- **RED runs (`tests/test_wr_c3_*.py`, 685 cases):** On `c70effe9`, **341 failed / 344 passed** (`322` corpus Rule-1 regressions, `4/4` delivery counterexamples, `15` positive-evidence table rows). On `c2039541`, **124 failed / 561 passed** (`70` corpus diffs, `4/4` delivery missing warning/receipt fields, `50/51` reconcile cases; only `test_legacy_2026_terminal_history_keeps_its_base_fold` passed).
- **Focused one-process run:** `1,180 passed, 1 failed` (`test_web_saver::test_do_post_saver_transition_socket_wiring` sandbox loopback `PermissionError` only). **2027 & 2028 wall clocks** (`2027-07-15`, `2027-12-31`, `2028-01-01`, `2028-07-15`): **`1,168 passed, 0 failed`** each.
- **Mutants:** All 5 requested mutants (`M1` drop lock check: 3 failed; `M2` missing `chosen_on` as preview: 24 failed; `M3` non-scoreable overrides delivery: 7 failed; `M4` strict terminal validation: 273 failed; `M5` skip unreadable decision: 4 failed) were **KILLED** by `tests/test_wr_c3_*.py`. Bonus mutant `M6` (drop `bluesky_uri`/`delivered_at` str check) **SURVIVED** all 1,168 tests; `M7` (unmocked non-atomic `_write_streak`) survived `test_wr_c3_*.py` and was killed by `test_streak_season_cs2b_review.py` (both killed by `adopt/test_wr_c3_r2_coverage_and_mutants.py`).

## 7. Restored-box dry comparison check

Both round-1 dry commands compared `c2039541` vs `c70effe9` instead of deployed **`619abda7`** vs **`aa61bb43`**, Codex's wrote scratch under `.codex-review/`, and neither isolated the first differing/blocking date on `619abda7` (which has no `_fold_season` and uses `f.stem >= today_iso`, pre-L02). Passing `f"{d}~"` for `d < horizon` evaluates prefix `<= d` as past on both revisions (`verify_dry_compare_box.py`).

## 8. Other diff checks

Nothing else in `git diff c2039541 HEAD` was wrong beyond the Blocking Findings and Notes above; the three round-1 existing-test edits remain unchanged and authorized.

## For the restored box

Run read-only from the repo root before the first `bts reconcile` (uses `/tmp` only; exits `0` on match, `2` on divergence, `3` when both block; or run `.codex-review/wr-c3-r2/dry_compare_box.py data/picks --today 2026-12-31` if this review folder is present):

```sh
env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '
import hashlib, io, json, os, re, subprocess, sys, tarfile, tempfile
from pathlib import Path
picks = Path(sys.argv[1]).resolve(strict=True); horizon = sys.argv[2] if len(sys.argv) > 2 else "2026-12-31"; yr = int(horizon[:4])
snap = lambda d: ((d/"streak.json").stat().st_mtime_ns, hashlib.sha256((d/"streak.json").read_bytes()).hexdigest()) if (d/"streak.json").exists() else None
before = snap(picks); rx = re.compile(r"^\d{4}-\d{2}-\d{2}$")
dates = sorted({s for e in os.scandir(picks) if e.name.startswith(f"{yr}-") for s in [e.name[:-5] if e.name.endswith(".json") else e.name] if rx.fullmatch(s) and s <= horizon})
w = """import json, socket, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import bts.picks as p
socket.socket.connect = socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net"))
d, yr = Path(sys.argv[2]), int(sys.argv[3])
for t in sys.stdin.read().splitlines():
    try:
        v = p._replay_season_streak(d, yr, t); o = {"state": "unavailable" if v is None else "ok", "value": list(v) if v else None}
        if v is None and hasattr(p, "_fold_season"):
            _, b = p._fold_season(d, yr, t, p._replay_markers(p._read_reconcile_streak(d/"streak.json") or {}, yr))
            if b: o["blocked_date"], o["reason"] = b
    except Exception as e: o = {"state": "raised", "value": None, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(o))"""
qs = "\n".join([horizon, *[f"{d}~" if d < horizon else horizon for d in dates]]) + "\n"; res = {}
with tempfile.TemporaryDirectory() as tmp:
    for rev in ("619abda7", "aa61bb43"):
        dst = Path(tmp)/rev; dst.mkdir()
        tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "archive", rev, "src"]))).extractall(dst, filter="data")
        res[rev] = [json.loads(x) for x in subprocess.check_output([sys.executable, "-c", w, str(dst/"src"), str(picks), str(yr)], input=qs, text=True).splitlines()]
assert snap(picks) == before
first = next(({"date": d, "619abda7": o, "aa61bb43": n} for d, o, n in zip(dates, res["619abda7"][1:], res["aa61bb43"][1:]) if o["state"] != "ok" or n["state"] != "ok" or o["value"] != n["value"]), None)
print(json.dumps({"619abda7": res["619abda7"][0], "aa61bb43": res["aa61bb43"][0], "first_diff_or_block": first}, indent=2))
sys.exit(0 if res["619abda7"][0]["state"] == "ok" and res["619abda7"][0] == res["aa61bb43"][0] else (3 if res["619abda7"][0]["state"] != "ok" and res["aa61bb43"][0]["state"] != "ok" else 2))
' data/picks 2026-12-31
```

## What I ran

- `probe_1_and_2_v5.py` (`c2039541`, `c70effe9`, `aa61bb43` across Codex's 4 cases, `N1`–`N8`, and Carbon's directory states)
- `probe_3_4_corpus_and_table.py` (Codex 630 + Carbon 435 corpora + clean-room fold + all 4,480 Rule-2 cells)
- `probe_5_7_and_fixes_v3.py` & `verify_dry_compare_box.py` (Item 5 e2e, bounded FIFOs, smallest-fix controls, 180-day scan benchmark, `619abda7` vs `aa61bb43` dry compare)
- Sequential `/usr/bin/nice -n 19 .venv/bin/python -m pytest` via `probe_6_tests_and_mutations_v2.py` (focused 1,181-test run, `c70effe9` & `c2039541` RED runs, 4 pinned 2027/2028 wall clocks, 7 mutants) plus `adopt/test_wr_c3_r2_coverage_and_mutants.py` (`23 passed`) and `adopt/test_wr_c3_r2_entered_pick_blockers.py` (`9 RED` on `aa61bb43`, `9 passed` under fix)
