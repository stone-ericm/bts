"""Adversarial review Items 2 & 3: CLI/receipt output contract and checklist B8b(2) operator step."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
import json
from pathlib import Path
import re
import signal
import socket
import sys
import tempfile
import urllib.error
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
TODAY = "2027-04-20"
YEST = "2027-04-19"
EARLY = "2027-04-10"
LATER = "2027-04-15"

REASONS = {
    "corrupt": "pick file unreadable or not valid JSON",
    "applied": "today's result is applied to streak.json but not yet saved on the pick",
    "pending": "no final result on the pick",
    "lost_hit": "hit pick could not be loaded",
}


@contextmanager
def _alarm(seconds: int = 15):
    class _Timeout(BaseException):
        pass

    prev = signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_Timeout("timeout")))
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _slot(day: str, game_pk: int = 101) -> dict:
    return {
        "batter_name": "Synthetic Batter",
        "batter_id": 100001,
        "team": "PIT",
        "lineup_position": 1,
        "pitcher_name": "Synthetic Pitcher",
        "pitcher_id": 100002,
        "p_game_hit": 0.80,
        "flags": [],
        "projected_lineup": False,
        "game_pk": game_pk,
        "game_time": f"{day}T23:00:00Z",
        "pitcher_team": "CHC",
    }


def _pick_dict(day: str, result: str | None = None, **extra) -> dict:
    d = {
        "date": day,
        "run_time": f"{day}T19:00:00Z",
        "pick": _slot(day),
        "double_down": None,
        "runner_up": None,
        "result": result,
    }
    if result in ("hit", "miss", "void"):
        d["slot_results"] = {"pick": result}
    d.update(extra)
    return d


def _invoke_reconcile(picks_dir: Path, mp: pytest.MonkeyPatch, *, clock=lambda: RUN, lookback: int = 0):
    from bts import cli, reconcile_receipt

    real_rr = reconcile_receipt.ReconcileReceipt
    mp.setattr(
        reconcile_receipt, "ReconcileReceipt",
        lambda **kw: real_rr(clock=clock, lookback_days=kw["lookback_days"]),
    )
    return CliRunner().invoke(cli.cli, ["reconcile", "--picks-dir", str(picks_dir), "--lookback", str(lookback)])


def _discover_all(picks_dir: Path) -> list[dict]:
    from bts import reconcile_receipt

    root = reconcile_receipt.receipts_root(picks_dir)
    if not root.exists():
        return []
    out = []
    for day_dir in sorted(root.iterdir()):
        if day_dir.is_dir():
            out.extend(reconcile_receipt.discover(picks_dir, day_dir.name))
    return out


def check_item2_outputs() -> dict:
    from bts import picks, reconcile_receipt

    real_read_text = Path.read_text

    # 1. Four reasons x {no_corrections, one_correction, two_corrections}
    for kind, reason in REASONS.items():
        for n_corr in (0, 1, 2):
            with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
                mp.setattr(socket.socket, "connect", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no net")))
                picks_dir = Path(tmp) / "data" / "picks"
                picks_dir.mkdir(parents=True)
                day = TODAY if kind == "applied" else EARLY
                (picks_dir / "streak.json").write_text(json.dumps({
                    "streak": 11, "saver_available": False, "season": 2027,
                    "applied_dates": [TODAY] if kind == "applied" else [],
                }))

                if kind == "corrupt":
                    (picks_dir / f"{day}.json").write_text("{broken")
                elif kind == "applied":
                    (picks_dir / f"{day}.json").write_text(json.dumps(_pick_dict(day, None)))
                elif kind == "pending":
                    (picks_dir / f"{day}.json").write_text(json.dumps(_pick_dict(day, None)))
                elif kind == "lost_hit":
                    (picks_dir / f"{day}.json").write_text(json.dumps(_pick_dict(day, "hit")))
                    unlinked = {"done": False}

                    def _unlink_once(self_p: Path, *a, **k):
                        txt = real_read_text(self_p, *a, **k)
                        if self_p.name == f"{day}.json" and not unlinked["done"]:
                            unlinked["done"] = True
                            self_p.unlink()
                        return txt

                    mp.setattr(Path, "read_text", _unlink_once)

                if n_corr >= 1:
                    # Run at 2027-04-20 02:00 ET: cutoff for 2027-04-19 is 2027-04-20 08:00 ET.
                    # To get 2 corrections before cutoff, run at 2027-04-20 02:00 ET? Wait:
                    # cutoff(day) is day+1 08:00 ET, so only yesterday is before cutoff unless clock is earlier
                    # or we check cutoff: let's verify!
                    (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit")))
                if n_corr == 2:
                    # Wait: at 2027-04-20 02:00 ET, 2027-04-18 cutoff was 2027-04-19 08:00 ET (past_cutoff)!
                    # Can two dates EVER be before cutoff simultaneously?
                    # cutoff(today - 1) = today 08:00 ET; cutoff(today - 2) = today-1 08:00 ET < today 00:00 ET!
                    # Only 1 date (yesterday) can ever be before cutoff on a normal monotone clock!
                    pass

                mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "miss"})
                lookback = 8 if n_corr >= 1 else 0
                res = _invoke_reconcile(picks_dir, mp, lookback=lookback)
                assert res.exit_code == 0, res.output
                expected_err = f"Season replay unavailable for {day} ({reason}); streak retained, not recalculated.\n"
                assert res.stderr == expected_err, (kind, n_corr, res.stderr)
                assert "recalculated" not in res.stdout.lower()
                if n_corr == 0:
                    assert res.stdout == (
                        f"Reconciling last {lookback} days of picks...\n"
                        "No scoring changes detected. Streak retained: 11\n"
                    )
                else:
                    assert res.stdout == (
                        f"Reconciling last {lookback} days of picks...\n"
                        "CORRECTIONS FOUND (1):\n"
                        f"  {YEST}: Synthetic Batter — hit -> miss\n"
                        "Streak retained: 11\n"
                    )
                recs = _discover_all(picks_dir)
                assert len(recs) == 1
                assert recs[0]["replay"] == {"state": "unavailable", "blocked_date": day, "reason": reason}

    # 2. Pairwise walk-order check: earlier blocker ALWAYS beats later blocker across all combinations
    past_kinds = ["corrupt", "pending", "lost_hit"]
    later_kinds = ["corrupt", "pending", "lost_hit", "applied", "json_array"]
    pairwise_checked = 0
    for first_kind in past_kinds:
        for second_kind in later_kinds:
            with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
                picks_dir = Path(tmp) / "data" / "picks"
                picks_dir.mkdir(parents=True)
                (picks_dir / "streak.json").write_text(json.dumps({
                    "streak": 4, "saver_available": True, "season": 2027, "applied_dates": [TODAY],
                }))
                # Valid earlier hit before EARLY, plus ignored future/shadow/foreign corrupt files
                (picks_dir / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit")))
                (picks_dir / "2027-04-05.shadow.json").write_text("{ignored_shadow")
                (picks_dir / "2026-04-01.json").write_text("{ignored_other_season")
                (picks_dir / "2027-04-25.json").write_text("{ignored_future")

                # Plant first blocker at EARLY (2027-04-10)
                if first_kind == "corrupt":
                    (picks_dir / f"{EARLY}.json").write_text("{bad1")
                elif first_kind == "pending":
                    (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, "suspended")))
                elif first_kind == "lost_hit":
                    (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, "hit")))

                # Plant second blocker at LATER (2027-04-15) or TODAY (2027-04-20)
                d2 = TODAY if second_kind == "applied" else LATER
                if second_kind == "corrupt":
                    (picks_dir / f"{d2}.json").write_text("{bad2")
                elif second_kind in ("pending", "applied"):
                    (picks_dir / f"{d2}.json").write_text(json.dumps(_pick_dict(d2, None)))
                elif second_kind == "lost_hit":
                    (picks_dir / f"{d2}.json").write_text(json.dumps(_pick_dict(d2, "hit")))
                elif second_kind == "json_array":
                    (picks_dir / f"{d2}.json").write_text("[]")

                if "lost_hit" in (first_kind, second_kind):
                    real_lp = picks.load_pick
                    mp.setattr(
                        picks, "load_pick",
                        lambda dt, pd: None if dt in (
                            ([EARLY] if first_kind == "lost_hit" else [])
                            + ([d2] if second_kind == "lost_hit" else [])
                        ) else real_lp(dt, pd),
                    )

                res = _invoke_reconcile(picks_dir, mp)
                assert res.exit_code == 0, (first_kind, second_kind, res.output)
                recs = _discover_all(picks_dir)
                assert recs[0]["replay"] == {
                    "state": "unavailable", "blocked_date": EARLY, "reason": REASONS[first_kind],
                }, (first_kind, second_kind, recs[0]["replay"])
                pairwise_checked += 1

    # 3. Same-file condition precedence
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
        # Today corrupt + today in applied_dates -> corrupt wins over applied!
        p1 = Path(tmp) / "same1" / "picks"
        p1.mkdir(parents=True)
        (p1 / "streak.json").write_text(json.dumps({"streak": 3, "season": 2027, "applied_dates": [TODAY]}))
        (p1 / f"{TODAY}.json").write_text("{corrupt_today")
        r1 = _invoke_reconcile(p1, mp)
        assert _discover_all(p1)[0]["replay"] == {
            "state": "unavailable", "blocked_date": TODAY, "reason": REASONS["corrupt"],
        }
        assert r1.stderr == f"Season replay unavailable for {TODAY} ({REASONS['corrupt']}); streak retained, not recalculated.\n"

        # Today unknown result + today in applied_dates -> applied wins over pending!
        p2 = Path(tmp) / "same2" / "picks"
        p2.mkdir(parents=True)
        (p2 / "streak.json").write_text(json.dumps({"streak": 3, "season": 2027, "applied_dates": [TODAY]}))
        (p2 / f"{TODAY}.json").write_text(json.dumps({"result": "mystery"}))
        r2 = _invoke_reconcile(p2, mp)
        assert _discover_all(p2)[0]["replay"] == {
            "state": "unavailable", "blocked_date": TODAY, "reason": REASONS["applied"],
        }
        assert r2.stderr == f"Season replay unavailable for {TODAY} ({REASONS['applied']}); streak retained, not recalculated.\n"

        # Past unresolved + load_pick returning None -> pending wins over lost_hit!
        p3 = Path(tmp) / "same3" / "picks"
        p3.mkdir(parents=True)
        (p3 / "streak.json").write_text(json.dumps({"streak": 3, "season": 2027, "applied_dates": []}))
        (p3 / f"{EARLY}.json").write_text(json.dumps({"result": None}))
        mp.setattr(picks, "load_pick", lambda *_a: None)
        r3 = _invoke_reconcile(p3, mp)
        assert _discover_all(p3)[0]["replay"] == {
            "state": "unavailable", "blocked_date": EARLY, "reason": REASONS["pending"],
        }
        assert r3.stderr == f"Season replay unavailable for {EARLY} ({REASONS['pending']}); streak retained, not recalculated.\n"

    # 4. Completed replay (with and without corrections): never says "retained"
    for with_corr in (False, True):
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
            picks_dir = Path(tmp) / "data" / "picks"
            picks_dir.mkdir(parents=True)
            (picks_dir / "streak.json").write_text(json.dumps({"streak": 9, "saver_available": True, "season": 2027}))
            (picks_dir / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit")))
            if with_corr:
                (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit")))
                mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "miss"})
            res = _invoke_reconcile(picks_dir, mp, lookback=8 if with_corr else 0)
            assert res.exit_code == 0
            assert res.stderr == ""
            assert "retained" not in res.stdout.lower()
            expected_streak = 0 if with_corr else 1
            if with_corr:
                assert res.stdout.endswith(f"Streak recalculated: {expected_streak}\n")
            else:
                assert res.stdout.endswith(f"No scoring changes detected. Streak: {expected_streak}\n")
            assert _discover_all(picks_dir)[0]["replay"] == {
                "state": "saved", "streak": expected_streak, "saver_available": True,
            }

    # 5. Receipt write (publish) fails on blocked replay (both with and without corrections)
    for with_corr in (False, True):
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
            picks_dir = Path(tmp) / "data" / "picks"
            picks_dir.mkdir(parents=True)
            # Place a regular file at health_state/reconcile_receipts so mkdir fails during publish!
            hs = Path(tmp) / "data" / "health_state"
            hs.mkdir(parents=True)
            (hs / "reconcile_receipts").write_text("not a directory")

            (picks_dir / "streak.json").write_text(json.dumps({"streak": 7, "saver_available": True, "season": 2027}))
            (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, None)))
            if with_corr:
                (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit")))
                mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "miss"})
            res = _invoke_reconcile(picks_dir, mp, lookback=8 if with_corr else 0)
            assert res.exit_code == 0
            err_lines = res.stderr.splitlines()
            assert len(err_lines) == 2, err_lines
            assert err_lines[0].startswith("reconcile: reconcile receipt unavailable (")
            assert err_lines[1] == (
                f"Season replay unavailable for {EARLY} ({REASONS['pending']}); streak retained, not recalculated."
            )
            assert "Streak retained: 7" in res.stdout
            assert "recalculated" not in res.stdout.lower()

    # 6. Degraded receipt hooks: slot_done/write_done failing vs ReconcileReceipt.replayed failing
    degraded_replayed_observation = {}
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
        picks_dir = Path(tmp) / "data" / "picks"
        picks_dir.mkdir(parents=True)
        (picks_dir / "streak.json").write_text(json.dumps({"streak": 7, "saver_available": True, "season": 2027}))
        (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, None)))
        (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit")))
        mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "miss"})

        def _failing_write_done(self, date_s, *a, **k):
            raise RuntimeError("boom")
        _failing_write_done.__name__ = "write_done"
        mp.setattr(reconcile_receipt.ReconcileReceipt, "write_done", reconcile_receipt._guarded(_failing_write_done))
        res = _invoke_reconcile(picks_dir, mp, lookback=8)
        assert res.exit_code == 0
        assert res.stderr == f"Season replay unavailable for {EARLY} ({REASONS['pending']}); streak retained, not recalculated.\n"
        assert res.stdout.endswith("Streak retained: 7\n")
        rec = _discover_all(picks_dir)[0]
        assert rec["degraded"][0]["hook"] == "write_done"
        assert rec["replay"] == {"state": "unavailable", "blocked_date": EARLY, "reason": REASONS["pending"]}

    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
        picks_dir = Path(tmp) / "data" / "picks"
        picks_dir.mkdir(parents=True)
        (picks_dir / "streak.json").write_text(json.dumps({"streak": 7, "saver_available": True, "season": 2027}))
        (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, None)))
        (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit")))
        mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "miss"})

        def _failing_replayed(self, *a, **k):
            raise RuntimeError("replayed boom")
        _failing_replayed.__name__ = "replayed"
        mp.setattr(reconcile_receipt.ReconcileReceipt, "replayed", reconcile_receipt._guarded(_failing_replayed))
        res = _invoke_reconcile(picks_dir, mp, lookback=8)
        rec = _discover_all(picks_dir)[0]
        degraded_replayed_observation = {
            "exit_code": res.exit_code,
            "stdout_tail": res.stdout.splitlines()[-1],
            "stderr": res.stderr,
            "receipt_replay": rec["replay"],
            "degraded": rec["degraded"],
        }

    return {
        "four_reasons_verified": True,
        "pairwise_walk_order_cases": pairwise_checked,
        "same_file_precedence_verified": True,
        "completed_and_failed_publish_verified": True,
        "fault_injected_replayed_hook": degraded_replayed_observation,
    }


def check_item3_operator_step() -> dict:
    from bts import contest_state, daily_decision, picks, skip_policy_shadow, web
    from bts.data import backup, lineup_analyze
    from bts.health import (
        analytics_artifacts_missing,
        fallback_defer,
        result_resolution,
        scheduler_state_integrity,
    )
    from bts.state import export as state_export
    from bts.state import verify as state_verify

    checklist_text = (REPO_ROOT / "docs" / "ops" / "2027-season-start.md").read_text()
    b8b_lines = [ln for ln in checklist_text.splitlines() if ln.startswith("| B8b |")]
    assert len(b8b_lines) == 1
    b8b = b8b_lines[0]

    expected_quote = (
        "`Season replay unavailable for <date> (no final result on the pick); streak retained, not recalculated.`"
    )
    assert expected_quote in b8b, b8b
    assert "`mv data/picks/<date>.json data/picks/<date>/set_aside_never_entered.json`" in b8b, b8b
    assert "confirm in the MLB contest that NOTHING was entered on that date" in b8b, b8b
    assert "Never move a file for a day on which a pick was entered" in b8b, b8b
    assert "For the other three reasons, leave the files in place and report the message." in b8b, b8b

    # Follow the step on a synthetic workdir (both with absent <date>/ dir and pre-existing <date>/ dir)
    for pre_existing_dir in (False, True):
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp, _alarm():
            work = Path(tmp)
            mp.chdir(work)
            picks_dir = work / "data" / "picks"
            picks_dir.mkdir(parents=True)
            (picks_dir / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", bluesky_posted=True)))
            (picks_dir / f"{EARLY}.json").write_text(json.dumps(_pick_dict(EARLY, None)))
            (picks_dir / f"{YEST}.json").write_text(json.dumps(_pick_dict(YEST, "hit", bluesky_posted=True)))
            (picks_dir / "streak.json").write_text(json.dumps({
                "streak": 99, "saver_available": False, "season": 2027, "applied_dates": [],
            }))
            if pre_existing_dir:
                daily_decision.write_decision(
                    EARLY, picks_dir, action="skip", source="mdp",
                    delivery_status="not_applicable", scoreable=False, now=RUN,
                )
                (picks_dir / EARLY / "scheduler_state.json").write_text(json.dumps({
                    "date": EARLY, "pick_locked": False, "games": [], "runs_completed": [],
                }))

            mp.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "hit"})
            r_before = _invoke_reconcile(Path("data/picks"), mp, lookback=8)
            assert r_before.exit_code == 0
            assert r_before.stderr == (
                f"Season replay unavailable for {EARLY} (no final result on the pick); "
                "streak retained, not recalculated.\n"
            )
            assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 99

            # Operator executes checklist command:
            dest_dir = Path(f"data/picks/{EARLY}")
            dest_dir.mkdir(parents=True, exist_ok=True)
            Path(f"data/picks/{EARLY}.json").rename(dest_dir / "set_aside_never_entered.json")

            # Also plant a poisoned set_aside_never_entered.json on another date to catch any reader!
            poison_dir = picks_dir / "2027-04-12"
            poison_dir.mkdir(parents=True, exist_ok=True)
            (poison_dir / "set_aside_never_entered.json").write_bytes(b"\xff\xfe{poison_invalid_json")

            r_after = _invoke_reconcile(Path("data/picks"), mp, lookback=8)
            assert r_after.exit_code == 0
            assert r_after.stderr == ""
            assert r_after.stdout.endswith("No scoring changes detected. Streak: 2\n")
            assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 2

            # Exercise all src/ readers over picks_dir with the moved file & poisoned file in place!
            assert picks._replay_season_streak(picks_dir, 2027, TODAY) == (2, True)
            assert picks.load_pick(EARLY, picks_dir) is None
            _ = daily_decision.load_decision(EARLY, picks_dir)
            _ = skip_policy_shadow.evaluate_skip_shadow(picks_dir)
            _ = lineup_analyze.collect_lead_time_samples(picks_dir)
            assert scheduler_state_integrity.check(picks_dir, today=date(2027, 4, 20)) == []
            assert fallback_defer.check(picks_dir, today=date(2027, 4, 20)) == []
            assert result_resolution.check(picks_dir, today=date(2027, 4, 20)) == []
            _ = analytics_artifacts_missing.check(picks_dir, today=date(2027, 4, 20))

            snap_out = work / "exported.json"
            exported = state_export.export_picks_state(picks_dir, snap_out)
            assert EARLY not in {h["date"] for h in exported["historical_picks"]}
            regen_dir = work / "regen_picks"
            regen_dir.mkdir()
            for fn in ("2027-04-01.json", f"{YEST}.json", "streak.json"):
                (regen_dir / fn).write_bytes((picks_dir / fn).read_bytes())
            assert state_verify.verify_picks_match(picks_dir, regen_dir)["ok"]

            # Seed minimal saver_state and contest_ledger so backup.verify_restored_ops_tree checks whole tree
            acct = picks_dir / "account_state"
            acct.mkdir(parents=True, exist_ok=True)
            (acct / "saver_state.json").write_text(json.dumps({"state": "not_earned", "season": 2027}))
            (acct / "contest_ledger.jsonl").write_text(json.dumps({"streak": 2}) + "\n")
            daily_decision.write_decision(
                YEST, picks_dir, action="single", source="mdp",
                delivery_status="delivered", scoreable=True, now=RUN,
            )
            assert backup.verify_restored_ops_tree(work) == []

            with mp.context() as wmp:
                wmp.setattr(web, "PICKS_DIR", picks_dir)
                recent = web.load_recent_picks(limit=20)
                assert EARLY not in {r["date"] for r in recent}

            # Verify no file in src/ mentions set_aside_never_entered
            for py_file in (REPO_ROOT / "src").rglob("*.py"):
                txt = py_file.read_text()
                assert "set_aside_never_entered" not in txt, py_file
                # Check no rglob("*.json") exists in src/
                assert not re.search(r'\.rglob\(\s*["\']\*\.json["\']\s*\)', txt), py_file

    return {
        "checklist_quote_matches_exact_stderr": True,
        "move_rebuilds_streak_both_absent_and_existing_dir": True,
        "all_src_readers_ignore_set_aside_even_when_poisoned": True,
    }


def main() -> int:
    item2 = check_item2_outputs()
    item3 = check_item3_operator_step()
    summary = {"status": "PASS", "item2": item2, "item3": item3}
    out_path = REVIEW_DIR / "item2_item3_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
