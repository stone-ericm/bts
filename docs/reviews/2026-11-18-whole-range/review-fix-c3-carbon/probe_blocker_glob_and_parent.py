"""Reproduce the _fold_season directory-discovery & decision-parent-path blocker on HEAD (c70effe9),
and verify the smallest fix control in memory without modifying any tracked file.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import socket
import stat
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import bts.cli as cli_mod
from bts.daily_decision import decision_path, parse_decision, scoreable_commit_from, write_decision
import bts.picks as picks_mod
import bts.reconcile_receipt as rr_mod
from click.testing import CliRunner


ET = ZoneInfo("America/New_York")
RUN_ET = datetime(2027, 4, 20, 2, 0, tzinfo=ET)


def _deny_net(*args, **kwargs):
    raise AssertionError("Network access is forbidden")


socket.socket = _deny_net  # type: ignore[assignment]
socket.create_connection = _deny_net  # type: ignore[assignment]


def _plant_pick(root: Path, day: str, result: str | None = None, *, delivered: bool = True) -> Path:
    slot = picks_mod.Pick(
        batter_name="Shohei Ohtani",
        batter_id=660271,
        team="LAD",
        lineup_position=1,
        pitcher_name="Gerrit Cole",
        pitcher_id=543037,
        p_game_hit=0.81,
        flags=[],
        projected_lineup=False,
        game_pk=745100,
        game_time=f"{day}T23:10:00Z",
        pitcher_team="NYY",
    )
    daily = picks_mod.DailyPick(
        date=day,
        run_time=f"{day}T18:00:00+00:00",
        pick=slot,
        double_down=None,
        runner_up=None,
        notification_sent=delivered,
        notification_id=f"dm-{day}" if delivered else None,
        delivery_attempted=delivered,
        result=result,
        slot_results={"pick": result} if result in ("hit", "miss", "void") else None,
    )
    return picks_mod.save_pick(daily, root)


def _invoke_cli(root: Path):
    real_rr = rr_mod.ReconcileReceipt
    with patch.object(picks_mod, "resolve_daily_slot_results", lambda *a, **k: {"pick": "hit"}), patch(
        "bts.reconcile_receipt.ReconcileReceipt",
        side_effect=lambda **kw: real_rr(clock=lambda: RUN_ET, lookback_days=kw["lookback_days"]),
    ):
        return CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(root)])


def _fixed_fold_season(picks_dir: Path, season: int, today: str, markers: list[str]):
    """Smallest fix control for picks._fold_season."""
    prefix = f"{season}-"
    dates: set[str] = set()
    try:
        with os.scandir(picks_dir) as it:
            for entry in it:
                name = entry.name
                if not name.startswith(prefix):
                    continue
                stem = name[:-5] if name.endswith(".json") else name
                if picks_mod._ISO_DATE_RE.fullmatch(stem):
                    dates.add(stem)
    except OSError as exc:
        raise picks_mod.NonRegularReplayFile(f"cannot scan {picks_dir}: {exc}") from exc

    streak, saver = 0, True
    for day in sorted(dates):
        if day > today:
            continue
        try:
            daily = picks_mod._read_reconcile_pick(picks_dir, day)
        except picks_mod._REPLAY_READ_ERRORS:
            return None, (day, "unreadable or invalid pick file")
        try:
            day_dir = picks_dir / day
            try:
                dir_mode = day_dir.lstat().st_mode
            except FileNotFoundError:
                text = None
            else:
                if not stat.S_ISDIR(dir_mode):
                    raise picks_mod.NonRegularReplayFile(f"{day_dir} is not a directory")
                text = picks_mod._read_reconcile_file(decision_path(day, picks_dir))
            decision = None if text is None else parse_decision(text)
            if text is not None and (decision is None or decision["date"] != day):
                raise ValueError("invalid decision record")
        except picks_mod._REPLAY_READ_ERRORS:
            return None, (day, "unreadable or invalid decision record")

        if daily is None:
            if scoreable_commit_from(decision, daily):
                return None, (day, "committed pick file missing")
        elif daily.result in picks_mod._TERMINAL_RESULTS:
            if daily.result != "void":
                streak, saver = picks_mod._apply_streak_day(
                    streak, saver, daily.result == "hit", picks_mod.streak_increment_for_resolved_hit(daily)
                )
        elif day in markers:
            return None, (day, "applied-date marker without a result")
        elif day < today:
            if scoreable_commit_from(decision, daily):
                return None, (day, "committed pick awaiting a result")
            if decision is None and daily.delivery_attempted:
                return None, (day, "delivery attempt without a decision or result")
    return (streak, saver), None


def _run_suite(label: str) -> dict:
    results = {}
    with TemporaryDirectory(dir=REVIEW_DIR, prefix=f"_blk_{label}_") as td:
        s = Path(td)

        # Case 1: chmod 000 on 2027-04-16/ with scoreable decision and missing 2027-04-16.json
        p1 = s / "c1" / "picks"
        p1.mkdir(parents=True)
        _plant_pick(p1, "2027-04-01", "hit")
        write_decision("2027-04-16", p1, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN_ET)
        picks_mod.save_streak(9, p1, saver_available=False, season=2027)
        os.chmod(p1 / "2027-04-16", 0o000)
        try:
            r1 = _invoke_cli(p1)
            st1 = json.loads((p1 / "streak.json").read_text())
            results["chmod000_decision_dir_no_root_pick"] = {
                "exit_code": r1.exit_code,
                "stdout": r1.stdout.strip(),
                "stderr": r1.stderr.strip(),
                "streak_after": [st1["streak"], st1["saver_available"]],
            }
        finally:
            os.chmod(p1 / "2027-04-16", 0o700)

        # Case 2: symlink decision dir 2027-04-16 -> external skip decision overriding a delivered pending pick
        write_decision("2027-04-16", s, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
        p2 = s / "c2" / "picks"
        p2.mkdir(parents=True)
        _plant_pick(p2, "2027-04-01", "hit")
        _plant_pick(p2, "2027-04-16", None, delivered=True)
        (p2 / "2027-04-16").symlink_to(s / "2027-04-16")
        picks_mod.save_streak(9, p2, saver_available=False, season=2027)
        r2 = _invoke_cli(p2)
        st2 = json.loads((p2 / "streak.json").read_text())
        results["symlink_decision_dir_over_delivered_pick"] = {
            "exit_code": r2.exit_code,
            "stdout": r2.stdout.strip(),
            "stderr": r2.stderr.strip(),
            "streak_after": [st2["streak"], st2["saver_available"]],
        }

        # Case 3: FIFO at 2027-04-16 (no 2027-04-16.json)
        p3 = s / "c3" / "picks"
        p3.mkdir(parents=True)
        _plant_pick(p3, "2027-04-01", "hit")
        os.mkfifo(p3 / "2027-04-16")
        picks_mod.save_streak(9, p3, saver_available=False, season=2027)
        r3 = _invoke_cli(p3)
        st3 = json.loads((p3 / "streak.json").read_text())
        results["fifo_at_date_dir_no_root_pick"] = {
            "exit_code": r3.exit_code,
            "stdout": r3.stdout.strip(),
            "stderr": r3.stderr.strip(),
            "streak_after": [st3["streak"], st3["saver_available"]],
        }

        # Case 4: chmod 0300 on picks_dir in reconcile_results library call
        p4 = s / "c4"
        p4.mkdir()
        for i in range(1, 12):
            _plant_pick(p4, f"2027-04-{i:02d}", "miss" if i == 11 else "hit")
        picks_mod.save_streak(10, p4, saver_available=False, season=2027)
        os.chmod(p4, 0o300)
        try:
            try:
                picks_mod.reconcile_results(p4, clock=lambda: RUN_ET)
                raised = None
            except Exception as exc:  # noqa: BLE001
                raised = type(exc).__name__
            st4 = json.loads((p4 / "streak.json").read_text())
            results["chmod0300_picks_dir_library_call"] = {
                "raised": raised,
                "streak_after": [st4["streak"], st4["saver_available"]],
            }
        finally:
            os.chmod(p4, 0o700)

        # Case 5: Normal healthy control (must still fold (1, True))
        p5 = s / "c5" / "picks"
        p5.mkdir(parents=True)
        _plant_pick(p5, "2027-04-01", "hit")
        _plant_pick(p5, "2027-04-16", None, delivered=False)
        write_decision("2027-04-16", p5, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
        picks_mod.save_streak(9, p5, saver_available=False, season=2027)
        r5 = _invoke_cli(p5)
        st5 = json.loads((p5 / "streak.json").read_text())
        results["healthy_skip_control"] = {
            "exit_code": r5.exit_code,
            "stderr": r5.stderr.strip(),
            "streak_after": [st5["streak"], st5["saver_available"]],
        }
    return results


def main() -> int:
    if os.geteuid() == 0:
        print("Cannot test chmod permissions as root")
        return 0
    observed_head = _run_suite("head")
    with patch.object(picks_mod, "_fold_season", _fixed_fold_season):
        fix_control = _run_suite("fix")

    payload = {"head_observed": observed_head, "fix_control": fix_control}
    (REVIEW_DIR / "blocker_glob_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
