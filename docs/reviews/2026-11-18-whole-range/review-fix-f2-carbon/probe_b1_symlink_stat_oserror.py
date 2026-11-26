"""Dedicated counterexample probe: base vs candidate on stat()-failing saver_state.json symlinks, plus unix_socket and NYE/Jan 1 EOD."""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from bts.health import alert, contest_state as health, runner

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SRC = Path(__file__).resolve().parent / "base_archive" / "src"
CANDIDATE_SRC = REPO_ROOT / "src"
ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 30, tzinfo=ET)
NEW_YEAR_0005_ET = REAL_DT(2028, 1, 1, 0, 5, 0, 0, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dated_obs(picks: Path, eff_date: dt.date, saver: str = "used", streak: int = 14) -> None:
    exp = REAL_DT.combine(eff_date, dt.time.max, ET)
    write_json(picks / "account_state" / "contest_streak.manual.json", {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": eff_date.isoformat(),
        "season": eff_date.year,
        "expires_at": exp.isoformat(),
        "active_streak": streak,
        "best_streak": max(streak, 15),
        "saver_state": saver,
        "recorded_at": exp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "cli_dated",
    })


@pytest.mark.parametrize("historical", [False, True])
def test_unix_socket_flag_preserves_collected_alerts(tmp_path, monkeypatch, historical):
    picks = tmp_path / "picks"
    write_dated_obs(picks, DAY.date(), "used", streak=14)
    resolved = [DAY.date() + dt.timedelta(days=1), DAY.date() + dt.timedelta(days=2)]
    monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: max(resolved))
    monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
    monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 2)

    baseline = health.check(picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None)
    assert len(baseline) == 1 and baseline[0].source == "contest_state"

    with tempfile.TemporaryDirectory(dir="/tmp") as short_dir:
        sock_path = Path(short_dir) / "flag.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind(str(sock_path))
            os.symlink(sock_path, picks / "account_state" / "saver_state.json")
            actual = health.check(picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None)
            assert actual == baseline
        finally:
            s.close()


def test_dec31_after_midnight_eod_c1_clock(tmp_path):
    picks = tmp_path / "picks"
    write_dated_obs(picks, dt.date(2027, 12, 31), "used", streak=14)
    write_json(picks / "account_state" / "saver_state.json", {"season": 2027, "state": "not_earned"})

    class WallClock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return NEW_YEAR_0005_ET.astimezone(tz) if tz else NEW_YEAR_0005_ET.replace(tzinfo=None)

    dms = []
    orig_safe_run = runner._safe_run
    with patch.object(runner, "datetime", WallClock), \
         patch.object(alert, "datetime", WallClock), \
         patch.object(health, "datetime", WallClock), \
         patch.object(alert, "send_dm", side_effect=lambda r, b: dms.append((r, b)) or "ok"), \
         patch.object(runner, "_safe_run", side_effect=lambda n, fn: orig_safe_run(n, fn) if n == "contest_state" else []):
        eod_alerts = runner.run_all_checks(
            picks_dir=picks,
            models_dir=tmp_path / "models",
            dm_recipient="ops.bsky.social",
            today=dt.date(2027, 12, 31),
            contest_state_expected=True,
        )
    assert len(eod_alerts) == 1 and eod_alerts[0].incident_key == INCIDENT
    assert len(dms) == 1
    reason = "The dated observation recorded the saver as used. At this check the flag for season 2027 read not_earned."
    assert reason in dms[0][1]
    assert dms[0][1].index(reason) < dms[0][1].index(" (1st observed day)") < dms[0][1].index(health.HISTORICAL_POINTER)


COMPARE_SYMLINK_SNIPPET = r"""
import datetime as dt
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

target_src = Path(sys.argv[1]).resolve()
work_dir = Path(sys.argv[2]).resolve()
symlink_kind = sys.argv[3]
historical = sys.argv[4] == "1"
repo_root = Path.cwd().resolve()

sys.path = [p for p in sys.path if Path(p or ".").resolve() != (repo_root / "src").resolve()]
sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__name__.lower()]
sys.path.insert(0, str(target_src))

import bts
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 30, tzinfo=ET)
wall = DAY + dt.timedelta(hours=3) if historical else DAY

class WallClock(REAL_DT):
    @classmethod
    def now(cls, tz=None):
        return wall.astimezone(tz) if tz else wall.replace(tzinfo=None)

picks = work_dir / "oom" / "picks"
acct = picks / "account_state"
acct.mkdir(parents=True, exist_ok=True)
exp = REAL_DT.combine(DAY.date(), dt.time.max, ET)
(acct / "contest_streak.manual.json").write_text(json.dumps({
    "schema_version": "bts_contest_streak_manual_v3",
    "effective_date": "2027-07-15",
    "season": 2027,
    "expires_at": exp.isoformat(),
    "active_streak": 14,
    "best_streak": 15,
    "saver_state": "used",
    "recorded_at": "2027-07-15T20:00:00Z",
    "source": "cli_dated",
}), encoding="utf-8")

locked_dir = None
target = acct / "saver_state.json"
if symlink_kind == "unsearchable_dir":
    locked_dir = acct / "locked_sub"
    locked_dir.mkdir()
    (locked_dir / "s.json").write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
    locked_dir.chmod(0o000)
    os.symlink(locked_dir / "s.json", target)
else:
    os.symlink("x" * 300, target)

try:
    resolved = [dt.date(2027, 7, 16), dt.date(2027, 7, 17)]
    orig_safe_run = runner._safe_run
    dms = []
    with patch.object(health, "latest_resolved_pick_date", side_effect=lambda _: max(resolved)), \
         patch.object(health, "_scan_resolved_pick_dates", side_effect=lambda *a, **kw: resolved), \
         patch.object(health, "resolved_pick_settlement_gap", side_effect=lambda *a: 2), \
         patch.object(runner, "datetime", WallClock), \
         patch.object(alert, "datetime", WallClock), \
         patch.object(health, "datetime", WallClock), \
         patch.object(alert, "send_dm", side_effect=lambda r, b: dms.append([r, b]) or "ok"), \
         patch.object(runner, "_safe_run", side_effect=lambda n, fn: orig_safe_run(n, fn) if n == "contest_state" else []):
        try:
            direct = [(a.level, a.source, a.message, a.incident_key) for a in health.check(
                picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None,
            )]
            direct_raised = None
        except Exception as exc:
            direct = None
            direct_raised = f"{type(exc).__name__}: {exc}"

        eod = [(a.level, a.source, a.message, a.incident_key) for a in runner.run_all_checks(
            picks_dir=picks,
            models_dir=work_dir / "models",
            dm_recipient="ops.bsky.social",
            today=DAY.date(),
            contest_state_expected=True,
        )]
    print(json.dumps({
        "bts_file": str(Path(bts.__file__).resolve()),
        "direct": direct,
        "direct_raised": direct_raised,
        "eod": eod,
        "dms": dms,
    }))
finally:
    if locked_dir is not None:
        locked_dir.chmod(0o700)
"""


@pytest.mark.parametrize("symlink_kind", ["unsearchable_dir", "enametoolong"])
@pytest.mark.parametrize("historical", [False, True])
def test_base_vs_candidate_on_stat_oserror_symlink(tmp_path, symlink_kind, historical):
    out = {}
    for label, src_dir in [("base", BASE_SRC), ("candidate", CANDIDATE_SRC)]:
        sub = tmp_path / label
        sub.mkdir()
        proc = subprocess.run(
            [sys.executable, "-c", COMPARE_SYMLINK_SNIPPET, str(src_dir), str(sub),
             symlink_kind, "1" if historical else "0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        out[label] = json.loads(proc.stdout)
        print(f"\n[{label} kind={symlink_kind} hist={historical}] -> {json.dumps(out[label], indent=2)}")

    # Base does not raise, preserves the collected coverage WARN, and sends 0 DMs:
    assert out["base"]["direct_raised"] is None
    assert len(out["base"]["direct"]) == 1 and out["base"]["direct"][0][0] == "WARN"
    assert out["base"]["dms"] == []

    # Contract requires candidate to match base when saver_state.json is unusable/non-regular:
    assert out["candidate"]["direct_raised"] is None
    assert out["candidate"]["direct"] == out["base"]["direct"]
    assert out["candidate"]["eod"] == out["base"]["eod"]
    assert out["candidate"]["dms"] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
