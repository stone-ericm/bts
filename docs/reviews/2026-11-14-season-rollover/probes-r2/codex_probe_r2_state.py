"""Round 2: real loaders, fetch CLI, policy, health, and first-week transitions."""
import datetime as dt
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
import pandas as pd
from click.testing import CliRunner
import bts.cli as cli_module
import bts.contest_fetch as fetch
import bts.contest_state as state
import bts.leaderboard.auth as auth
import bts.strategy as strategy
from bts.health.contest_state import check
from bts.saver_state import load_saver_state
from bts.web import saver_dashboard_context
from bts.simulate.mdp import transition_outcomes
from bts.picks import update_streak

HERE = Path(__file__).resolve().parent
out = {}
base = types.ModuleType("bts._r2_base_contest")
sys.modules[base.__name__] = base
exec(compile(subprocess.check_output(["git", "show", "12dbaaa:src/bts/contest_state.py"], text=True),
             "12dbaaa:src/bts/contest_state.py", "exec"), base.__dict__)
base_health = types.ModuleType("bts.health._r2_base_contest")
sys.modules[base_health.__name__] = base_health
exec(compile(subprocess.check_output(["git", "show", "12dbaaa:src/bts/health/contest_state.py"], text=True),
             "12dbaaa:src/bts/health/contest_state.py", "exec"), base_health.__dict__)
MDP = strategy._load_mdp()


def capture(fn):
    try:
        s = fn()
        return {"streak": s.streak, "status": s.status, "saver": s.saver_available, "source": s.source}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def write_observation(picks, *, active, recorded, source=None, best=None):
    folder = picks / "account_state"
    folder.mkdir(parents=True, exist_ok=True)
    payload = fetch.build_observation({"activeStreak": active, "seasonBestStreak": active if best is None else best},
                                      source, 1, "fixture", recorded)
    (folder / "contest_streak.json").write_text(json.dumps(payload))


def fetch_cli(picks, *, active, best, recorded, predictions, rounds):
    with pytest.MonkeyPatch.context() as mp:
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return recorded.astimezone(tz) if tz else recorded.replace(tzinfo=None)
        mp.setattr(dt, "datetime", Clock)
        mp.setattr(auth, "load_session_cookies", lambda: {})
        mp.setattr(auth, "extract_uid", lambda _: "fixture")
        mp.setattr(auth, "fetch_login_session", lambda **kw: auth.AuthSession(xsid="fixture", user_id=1, username="fixture"))
        mp.setattr(fetch, "fetch_profile", lambda *a: {"activeStreak": active, "seasonBestStreak": best,
                                                       "predictions": predictions})
        mp.setattr(cli_module, "_fetch_rounds", lambda: rounds)
        r = CliRunner().invoke(cli_module.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
        assert r.exit_code == 0, repr(r.exception)
        return r.output


def action(streak, saver, day="2027-03-25", p=.75, has_diff=True):
    return strategy.resolve_policy_decision(strategy.DecisionContext(p, p-.01, has_diff, day, True, MDP),
                                             streak, saver).action


now = datetime(2027, 3, 24, 16, tzinfo=timezone.utc)
shape_cases = []
for payload in ([], None, 7, "x", {"result": []}, {"result": {}}):
    for year in (2026, 2027):
        for observation in (False, True):
            for required in (False, True):
                with tempfile.TemporaryDirectory(dir=HERE) as directory:
                    picks = Path(directory)
                    (picks / f"{year}-03-23.json").write_text(json.dumps(payload))
                    if observation:
                        write_observation(picks, active=0, recorded=now)
                    got = capture(lambda: state.load_decision_streak_state(picks, now=now, require_contest_state=required))
                    assert (got.get("error") == "ContestStateError") if required and not observation else got.get("streak") == 0
                    shape_cases.append(got)
out["r1_shapes_closed"] = len(shape_cases)

out["normal_first_week"] = []
out["failed_fetch_first_week"] = []
for mode in ("normal", "failed_fetch"):
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        picks = Path(directory)
        if mode == "failed_fetch":
            write_observation(picks, active=7, recorded=datetime(2026, 9, 26, 16, tzinfo=timezone.utc),
                              source=date(2026, 9, 26))
        for offset in range(7):
            current = now + timedelta(days=offset)
            day = current.date().isoformat()
            if mode == "normal":
                fetch_cli(picks, active=offset, best=offset, recorded=current,
                          predictions=[{"roundId": offset, "result": "hit"}] if offset else [],
                          rounds={offset: current.date()-timedelta(days=1)} if offset else {})
            pending = picks / f"{day}.json"
            pending.write_text('{"result":null}')
            optional = capture(lambda: state.load_decision_streak_state(picks, now=current))
            required = capture(lambda: state.load_decision_streak_state(picks, now=current, require_contest_state=True))
            assert optional["streak"] == offset
            assert (required.get("error") == "ContestStateError") if mode == "failed_fetch" else required["streak"] == offset
            health = [a.level for a in check(picks, expected=True, now=current)]
            out[f"{mode}_first_week"].append({"day": day, "pending": optional, "required": required, "health": health})
            update_streak([True], picks, date=day)
            pending.write_text('{"result":"hit"}')

with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    write_observation(picks, active=0, recorded=now)
    (picks / "account_state" / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 9, "source_date": "2026-09-26", "recorded_at": "2026-09-26T12:00:00Z",
        "override_expires_at": "2027-03-25T12:00:00Z"}))
    decision = state.load_decision_streak_state(picks, now=now, require_contest_state=True)
    old_alerts = base_health.check(picks, expected=True, now=now)
    new_alerts = check(picks, expected=True, now=now)
    out["unexpired_manual_health"] = {"decision": decision.streak,
        "base_alerts": [vars(a) for a in old_alerts], "head_alerts": [vars(a) for a in new_alerts]}
    assert decision.streak == 9 and old_alerts == []
    assert any(a.level == "CRITICAL" for a in new_alerts)

# A different ordinary corrupt-file encoding still reaches the same new unconditional scan.
with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    (picks / "2026-09-26.json").write_bytes(b"\xff")
    out["non_utf8_old_pick"] = {"base": capture(lambda: base.load_decision_streak_state(picks, now=now)),
                                "head": capture(lambda: state.load_decision_streak_state(picks, now=now))}
    assert out["non_utf8_old_pick"]["base"]["streak"] == 0
    assert out["non_utf8_old_pick"]["head"]["error"] == "UnicodeDecodeError"
    write_observation(picks, active=0, recorded=now)
    out["non_utf8_required_first_fetch"] = capture(lambda: state.load_decision_streak_state(picks, now=now,
                                                                                            require_contest_state=True))
    assert out["non_utf8_required_first_fetch"]["error"] == "UnicodeDecodeError"

# Last season's 7 remains in the profile while MLB's first new round is pending.
# The new rounds mapping has rolled over, so old round 1 has no date witness.
with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    predictions = [{"roundId": 1, "result": "hit"}]
    current_rounds = {2: date(2027, 3, 24)}
    out["unreset_first_week"] = []
    settled_days = []
    for offset in range(7):
        current = now + timedelta(days=offset)
        (picks / "streak.json").write_text(json.dumps({"streak": len(settled_days)}))
        fetch_cli(picks, active=7, best=7, recorded=current, predictions=predictions, rounds=current_rounds)
        decision = state.load_decision_streak_state(picks, now=current, require_contest_state=True)
        observation = json.loads((picks / "account_state" / "contest_streak.json").read_text())
        critical = [a.message for a in check(picks, expected=True, now=current) if a.level == "CRITICAL"]
        chosen = action(decision.streak, decision.saver_available, current.date().isoformat(), has_diff=offset > 0)
        out["unreset_first_week"].append({"day": current.date().isoformat(), "settled": len(settled_days),
            "stored_active": observation["active_streak"], "source_date": observation["source_date"],
            "decision": decision.streak, "status": decision.status, "critical": critical,
            "policy_action": chosen})
        if chosen != "skip":
            settled_days.append(current.date().isoformat())
            (picks / f"{current.date().isoformat()}.json").write_text('{"result":"hit"}')
    assert out["unreset_first_week"][0]["decision"] == 0
    assert out["unreset_first_week"][1]["decision"] == 7
    assert out["unreset_first_week"][1]["policy_action"] == "skip"
    assert out["unreset_first_week"][1]["critical"] == []
    assert all(row["settled"] == 1 and row["critical"] == [] for row in out["unreset_first_week"][1:])
    out["correct_day_two_action_at_streak_one"] = action(1, True)
    assert out["correct_day_two_action_at_streak_one"] != "skip"

# A healthy current-season profile proves the season has started even if local files were lost.
with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    june = datetime(2027, 6, 1, 16, tzinfo=timezone.utc)
    write_observation(picks, active=13, best=20, recorded=june, source=date(2027, 5, 31))
    (picks / "account_state" / "saver_state.json").write_text('{"season":2027,"state":"active"}')
    before = state.load_decision_streak_state(picks, now=june, require_contest_state=True)
    fetch_cli(picks, active=13, best=20, recorded=june, predictions=[{"roundId": 99, "result": "hit"}], rounds={})
    after = state.load_decision_streak_state(picks, now=june, require_contest_state=True)
    old = base.load_decision_streak_state(picks, now=june, require_contest_state=True)
    out["midseason_mapping_failure_restore"] = {
        "before": before.streak, "after": after.streak, "base_after": old.streak, "status": after.status,
        "current_saver_flag": load_saver_state(picks, season=2027).state,
        "health": [a.message for a in check(picks, expected=True, now=june)],
        "head_action": action(after.streak, after.saver_available, "2027-06-01", .75),
        "true_action": action(13, True, "2027-06-01", .75)}
    assert (before.streak, after.streak, old.streak) == (13, 0, 13)
    assert out["midseason_mapping_failure_restore"]["health"] == []
    assert out["midseason_mapping_failure_restore"]["head_action"] != out["midseason_mapping_failure_restore"]["true_action"]

with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    june = datetime(2027, 6, 1, 16, tzinfo=timezone.utc)
    write_observation(picks, active=13, best=20, recorded=june, source=date(2027, 5, 31))
    decision = state.load_decision_streak_state(picks, now=june, require_contest_state=True)
    old_decision = base.load_decision_streak_state(picks, now=june, require_contest_state=True)
    dashboard = saver_dashboard_context(picks, now=june)
    health = check(picks, expected=True, now=june)
    out["current_round_no_local_history"] = {"base_saver": old_decision.saver_available,
        "head_saver": decision.saver_available, "dashboard": dashboard.state,
        "health": [a.message for a in health],
        "head_action": action(13, decision.saver_available, "2027-06-01", .80),
        "used_action": action(13, False, "2027-06-01", .80),
        "miss_if_head_saver": [vars(x) for x in transition_outcomes("single", 13, decision.saver_available, p_hit=.80, p_both=.64)]}
    assert not old_decision.saver_available and decision.saver_available
    assert dashboard.state == "uninitialized"
    assert out["current_round_no_local_history"]["head_action"] == "single"
    assert out["current_round_no_local_history"]["used_action"] == "skip"
    # One locally settled void changes nothing about the account/saver, yet flips policy input.
    (picks / "2027-05-30.json").write_text('{"result":"void"}')
    after = state.load_decision_streak_state(picks, now=june, require_contest_state=True)
    assert after.streak == 13 and not after.saver_available
    out["current_round_no_local_history"]["saver_after_unrelated_local_void"] = after.saver_available

# Demonstrate policy changes using the shipped reach-57 and tail artifacts, not a mock table.
out["policy_differences"] = []
for day in ("2027-03-25", "2027-06-01", "2027-09-01"):
    for streak in range(16):
        for p in (.70, .75, .80, .85):
            no, yes = action(streak, False, day, p), action(streak, True, day, p)
            if no != yes:
                out["policy_differences"].append({"day": day, "streak": streak, "p": p, "used": no, "reserve": yes})

# Round-1 CLI reproduction now succeeds for JSON shapes; byte corruption still crashes it.
with tempfile.TemporaryDirectory(dir=HERE) as directory, pytest.MonkeyPatch.context() as mp:
    import io
    from urllib.parse import parse_qs, urlparse
    from tests.conftest import sample_game_feed
    from tests.test_cold_start_integration import DATE, _fake_http, _game, _live_feed, _prepare_history
    root = Path(directory)
    sample = sample_game_feed.__wrapped__()
    mp.setenv("LOKY_MAX_CPU_COUNT", "1")
    mp.setenv("BTS_USE_CALIBRATION", "0")
    mp.delenv("BTS_REFRESH_ALWAYS", raising=False)
    mp.setattr(strategy, "_mdp_cache", None)
    _prepare_history(root, sample)
    mp.chdir(root)
    picks = root / "data" / "picks"
    picks.mkdir()
    old_pick = picks / "2026-09-26.json"
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    mp.setattr(state, "datetime", Clock)
    mp.setattr(base, "datetime", Clock)
    def network(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            body = {"dates": [{"games": [_game()] if parse_qs(urlparse(url).query).get("date") == [DATE] else []}]}
        elif "/9001/feed/live" in url:
            body = _live_feed(sample, posted=True)
        else:
            raise AssertionError(f"unfaked HTTP {url}")
        return io.BytesIO(json.dumps(body).encode())
    _fake_http(mp, network)
    cmd = ["preview", "--date", DATE, "--picks-dir", str(picks)]
    old_pick.write_text("[]")
    fixed = CliRunner().invoke(cli_module.cli, cmd)
    assert fixed.exit_code == 0 and "Saved to" in fixed.output
    old_pick.write_bytes(b"\xff")
    broken = CliRunner().invoke(cli_module.cli, cmd)
    assert isinstance(broken.exception, UnicodeDecodeError)
    mp.setattr(state, "load_decision_streak_state", base.load_decision_streak_state)
    comparison = CliRunner().invoke(cli_module.cli, cmd)
    assert comparison.exit_code == 0 and "Saved to" in comparison.output
    out["preview_cli"] = {"fixed_json_shape_exit": fixed.exit_code, "bad_bytes_exit": broken.exit_code,
                            "bad_bytes_exception": repr(broken.exception), "base_bad_bytes_exit": comparison.exit_code}

(HERE / "r2-state-results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
