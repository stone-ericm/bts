import datetime as dt
import json
import os
import shlex
import subprocess
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

from click.testing import CliRunner

from bts import cli, contest_state, web
from bts.health import contest_state as health
from bts.saver_state import load_saver_state, maybe_auto_earn_saver

from probe_support import DAY, ET, HERE, REAL_DATETIME, auto, clock, flag, manual


def render(picks, now):
    live, = health.check(picks, now=now, expected=True)
    return live.message, live.message.split("`")[1]


def invoke(command, now):
    arguments = shlex.split(command)
    assert arguments[0] == "bts"
    with clock(now):
        result = CliRunner().invoke(cli.cli, arguments[1:])
    print(now.isoformat(), result.exit_code, result.output.strip())
    return result.exit_code


with TemporaryDirectory(dir=HERE, prefix="command-") as directory:
    root = Path(directory)
    for prior in ("active", "not_earned"):
        seed = root / prior
        flag(seed, prior)
        manual(seed)
        picks = root / f"{prior} with space's"
        copytree(seed, picks)
        live, literal = render(picks, DAY)
        historical, = health.check(picks, now=DAY.replace(hour=23), historical_day=DAY.date(), expected=True)
        print(prior, live, historical.message, sep="\n")
        before = {p.name: p.read_bytes() for p in (picks / "account_state").iterdir()}
        env = os.environ.copy()
        env["PATH"] = str(HERE.parents[1] / ".venv/bin") + os.pathsep + env["PATH"]
        shell = subprocess.run(literal, shell=True, capture_output=True, text=True, env=env, timeout=4)
        print("literal shell", shell.returncode, shell.stdout.strip(), shell.stderr.strip())
        assert shell.returncode == 0
        after = {p.name: p.read_bytes() for p in (picks / "account_state").iterdir()}
        print("CLI changed", sorted(key for key, value in after.items() if before.get(key) != value))
        assert load_saver_state(picks, season=DAY.year).state == "used"
        assert not health.check(picks, now=DAY, expected=True)
        tomorrow = DAY + dt.timedelta(hours=12)
        auto(picks, tomorrow)
        maybe_auto_earn_saver(picks, best_streak=15, season=DAY.year)
        decision = contest_state.load_decision_streak_state(picks, now=tomorrow, require_contest_state=True)
        assert not decision.saver_available
        assert invoke(literal, tomorrow) == 0
        audit = picks / "account_state/saver_transitions.jsonl"
        assert [json.loads(row)["outcome"] for row in audit.read_text().splitlines()] == [
            "written", "rejected_state_mismatch",
        ]

    late = REAL_DATETIME(2027, 12, 31, 23, 59, tzinfo=ET)
    jan = late + dt.timedelta(minutes=2)
    dec_flag = root / "Dec retains old flag"
    flag(dec_flag, "active")
    manual(dec_flag, now=late)
    live, literal = render(dec_flag, late)
    print("December", live)
    assert invoke(literal, jan) == 0
    assert load_saver_state(dec_flag, season=2027).state == "used"
    assert load_saver_state(dec_flag, season=2028).state == "uninitialized"

    dec_flag = root / "Dec replaced by new season"
    flag(dec_flag, "active")
    manual(dec_flag, now=late)
    _, literal = render(dec_flag, late)
    flag(dec_flag, "active", season=jan.year)
    assert invoke(literal, jan) == 1
    assert load_saver_state(dec_flag, season=2028).state == "active"

    for scenario in ("next-day unsynchronized", "corrected", "dashboard used", "dashboard undo", "earned again"):
        picks = root / scenario
        flag(picks, "active")
        manual(picks)
        _, literal = render(picks, DAY)
        when = DAY + dt.timedelta(hours=12) if scenario in ("next-day unsynchronized", "earned again") else DAY
        if scenario == "corrected":
            manual(picks, state="active")
        if scenario.startswith("dashboard"):
            assert web.saver_transition_response(picks, expected_prior="active", new_state="used",
                                                 same_origin=True, now=DAY)[0] == 200
        if scenario == "dashboard undo":
            manual(picks, state="active")
            assert web.saver_transition_response(picks, expected_prior="used", new_state="active",
                                                 same_origin=True, now=DAY)[0] == 200
        if scenario == "earned again":
            flag(picks, "used")
            maybe_auto_earn_saver(picks, best_streak=18, season=DAY.year)
            assert load_saver_state(picks, season=DAY.year).state == "used"
            assert web.saver_transition_response(picks, expected_prior="used", new_state="active",
                                                 same_origin=True, now=when)[0] == 200
        print(scenario)
        assert invoke(literal, when) == 0
        assert load_saver_state(picks, season=DAY.year).state == "used"
