"""Cross-version file acceptance and real readers for three finalized outcomes."""
from contextlib import ExitStack
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import bts
from bts import daily_decision, picks, scheduler, skip_policy_shadow, strategy
from bts.cli import cli
from click.testing import CliRunner
from scripts.audit.boundary_shadow_census import run_census
from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = [("2026-03-25", 50, .96, "single"), ("2026-03-26", 50, .83, "single"),
            ("2026-03-27", 56, .99, "skip")]
GUARD = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}


def clock(day):
    return walk.REAL_DATETIME.fromisoformat(day + "T12:00:00").replace(tzinfo=walk.ET)


def create(directory, day, streak, p, action):
    frame = walk._predictions(date.fromisoformat(day), p)
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    now = clock(day)
    state = scheduler.SchedulerState(date=day, schedule_fetched_at=now.isoformat(), games=[],
                                     confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                                     pick_locked_at=None, result_status=None, next_wakeup=None)
    with walk.offline(now):
        sel = strategy.select_pick(frame, day, directory, streak=streak, saver_available=False,
                                   game_statuses_detailed=statuses, require_detailed_statuses=True)
        assert sel.action == action
        if action == "skip":
            old_preview = picks.DailyPick(date=day, run_time=now.isoformat(),
                                          pick=picks.pick_from_row(frame.iloc[0]), double_down=None,
                                          runner_up=None)
            picks.save_pick(old_preview, directory)
            state.final_skip_candidate = {"primary": sel.primary_candidate, "source": "mdp",
                                          "streak": streak, "saver_available": False, "objective": "reach57"}
            scheduler._write_endofday_skip(directory, day, state)
        else:
            daily = sel.pick_result.daily
            picks.save_pick(daily, directory)
            scheduler._commit_decision_for_pick(directory, day, daily, selection=sel, state=state,
                                                delivery_status="private_locked")


def check_files(directory, day, action, guard, new=True):
    daily = picks.load_pick(day, directory)
    assert daily.pick.batter_id == 100 and daily.double_down is None
    assert (daily.policy_decision or {}).get("guard") == guard
    record = daily_decision.load_decision(day, directory)
    assert record["action"] == action and record["scoreable"] == (action != "skip")
    assert record["primary"]["batter_id"] == daily.pick.batter_id
    assert record.get("policy_guard") == guard and ("policy_guard" in record) == new
    assert daily_decision.is_scoreable_commit(day, directory, daily) == (action != "skip")
    path = daily_decision.decision_path(day, directory)
    assert daily_decision.parse_decision(path.read_text()) == record
    return path


def archived(side, directory):
    expected_src = HERE / side / "src"
    assert Path(bts.__file__).resolve().is_relative_to(expected_src)
    print(side, bts.__file__)
    for i, (day, _, _, action) in enumerate(FIXTURES):
        check_files(directory, day, action, GUARD if i == 0 else None)
    if side == "base":
        for fixture in FIXTURES[1:]:
            create(directory / "base-files", *fixture)


def census(directory):
    ledger = directory / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps({"recorded_at": f"{day}T00:00:00Z", "active_streak": s})
                                 for day, s, *_ in FIXTURES))
    first, last = FIXTURES[0][0], FIXTURES[-1][0]
    args = dict(picks_dir=directory, ledger_path=ledger, policy_path=ROOT / "data/models/mdp_policy.npz",
                saver_active=False, start=first, end=last, as_of=last, output=directory / "census.json")
    result = run_census(**args)
    assert [(r["raw_deployed"], r["parity"]) for r in result["rows"]] == [
        ("skip", "guarded_play_excluded"), ("single", "parity"), ("skip", "parity")]
    assert result["boundaries"]["primary"]["n"] == 3
    records = [daily_decision.load_decision(day, directory) for day, *_ in FIXTURES]
    assert sum(map(daily_decision.is_reach57_mdp_skip, records)) == 1
    possible_old_record = records[0] | {"action": "skip", "policy_guard": None, "scoreable": False}
    assert sum(map(daily_decision.is_reach57_mdp_skip, [possible_old_record, *records[1:]])) == 2
    print("Census raw/parity:", [(r["raw_deployed"], r["parity"]) for r in result["rows"]])
    print("Skip membership 2 → 1")


def cli_readers(directory):
    config = directory / "config.toml"
    config.write_text('[scheduler]\npick_delivery="private"\nentry_intent="research"\n')
    runner = CliRunner()
    for day, _, _, action in FIXTURES:
        with walk.offline(clock(day)), ExitStack() as stack:
            stack.enter_context(patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 70000}]))
            resolve = stack.enter_context(patch.object(picks, "resolve_daily_slot_results",
                                                       Mock(return_value={"pick": "hit"})))
            status = runner.invoke(cli, ["day-outcome", "--date", day, "--config", str(config),
                                         "--picks-dir", str(directory)])
            assert status.exit_code == 0, (status.output, status.exception)
            expected = "deliberate_skip" if action == "skip" else "research_commit"
            assert json.loads(status.output)["state"] == expected
            grade = runner.invoke(cli, ["check-results", "--date", day, "--picks-dir", str(directory),
                                        "--allow-stale-scoring"])
            assert grade.exit_code == 0, (grade.output, grade.exception)
            assert resolve.call_count == int(action != "skip")
            if action == "skip":
                assert "not scoring" in grade.output
            else:
                assert picks.load_pick(day, directory).result == "hit"
        shadow = skip_policy_shadow.record_skip_from_decision(day, directory, now=clock(day))
        assert (shadow is not None) == (action == "skip")
        print(day, "day-outcome", expected, "grading calls", resolve.call_count, "shadow", bool(shadow))


def head(directory):
    signatures = {}
    for i, fixture in enumerate(FIXTURES):
        create(directory, *fixture)
        day, _, _, action = fixture
        path = check_files(directory, day, action, GUARD if i == 0 else None)
        signatures[path] = path.read_bytes()
    for side in ("base", "rollback"):
        subprocess.run([sys.executable, __file__, side, str(directory)],
                       env=os.environ | {"PYTHONPATH": f"{HERE / side / 'src'}:{ROOT}"},
                       cwd=ROOT, timeout=15, check=True)
    for day, _, _, action in FIXTURES[1:]:
        check_files(directory / "base-files", day, action, None, new=False)
    census(directory)
    cli_readers(directory)
    assert all(path.read_bytes() == old for path, old in signatures.items())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        archived(sys.argv[1], Path(sys.argv[2]))
    else:
        with TemporaryDirectory() as temp:
            head(Path(temp))
