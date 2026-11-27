import os
import subprocess
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch

from bts import picks, scheduler, strategy
from bts.daily_decision import load_decision
from scripts.audit.season_walk import walk
from rollover import clock, scenario


ROW = next(row for row in Path("docs/ops/2027-season-start.md").read_text().splitlines() if row.startswith("| B8a |"))
READ = next(piece for piece in ROW.split("`") if piece.startswith("IFS= read -r DIR"))
FILE = next(piece for piece in ROW.split("`") if piece.startswith("bts set-contest-streak --date <today>"))
STATUS = next(piece for piece in ROW.split("`") if piece.startswith("bts season-contract status --picks-dir \"${DIR"))


def run_commands(sim, parent, day, *, expect=0):
    service_cwd = parent / "service cwd"
    operator_cwd = parent / "operator cwd"
    service_cwd.mkdir(exist_ok=True)
    operator_cwd.mkdir(exist_ok=True)
    relative = os.path.relpath(sim.config["orchestrator"]["picks_dir"], service_cwd)
    absolute = (service_cwd / relative).resolve()
    assert absolute == sim.picks
    command = FILE.replace("<today>", day).replace(" N ", " 57 ").replace(" B ", " 57 ").replace(" S ", " used ")
    shell = f'{READ}\nprintf "%s\\0" {command.removeprefix("bts ")}\nprintf "%s\\0" {STATUS.removeprefix("bts ")}'
    expanded = subprocess.run(["/bin/zsh", "-f", "-c", shell], input=f"{absolute}\n",
                              capture_output=True, text=True, cwd=operator_cwd, timeout=2, check=True)
    arguments = expanded.stdout.rstrip("\0").split("\0")
    split = arguments.index("season-contract")
    assert arguments[split - 2:split] == ["--picks-dir", str(absolute)]
    assert arguments[-2:] == ["--picks-dir", str(absolute)]
    with MonkeyPatch.context() as mp:
        mp.chdir(operator_cwd)
        code, message = walk._invoke(*arguments[:split])
        assert code == expect, message
        if code == 0:
            status_code, status = walk._invoke(*arguments[split:])
            assert status_code == 0, status
            assert "live decision now: accepted" in status and "automatic path alone: refused" in status
            assert day in status
        assert not (operator_cwd / "data" / "picks").exists()
    print(f"CLI file {day} at {scheduler._now_et().isoformat()} via {absolute.name!r}: {message.strip()}", flush=True)


def late_start(parent, repair):
    start = clock("2027-06-14", 18, 15)
    day = start.date().isoformat()
    games = [start + timedelta(minutes=45), start + timedelta(minutes=105)]
    with scenario(parent, start, games) as (sim, _):
        filings = []

        def between_checks(old, new):
            if not repair or filings or not sim.dispatches:
                return
            assert not (sim.picks / f"{day}.json").exists()
            assert len(sim.dispatches) == 1 and sim.dispatches[0]["sources"] == ["contest_state"]
            walk._set_clock(old + timedelta(minutes=1))
            run_commands(sim, parent, day)
            walk._set_clock(new)
            filings.append(True)

        sim.on_sleep = between_checks
        sim.run()
        decision = load_decision(day, sim.picks)
        if repair:
            saved = picks.load_pick(day, sim.picks)
            assert len(filings) == len(sim.pick_dms) == 1
            assert saved.pick.game_pk == sim.games[1]["gamePk"]
            assert saved.chosen_on["dated"]["active_streak"] == 57
            assert decision["delivery_status"] == "delivered"
            assert not (sim.picks / "account_state" / "saver_state.json").exists()
        else:
            assert decision is None and not sim.pick_dms and not filings
        print(f"late start repair={repair}: cascades={sim.cascade.call_count} pick_dms={len(sim.pick_dms)}", flush=True)


def last_check(parent):
    start = clock("2027-06-14", 18, 15)
    day = start.date().isoformat()
    with scenario(parent, start, [start + timedelta(minutes=45)]) as (sim, _):
        sim.run()
        assert sim.cascade.call_count == 1 and not sim.pick_dms
        assert load_decision(day, sim.picks) is None and picks.load_pick(day, sim.picks) is None
        assert not (sim.picks / "account_state" / "contest_streak.manual.json").exists()
        run_commands(sim, parent, day)
        assert load_decision(day, sim.picks) is None and picks.load_pick(day, sim.picks) is None
        print("last mandatory check refused: no saved candidate, no fallback; an idle-time filing did not re-pick", flush=True)


def overnight(parent):
    start = clock("2027-06-14", 23, 58)
    day = start.date().isoformat()
    with scenario(parent, start, [start + timedelta(minutes=44)]) as (sim, _):
        sim.file_today()
        sim.seed()
        dated = sim.picks / "account_state" / "contest_streak.manual.json"
        initial = dated.read_bytes()
        sim.cascade.side_effect = [(None, None), (sim.predictions, "final-fallback")]
        sim.run()
        assert scheduler._now_et().date() != start.date() and not sim.pick_dms
        assert any(r["sources"] == ["contest_state"] for r in sim.dispatches)
        assert initial == dated.read_bytes()
        assert load_decision(day, sim.picks) is None
        assert picks.load_pick(day, sim.picks).notification_sent is False
        run_commands(sim, parent, day, expect=2)
        assert initial == dated.read_bytes()
        print("after-midnight prior slate pending: no filing for it; old-date negative control rejected; unchanged bytes", flush=True)


if __name__ == "__main__":
    assert strategy._load_mdp()["tail"] is not None
    for test in (lambda p: late_start(p, False), lambda p: late_start(p, True), last_check, overnight):
        with TemporaryDirectory() as tmp:
            test(Path(tmp).resolve())
