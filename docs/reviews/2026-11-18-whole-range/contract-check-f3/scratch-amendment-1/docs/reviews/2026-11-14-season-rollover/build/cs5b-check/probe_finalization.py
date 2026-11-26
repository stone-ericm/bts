import json
from pathlib import Path
from tempfile import TemporaryDirectory

from probe_run_day_v3 import DAY, MonkeyPatch, daily_pick, et, run, setup
from bts import orchestrator, picks, scheduler
from bts.daily_decision import decision_path
from tests.test_pick_chosen_on_cs5a import set_dated


def carried_skip(root, mp, keep):
    config, directory, clock, _, frame, tier = setup(root, mp, times=("19:10",))
    clock[0] = et("18:25")
    set_dated(directory, now=clock[0], streak=56, best=56)
    picks.save_pick(daily_pick(), directory)
    orchestrator.run_cascade.return_value = frame.assign(p_game_hit=0.10), tier
    first = run(config, directory, clock)
    assert first["skip"] and first["summary"]
    state = scheduler.load_state(DAY, directory)
    if keep != "candidate":
        state.final_skip_candidate = None
    if keep != "summary":
        state.skip_summary = None
    scheduler.save_state(state, directory)
    clock[0] = et("18:26")
    set_dated(directory, now=clock[0], streak=30, best=30)
    orchestrator.run_cascade.side_effect = (
        lambda *_: (frame.assign(p_game_hit=0.99), tier) if clock[0] >= et("18:35")
        else (None, None)
    )
    restarted = run(config, directory, clock)
    assert restarted["locked"] == (keep == "neither"), restarted
    print("carry", keep, json.dumps(restarted))


def missing_pick_after_commit(root, mp):
    config, directory, clock, _, frame, tier = setup(root, mp)
    clock[0] = et("18:31")
    set_dated(directory, now=clock[0], streak=30, best=30)
    orchestrator.run_cascade.return_value = frame.assign(p_game_hit=[0.99, 0.98]), tier
    first = run(config, directory, clock, stop_on=1)
    assert first["locked"] and first["slot"] == 101 and len(first["pick_dms"]) == 1
    original = decision_path(DAY, directory).read_bytes()
    (directory / f"{DAY}.json").unlink()
    clock[0] = et("18:43")
    set_dated(directory, now=clock[0], streak=31, best=31)
    orchestrator.run_cascade.return_value = frame.iloc[[1, 0]].assign(p_game_hit=[0.99, 0.98]), tier
    second = run(config, directory, clock)
    assert second["locked"] and second["slot"] == 102 and len(second["pick_dms"]) == 1
    assert decision_path(DAY, directory).read_bytes() != original
    print("missing pick after commit", json.dumps(second))


if __name__ == "__main__":
    for keep in ("candidate", "summary", "neither"):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, MonkeyPatch.context() as mp:
            carried_skip(Path(tmp), mp, keep)
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, MonkeyPatch.context() as mp:
        missing_pick_after_commit(Path(tmp), mp)
