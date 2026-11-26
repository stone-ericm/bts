"""Check the dated-write boundary and status path with real scheduler and CLI calls."""
import datetime as calendar
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-e")]

from bts import cli, contest_fetch, contest_state, dm, picks, scheduler
from bts.daily_decision import load_decision
from bts.leaderboard import auth
from probe_state import invoke, observation, run_case
from probe_text import offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW, set_automatic
from tests.test_saved_pick_paths_cs5a import scheduler_state, world
from tests.test_stale_delivery_cs4c import seed_automatic


def between_selection_and_send(root, mp, timing):
    config, directory, _, _ = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    seed_automatic(directory, streak=7)
    result = run_case(config)
    selected = result["selection"]
    daily = selected.pick_result.daily
    picks.save_pick(daily, directory)
    original_send = dm.send_dm

    def file_dated():
        response = observation(DAY, directory, streak=14, best=15)
        assert response.exit_code == 0, response.output

    def send(*args, **kwargs):
        file_dated()
        return original_send(*args, **kwargs)

    if timing == "before":
        file_dated()
    else:
        mp.setattr(dm, "send_dm", send)
    scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "probe", selection=selected)
    saved = picks.load_pick(DAY, directory)
    made = saved is not None and picks.pick_was_delivered(saved)
    print(timing, "transport calls", original_send.call_count, "delivered", made)
    assert made is (timing == "during")
    assert original_send.call_count == int(made)


def status_paths(root, mp):
    _, directory, _, _ = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    set_automatic(directory, streak=7)
    unrelated = root / "operator-shell"
    unrelated.mkdir()
    set_automatic(unrelated / "data/picks", streak=14)
    mp.chdir(unrelated)
    default = invoke("season-contract", "status")
    correct = invoke("season-contract", "status", directory=directory)
    assert default.exit_code == correct.exit_code == 0
    both = [next(line for line in result.output.splitlines() if "automatic path alone:" in line)
            for result in (default, correct)]
    assert "streak 14" in both[0] and "streak 7" in both[1], both
    print("B8b comparison from another checkout", both)


def private_entry_window(root, mp):
    config, directory, _, _ = world.__wrapped__(root, mp)
    config["scheduler"].update(pick_delivery="private", entry_intent="research")
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    seed_automatic(directory, streak=7)
    selected = run_case(config)["selection"]
    daily = selected.pick_result.daily
    scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "probe", selection=selected)
    assert load_decision(DAY, directory)["delivery_status"] == "private_locked"
    assert not dm.send_dm.called
    config["scheduler"].update(pick_delivery="dm", entry_intent="enter")

    mp.setattr(auth, "load_session_cookies", lambda: {})
    mp.setattr(auth, "extract_uid", lambda _: "1")
    mp.setattr(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x"))
    mp.setattr(contest_fetch, "fetch_profile", lambda *a: {"predictions": []})
    mp.setattr(contest_fetch, "fetch_pending_predictions", lambda *a: [{"roundId": 1, "playerId": 1}])
    mp.setattr(cli, "_fetch_rounds", lambda: {1: NOW.date()})
    mp.setattr(cli, "_fetch_bts_to_mlb", lambda: {1: 123456})
    args = ("check-pick-entered", "--expected-username", "offline", "--dm-recipient", "offline.test")
    before = invoke(*args, "--now-et", f"{DAY}T18:45:00", directory=directory)
    dm_text = dm.send_dm.call_args.args[1]
    assert before.exit_code == 1 and "Fix it now!" in dm_text and daily.pick.batter_name in dm_text
    after = invoke(*args, "--now-et", f"{DAY}T19:05:00", directory=directory)
    assert after.exit_code == 0 and "outside window" in after.output and dm.send_dm.call_count == 1
    print("B8d in entry window", before.exit_code, repr(dm_text))
    print("B8d at cutoff", after.exit_code, after.output.strip())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as name:
        root = Path(name)
        for timing in ("before", "during", "status", "entry"):
            folder = root / timing
            folder.mkdir()
            with MonkeyPatch.context() as mp:
                mp.setattr(socket.socket, "connect", offline)
                mp.setattr(socket, "create_connection", offline)
                mp.setattr(socket, "getaddrinfo", offline)
                if timing == "status":
                    status_paths(folder, mp)
                elif timing == "entry":
                    private_entry_window(folder, mp)
                else:
                    between_selection_and_send(folder, mp, timing)
