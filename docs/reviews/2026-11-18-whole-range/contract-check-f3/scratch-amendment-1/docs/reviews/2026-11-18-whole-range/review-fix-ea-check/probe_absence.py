"""Check the after-midnight empty-window boundary and live dated-row comparison."""

from pathlib import Path
from tempfile import TemporaryDirectory

from bts import picks
from scripts.audit.season_walk import walk

from probe_contest import WORK, at, command, observation, send_stale, stale, state


before = at("2027-07-14T23:59:55")
after = at("2027-07-15T00:00:20")
today = "2027-07-15"
with TemporaryDirectory(dir=WORK) as temp, walk.offline(before):
    path = Path(temp) / "picks"
    stale(path, before, streak=12, best=14, saver="active")
    send_stale(path, before, after)
    delivered = picks.load_pick("2027-07-14", path)
    cutoff = picks.submission_cutoff_et(delivered)
    assert after < cutoff < at(today + "T08:30")
    walk._set_clock(at(today + "T00:05"))
    # External MLB fixture: prior entry not visible and its window can still accept one.
    assert "manual: absent" in command(path, "season-contract", "status")
    print("Open prior entry window ends", cutoff.isoformat(), "so B8a rules out a dated write at 00:05.")
    walk._set_clock(at(today + "T08:30"))
    # External MLB fixture: entry absence confirmed after closure, MLB shows 13/14/active.
    assert "automatic path alone: accepted (streak 12" in command(path, "season-contract", "status")
    observation(path, today, 13)
    assert (state(path).streak, state(path).observation_kind) == (13, "dated")
    status = command(path, "season-contract", "status")
    assert "streak=13 best=14 saver=active in effect=True" in status
    assert "automatic path alone: accepted (streak 12" in status
    requests = path / "account_state/contest_manual_requests.jsonl"
    prior = requests.read_bytes()
    print("Confirmed absence allows TODAY write. MLB 13/14/active now matches dated status, so no second write.")
    assert requests.read_bytes() == prior and picks.load_pick("2027-07-14", path).notification_id == "sent-pick"
