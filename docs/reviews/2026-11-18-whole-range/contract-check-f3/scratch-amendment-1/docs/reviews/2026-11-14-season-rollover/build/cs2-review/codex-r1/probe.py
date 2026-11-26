"""Offline adversarial probes. Run once with each commit's src on PYTHONPATH.

env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.codex-review/cs2-review/offline:src .venv/bin/python \
  .codex-review/cs2-review/probe.py head
Replace src with .codex-review/cs2-review/base/src, and head with base.
All fixtures, exact bytes, reader observations and captured scorer output persist.
"""
import contextlib
import dataclasses
import inspect
import io
import json
import logging
import shutil
import sys
import tempfile
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import picks as p, scheduler as s, strategy as st
from bts.cli import cli
from bts.contest_state import load_decision_streak_state
from bts.health.streak_validation import check
from bts.reconcile_receipt import ReconcileReceipt

ET = ZoneInfo("America/New_York")
LABEL = sys.argv[1]
SEASONS = "season" in inspect.signature(p.load_streak).parameters
assert SEASONS == (LABEL == "head"), p.__file__
ROOT = Path(__file__).resolve().parent
RUN = Path(tempfile.mkdtemp(prefix=LABEL + "-", dir=ROOT / "runs"))
OUT = ROOT / (LABEL + "-probe.jsonl")
OUT.write_text("")
LEGACY = b'{"streak": 13, "saver_available": false, "updated": "2026-09-13T05:01:02.123456+00:00", "applied_dates": ["2026-09-11", "2026-09-12"]}'
MISSING = object()


def call(fn, *args, **kwargs):
    try:
        return {"value": fn(*args, **kwargs)}
    except BaseException as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def record(name, **data):
    with OUT.open("a") as f:
        f.write(json.dumps({"name": name, **data}, default=str) + "\n")


def new(name):
    d = RUN / name
    d.mkdir(parents=True)
    return d


def plant(d, tag=MISSING, **changes):
    data = json.loads(LEGACY)
    if tag is not MISSING:
        data["season"] = tag
    data.update(changes)
    (d / "streak.json").write_text(json.dumps(data))


def raw(d):
    f = d / "streak.json"
    return f.read_bytes() if f.exists() else None


def state(d):
    return json.loads(raw(d))


def plant_pick(d, day, result=None, double=False):
    pk = p.Pick(batter_name="A", batter_id=1, team="NYY", lineup_position=1,
                pitcher_name="P", pitcher_id=2, p_game_hit=.85, flags=[],
                projected_lineup=False, game_pk=111, game_time=day + "T23:10:00Z")
    dd = dataclasses.replace(pk, batter_name="B", batter_id=3, game_pk=222) if double else None
    daily = p.DailyPick(date=day, run_time=day + "T20:00:00Z", pick=pk,
                        double_down=dd, runner_up=None, result=result,
                        notification_sent=True, notification_channel="bluesky_dm", notification_id="probe")
    if result:
        daily.slot_results = {"pick": result, **({"double_down": result} if double else {})}
    p.save_pick(daily, d)
    from bts.daily_decision import write_decision
    write_decision(day, d, action="double" if double else "single", source="probe",
                   delivery_status="delivered", scoreable=True)


def cli_score(d, day, slots, now=None):
    now = now or datetime.combine(date.fromisoformat(day), datetime.min.time(), ET).replace(hour=23)
    with patch("bts.cli._now_et", return_value=now), \
         patch.object(p, "resolve_daily_slot_results", return_value=slots):
        r = CliRunner().invoke(cli, ["check-results", "--date", day, "--picks-dir", str(d),
                                    "--allow-stale-scoring", "--shadow-status-output", str(d / "shadow-status.json")])
    return {"exit": r.exit_code, "output": r.output,
            "error": type(r.exception).__name__ if r.exception else None}


def poll_score(d, day, slots, early=False):
    stream = io.StringIO()
    polls = iter(["live", "final"]) if early else None
    with patch.object(s, "_now_et", return_value=datetime.fromisoformat(day + "T23:00:00").replace(tzinfo=ET)), \
         patch.object(s, "poll_game_result", side_effect=(lambda *a: next(polls)) if early else None,
                      return_value="final"), \
         patch.object(s, "_check_hits_midgame", return_value=[True, True]), \
         patch.object(s, "_poll_interval_sleep", return_value=None), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "resolve_daily_slot_results", return_value=slots), \
         contextlib.redirect_stderr(stream):
        outcome = call(s.run_result_polling, 111, day, d)
    return {**outcome, "output": stream.getvalue()}


def reconcile(d, now):
    with patch.object(p, "resolve_daily_slot_results", side_effect=lambda daily, *a, **kw: daily.slot_results):
        return call(p.reconcile_results, d, clock=lambda: now)


def pred():
    import pandas as pd
    return pd.DataFrame([dict(batter_id=11, batter_name="Fresh", team="NYY", lineup=1,
                              pitcher_name="P", pitcher_id=2, game_pk=778899,
                              game_time="2027-03-24T23:10:00Z", p_hit_pa=.3, flags="", p_game_hit=.85)])


def readers(d, day, name):
    """Real row-6..10 entry points on disk clones; original directory untouched."""
    before = raw(d)
    md = new("readers/" + name)
    def clone(which):
        c = md / which
        c.mkdir()
        if before is not None:
            (c / "streak.json").write_bytes(before)
        return c
    c6 = clone("row6")
    plant_pick(c6, day)
    display = cli_score(c6, day, {"pick": "void"})
    c6p = clone("row6_poll")
    plant_pick(c6p, day)
    polling_display = poll_score(c6p, day, {"pick": "void"})
    c7 = clone("row7")
    # Real replay blocks on unresolved past history; summary must still select its year.
    year = int(day[:4])
    plant_pick(c7, f"{year}-01-01")
    fixed = datetime(year, 1, 15, 2, tzinfo=ET)
    real = ReconcileReceipt
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: real(clock=lambda: fixed, lookback_days=kw["lookback_days"])):
        rr = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(c7)])
    c8 = clone("row8")
    seen = []
    with patch.object(s, "predict_local_shadow", return_value=object()), \
         patch.object(s, "heartbeat_watchdog", side_effect=lambda *a, **kw: contextlib.nullcontext()), \
         patch.object(s, "_analytics_job_status", return_value={}), \
         patch.object(s, "_update_analytics_job_status", return_value=None), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(s, "select_pick", side_effect=lambda *a, **kw: (seen.append(kw["streak"]) or SimpleNamespace(pick_result=None))), \
         contextlib.redirect_stderr(io.StringIO()):
        shadow = call(s._run_shadow_prediction, {"orchestrator": {"picks_dir": str(c8)}}, day, "Prod")
    c9 = clone("row9")
    with patch.object(st, "get_game_statuses", return_value={778899: "P"}), \
         patch.object(st, "_load_mdp", return_value=None), \
         patch.object(st, "resolve_policy_decision", wraps=st.resolve_policy_decision) as decide:
        selection = call(st.select_pick, pred(), day, c9, streak=13)
        saver = decide.call_args.args[2] if decide.call_count else selection
    c10 = clone("row10")
    now = datetime.fromisoformat(day + "T23:00:00").replace(tzinfo=ET)
    decision = call(load_decision_streak_state, c10, now=now)
    if "value" in decision:
        decision["value"] = dataclasses.asdict(decision["value"])
    sd = c10 / "account_state"
    sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({"schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7, "best_streak": 7, "source": "mlb_bts_profile", "source_date": day}))
    contest = call(load_decision_streak_state, c10, now=now)
    if "value" in contest:
        contest["value"] = dataclasses.asdict(contest["value"])
    assert raw(d) == before, "reader modified original state"
    return {"row6_cli_void": display, "row6_polling": polling_display,
            "row7": {"exit": rr.exit_code, "output": rr.output},
            "row8": {"streaks": seen, **shadow}, "row9_saver": saver,
            "row10_model": decision, "row10_contest": contest}


def snap(name, d, day="2027-03-24", observe=True, **more):
    b = raw(d)
    record(name, directory=str(d), bytes_hex=b.hex() if b is not None else None,
           decoded=b.decode("utf8", errors="replace") if b is not None else None,
           health=call(check, d),
           reader_results=readers(d, day, name) if observe else None, **more)


record("source", module=p.__file__, seasons=SEASONS, directory=str(RUN))
# Rows 1/2/10/11; unusual tags include explicit JSON null (not merely absent).
for i, tag in enumerate([MISSING, 2026, 2027, True, "2027", 2027.0, None]):
    d = new(f"tags/{i}")
    plant(d, tag)
    before = raw(d)
    loaders = [p.load_streak, p.load_saver_available, p.load_applied_dates]
    record(f"row1-{i}", tag=repr(tag) if tag is MISSING else tag,
           legacy=[call(fn, d) for fn in loaders],
           seasonal=[call(fn, d, season=2027) for fn in loaders])
    assert raw(d) == before
    snap(f"tags-{i}", d)
    result = call(p.save_streak, 4, d, applied_date="2027-03-24", **({"season": 2027} if SEASONS else {}))
    snap(f"row2-{i}", d, observe=False, outcome=result)
    if SEASONS:
        data = state(d)
        assert data["season"] == 2027
        assert data["applied_dates"] == (["2026-09-11", "2026-09-12", "2027-03-24"] if type(tag) is int and tag == 2027 else ["2027-03-24"])
    plant(d, tag)
    result = call(p.save_streak, 9, d, saver_available=False)
    snap(f"row2-dateless-{i}", d, observe=False, outcome=result)
    if SEASONS and tag is not MISSING:
        assert type(state(d)["season"]) is type(tag) and state(d)["season"] == tag

# Every branch uses one persistent directory across winter and multiple scorers.
for opening in ["hit", "miss", "void", "double"]:
    for scorer in ["cli", "poll"]:
        d = new(f"sequence/{opening}-{scorer}")
        plant(d, 2026, streak=12, saver_available=False, applied_dates=[])
        plant_pick(d, "2026-12-31")
        result = cli_score(d, "2026-12-31", {"pick": "hit"})
        snap(f"seq-{opening}-{scorer}-last2026", d, day="2026-12-31", outcome=result)
        winter_before = raw(d)
        snap(f"seq-{opening}-{scorer}-winter", d, day="2027-01-15")
        assert raw(d) == winter_before
        plant_pick(d, "2027-03-24", double=(opening == "double"))
        slots = {"pick": "hit" if opening == "double" else opening}
        if opening == "double":
            slots["double_down"] = "hit"
        score = cli_score if scorer == "cli" else poll_score
        result = score(d, "2027-03-24", slots)
        snap(f"seq-{opening}-{scorer}-opening", d, outcome=result)
        current = state(d)
        assert current["streak"] == ({"hit": 1, "miss": 0, "void": 13, "double": 2}[opening] if SEASONS else {"hit": 14, "miss": 0, "void": 13, "double": 15}[opening])
        scored_before = raw(d)
        peer = poll_score if scorer == "cli" else cli_score
        result = peer(d, "2027-03-24", slots)
        snap(f"seq-{opening}-{scorer}-peer-retry", d, outcome=result)
        assert raw(d) == scored_before
        result = reconcile(d, datetime(2027, 3, 25, 2, tzinfo=ET))
        snap(f"seq-{opening}-{scorer}-0200", d, day="2027-03-25", outcome=result)
        if SEASONS:
            assert state(d)["season"] == 2027
        plant_pick(d, "2027-03-25")
        result = peer(d, "2027-03-25", {"pick": "hit"})
        snap(f"seq-{opening}-{scorer}-day2", d, day="2027-03-25", outcome=result)
        assert state(d)["streak"] == {"hit": 2, "miss": 1, "void": 1, "double": 3}[opening]

# Restart at exactly the streak/pick gap; both scorer entry paths and early hit.
class Death(BaseException):
    pass


for kind in ["cli", "poll", "early"]:
    d = new("crash/" + kind)
    plant(d, 2026, streak=13)
    plant_pick(d, "2027-03-24")
    with patch.object(p, "save_pick", side_effect=Death("between durable writes")):
        outcome = call(cli_score, d, "2027-03-24", {"pick": "hit"}) if kind == "cli" else call(poll_score, d, "2027-03-24", {"pick": "hit"}, early=(kind == "early"))
    assert p.load_pick("2027-03-24", d).result is None
    snap("crash-" + kind + "-gap", d, outcome=outcome)
    before = raw(d)
    outcome = reconcile(d, datetime(2027, 3, 24, 23, tzinfo=ET))
    assert raw(d) == before, "pp1 B1 guard lost applied hit"
    snap("crash-" + kind + "-reconcile", d, outcome=outcome)
    outcome = cli_score(d, "2027-03-24", {"pick": "hit"})
    snap("crash-" + kind + "-restart", d, outcome=outcome)
    assert state(d)["streak"] == (1 if SEASONS else 14)
    assert p.load_pick("2027-03-24", d).result == "hit"

# Real overlapping resolvers and the shared flock, on the season's first date.
d = new("race")
plant(d, 2026, streak=13)
plant_pick(d, "2027-03-24")
barrier = threading.Barrier(2)
results = []
def resolve_racing(*a, **kw):
    barrier.wait(timeout=10)
    return {"pick": "hit"}
def worker():
    results.append(call(s.run_result_polling, 111, "2027-03-24", d))
with patch.object(s, "_now_et", return_value=datetime(2027, 3, 24, 23, tzinfo=ET)), \
     patch("bts.cli._now_et", return_value=datetime(2027, 3, 24, 23, tzinfo=ET)), \
     patch.object(s, "poll_game_result", return_value="final"), \
     patch.object(p, "get_game_statuses_detailed", return_value={}), \
     patch.object(p, "resolve_daily_slot_results", side_effect=resolve_racing):
    t = threading.Thread(target=worker)
    t.start()
    cli_race = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(d),
        "--shadow-status-output", str(d / "shadow-status.json")])
    t.join(15)
    assert not t.is_alive()
    results.append({"cli_exit": cli_race.exit_code, "output": cli_race.output})
    assert cli_race.exit_code == 0, cli_race.output
    assert results[0] == {"value": "final"}, results
snap("race", d, outcome=results)
assert state(d)["streak"] == (1 if SEASONS else 14)

# Midseason saver zone, dateless between dated writes, marker and replay invariants.
for streak in range(9, 17):
    d = new(f"midseason/{streak}")
    plant(d, 2027, streak=streak, saver_available=True, applied_dates=[])
    plant_pick(d, "2027-06-10")
    outcome = cli_score(d, "2027-06-10", {"pick": "miss"})
    expected = streak if 10 <= streak <= 15 else 0
    assert state(d)["streak"] == expected
    assert state(d)["saver_available"] is (not 10 <= streak <= 15)
    before = raw(d)
    cli_score(d, "2027-06-10", {"pick": "miss"})
    assert raw(d) == before
    snap(f"midseason-{streak}", d, day="2027-06-10", observe=False, outcome=outcome)
d = new("dateless")
plant(d, 2027, streak=5, saver_available=True, applied_dates=[])
vals = [p.update_streak([True], d, date="2027-06-10"), p.update_streak([True], d),
        p.update_streak([True], d, date="2027-06-11")]
assert vals == [6, 7, 8]
snap("dateless-between-dated", d, day="2027-06-11", outcome=vals)
d = new("tagless-pp1")
plant(d, streak=14, applied_dates=["2027-03-24"])
plant_pick(d, "2027-03-24")
before = raw(d)
outcome = reconcile(d, datetime(2027, 3, 24, 23, tzinfo=ET))
assert raw(d) == before
snap("tagless-pp1", d, outcome=outcome)

# Corrupt inputs in ALL scorer branches. Preserve pre/post evidence and probe replay failure.
for i, bad in enumerate([b"{", b"\xff", b"[]", b"null"]):
    for kind in ["cli", "poll", "early", "void", "reconcile"]:
        d = new(f"corrupt/{i}-{kind}")
        (d / "streak.json").write_bytes(bad)
        plant_pick(d, "2027-06-10", result="hit" if kind == "reconcile" else None)
        # An unreadable past pick blocks later season replay; repair cannot be assumed.
        (d / "2027-04-01.json").write_text("{")
        record(f"corrupt-{i}-{kind}-before", health=call(check, d), raw=bad.hex(),
               update=call(p.load_streak, d))
        if kind == "reconcile":
            outcome = reconcile(d, datetime(2027, 6, 11, 2, tzinfo=ET))
        elif kind == "cli" or kind == "void":
            outcome = cli_score(d, "2027-06-10", {"pick": "void" if kind == "void" else "hit"})
        else:
            outcome = call(poll_score, d, "2027-06-10", {"pick": "hit"}, early=(kind == "early"))
        snap(f"corrupt-{i}-{kind}-after", d, day="2027-06-10", outcome=outcome)
        outcome = reconcile(d, datetime(2027, 6, 11, 2, tzinfo=ET))
        snap(f"corrupt-{i}-{kind}-nightly", d, day="2027-06-11", observe=False, outcome=outcome)
        if SEASONS and kind in ("cli", "poll", "early"):
            assert state(d)["streak"] == 1 and state(d)["season"] == 2027
            assert check(d) == []

# A reconstructible corrupt-file reconcile is a separate, legitimate rebuild.
d = new("corrupt-reconstructible")
plant_pick(d, "2027-06-10", result="hit")
(d / "streak.json").write_bytes(b"{")
outcome = reconcile(d, datetime(2027, 6, 11, 2, tzinfo=ET))
snap("corrupt-reconstructible", d, day="2027-06-11", outcome=outcome)
assert state(d)["streak"] == 1

# Date validation, including types; existing corrupt bytes must remain on failure.
for i, day in enumerate(["oops", "2027-02-29", "2027-03-24T00:00:00", date(2027, 3, 24),
                         "20270324", "2027-03-24\n", "2027-12-31", "2028-01-01"]):
    d = new(f"date/{i}")
    plant(d, 2027, streak=5, applied_dates=[])
    before = raw(d)
    outcome = call(p.update_streak, [True], d, date=day)
    snap(f"date-{i}", d, observe=False, day_arg=str(day), type=type(day).__name__, outcome=outcome)
    if SEASONS and i < 6:
        assert raw(d) == before

# Row 7: real command/replay, UTC and ET year disagreement with blocked replay.
for i, stamp in enumerate(["2028-01-01T01:00:00+00:00", "2028-01-01T06:00:00+00:00"]):
    d = new(f"utc/{i}")
    plant(d, 2027, streak=13)
    plant_pick(d, "2027-06-01")
    fixed = datetime.fromisoformat(stamp)
    real = ReconcileReceipt
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               side_effect=lambda **kw: real(clock=lambda: fixed, lookback_days=kw["lookback_days"])):
        r = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(d)])
    snap(f"utc-{i}", d, observe=False, outcome={"exit": r.exit_code, "output": r.output}, utc=stamp)

# Row 12: real direct consumers plus regeneration discarding season metadata.
from bts.state.export import export_initial_state
from bts.state.regenerate import Timeline, compose_state_from_snapshot_and_timeline
from bts.state.verify import diff_pick_files
from bts.data.backup import verify_restored_ops_tree
from bts import web
d = new("row12")
plant(d, 2027)
snap_path = d / "export.json"
exported = export_initial_state(d, snap_path)
regen = d / "regen"
compose_state_from_snapshot_and_timeline(snap_path, Timeline([], 0, True), regen)
with patch.object(web, "PICKS_DIR", d):
    dashboard = web.load_streak()
record("row12", health=call(check, d), export=exported, original=state(d),
       regenerated=state(regen), verify=dataclasses.asdict(diff_pick_files(d, regen)),
       backup=verify_restored_ops_tree(d), dashboard=dashboard,
       regen_fallback=call(load_decision_streak_state, regen, now=datetime(2027, 3, 24, 23, tzinfo=ET)))

print(json.dumps({"label": LABEL, "module": p.__file__, "events": len(OUT.read_text().splitlines()),
                  "output": str(OUT), "fixtures": str(RUN)}))
