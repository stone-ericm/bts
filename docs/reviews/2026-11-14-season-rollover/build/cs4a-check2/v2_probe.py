"""V2 contract probes of main and explicitly injected future-boundary inputs.

This does not implement the dated parser, command, or decision branch.
All transport calls are stubs. A process audit hook forbids DNS and connects.
"""
import sys


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.getnameinfo"}:
        raise RuntimeError("network forbidden: " + event)


sys.addaudithook(deny_network)
import copy
import importlib
import json
import logging
from contextlib import ExitStack
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import click
import pandas as pd
from click.testing import CliRunner
import bts.contest_state as cs
import bts.saver_state as ss
import bts.picks as picks
import bts.scheduler as sch
import bts.strategy as strategy
import bts.orchestrator as orch
import bts.web as web
from bts.daily_decision import load_decision
from bts.util import atomic_write_text

cli = importlib.import_module("bts.cli")
hc = importlib.import_module("bts.health.contest_state")
logging.disable(logging.CRITICAL)
ET = ZoneInfo("America/New_York")
NOW = datetime(2027, 7, 15, 12, tzinfo=ET)
DAY = "2027-07-15"
V3 = "bts_contest_streak_manual_v3"
OUT = {}


def folder(name):
    p = HERE / "v2-data" / name
    (p / "account_state").mkdir(parents=True, exist_ok=False)
    return p


def observation(saver="active", streak=12):
    return dict(schema_version=V3, effective_date=DAY, season=2027,
                expires_at="2027-07-16T03:59:59.999999Z", active_streak=streak,
                best_streak=streak, saver_state=saver, source="manual_cli",
                recorded_at=NOW.isoformat())


def write(p, name, obj):
    path = p / "account_state" / name
    path.write_text(json.dumps(obj))
    return path


def result(fn):
    try:
        value = fn()
        return dict(ok=True, value=asdict(value) if hasattr(value, "__dataclass_fields__") else value)
    except Exception as exc:
        return dict(ok=False, error=type(exc).__name__, message=str(exc))


# Construct exactly the SaverState row 9 asks for, without adding metadata.
OUT["constructor_and_policy"] = {
    s: {str(n): ss.policy_saver_available(ss.SaverState(state=s, season=2027), n)
        for n in (0, 9, 10, 15, 16, 57)}
    for s in ("not_earned", "active", "used", "uninitialized")}

# Parse the actual options, adding only V2's proposed option declarations.
# This tests Click's parameter sources; it is not a proposed CLI implementation.
real_command = cli.cli.get_command(click.Context(cli.cli), "set-contest-streak")
sources_cmd = click.Command("sources", params=copy.deepcopy(real_command.params) + [
    click.Option(["--date"], type=click.DateTime(formats=["%Y-%m-%d"])),
    click.Option(["--saver"], type=click.Choice(["not_earned", "active", "used"]))])
source_rows = []
for name, extra in [("default", []), ("source_date", ["--source-date", DAY]),
                    ("source", ["--source", "manual_cli"]), ("username", ["--username", ""]),
                    ("ttl_hours", ["--ttl-hours", "24"]),
                    ("override_expires_at", ["--override-expires-at", "2027-07-16T04:00:00Z"]),
                    ("saver_true", ["--saver-available"]),
                    ("saver_false", ["--saver-unavailable"])]:
    p = folder("click-" + name)
    stat_calls = []
    real_stat = click.types.os.stat

    def stat_spy(path, *args, **kwargs):
        if str(path) == str(p):
            stat_calls.append(str(path))
        return real_stat(path, *args, **kwargs)

    with patch.object(click.types.os, "stat", stat_spy):
        ctx = sources_cmd.make_context("sources", ["--streak", "12", "--best-streak", "12",
            "--date", DAY, "--saver", "active", "--picks-dir", str(p), *extra])
    source_rows.append(dict(case=name, values=ctx.params,
                           sources={n: ctx.get_parameter_source(n).name for n in ctx.params},
                           picks_dir_stat_before_callback=stat_calls))
OUT["click_parameter_sources"] = source_rows

# The actual legacy loader parses manual first and returns before even parsing bad auto.
p = folder("parse-order")
write(p, "contest_streak.manual.json", dict(active_streak=12, source_date=DAY,
    override_expires_at=(NOW + timedelta(days=1)).isoformat()))
write(p, "contest_streak.json", {})
with patch.object(cs, "_parse_state_file", wraps=cs._parse_state_file) as spy:
    selected = cs.load_contest_streak_state(p, now=NOW)
    OUT["main_parse_order"] = dict(selected=selected.path.name,
        parsed=[str(call.args[0].name) for call in spy.call_args_list])

# Health has one clock already; legacy command delegates additional clocks to saver helpers.
clock_reads = []


class TickClock(datetime):
    @classmethod
    def now(cls, tz=None):
        instant = NOW + timedelta(microseconds=len(clock_reads))
        clock_reads.append(instant.isoformat())
        return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)


p = folder("health-clock")
write(p, "contest_streak.json", dict(active_streak=7, best_streak=7, source_date=DAY))
with patch.object(hc, "datetime", TickClock), patch.object(cs, "datetime", TickClock):
    hc.check(p, now=None)
OUT["main_health_clock_reads"] = list(clock_reads)
clock_reads.clear()
p = folder("legacy-clock")
with patch("datetime.datetime", TickClock), patch.object(ss, "datetime", TickClock):
    r = CliRunner().invoke(cli.cli, ["set-contest-streak", "--picks-dir", str(p),
                                    "--streak", "7", "--best-streak", "7"])
OUT["main_legacy_clock"] = dict(exit=r.exit_code, reads=list(clock_reads),
    manual=json.loads((p / "account_state/contest_streak.manual.json").read_text()),
    flag=json.loads((p / "account_state/saver_state.json").read_text()),
    audit=json.loads((p / "account_state/saver_transitions.jsonl").read_text()))

# Primitives: a correction interrupted between log append and replace keeps old authority.
p = folder("interrupted-correction")
old = observation("used", 9)
manual = write(p, "contest_streak.manual.json", old)
request = dict(observation("active", 12), requested_at=NOW.isoformat())
log = p / "account_state/contest_manual_requests.jsonl"
with log.open("a") as f:
    f.write(json.dumps(request) + "\n")
OUT["crash_between_two_writes"] = dict(old_observation_still_present=json.loads(manual.read_text()) == old,
                                       request_log=json.loads(log.read_text()))
with patch("bts.util.os.replace", side_effect=OSError("injected replace failure")):
    failed = result(lambda: atomic_write_text(manual, json.dumps(request)))
OUT["replace_failure_preserves_old"] = dict(failure=failed, old_still_present=json.loads(manual.read_text()) == old)


def partial_append_failure():
    # A deterministic write-then-error analogue of a short append / full disk.
    with log.open("a") as f:
        f.write('{"schema_version":')
        f.flush()
        raise OSError("injected failure after a prefix")


OUT["partial_log_failure"] = result(partial_append_failure)
OUT["partial_log_failure"]["lines"] = [result(lambda line=line: json.loads(line))
                                           for line in log.read_text().splitlines()]

# Substitute only the saver input required by V2 row 11, and use main's actual message.
p = folder("health-remediation")
manual = write(p, "contest_streak.manual.json", observation("not_earned", 12))
selected = cs._parse_state_file(manual)
with patch.object(hc, "load_contest_streak_state", return_value=selected), \
     patch.object(ss, "load_saver_state", return_value=ss.SaverState("not_earned", 2027)):
    alert = next(a for a in hc.check(p, now=NOW, expected=True) if a.source == "saver_state")
before = manual.read_bytes()
r = CliRunner().invoke(cli.cli, ["saver-state", "--picks-dir", str(p), "--season", "2027",
                                "--init", "active", "--force"])
OUT["row11_message_with_required_saver_input"] = dict(message=alert.message,
    pasted_command_exit=r.exit_code, flag_after=ss.load_saver_state(p, season=2027).state,
    observation_unchanged=before == manual.read_bytes(),
    required_policy_saver_after=ss.policy_saver_available(ss.SaverState("not_earned", 2027), 12))

# Inject exactly row 9's output at its interface. Everything downstream is main's real code.
paths = []
for observation_saver, flag in (("active", "used"), ("used", "active"), ("not_earned", "used")):
    p = folder("downstream-" + observation_saver)
    write(p, "contest_streak.manual.json", observation(observation_saver))
    flag_path = write(p, "saver_state.json", dict(season=2027, state=flag))
    flag_before = flag_path.read_bytes()
    saver = ss.policy_saver_available(ss.SaverState(observation_saver, 2027), 12)
    decision = cs.DecisionStreakState(streak=12, saver_available=saver, allow_double=True,
        source="contest", status="fresh", model_streak=0, model_saver_available=True,
        contest_streak=12, contest_source_date=date(2027, 7, 15), best_streak=12,
        best_status="trusted", saver_flag_state=observation_saver)
    predictions = pd.DataFrame([dict(batter_id=1, batter_name="Probe", team="T", lineup=1,
        pitcher_name="Pitcher", pitcher_id=2, game_pk=123, game_time="2027-07-15T23:00:00Z",
        p_game_hit=.99, flags="")])
    config = dict(orchestrator=dict(picks_dir=str(p)), tiers=[],
                  scheduler=dict(pick_delivery="dm"), bluesky=dict(dm_recipient="probe"))
    with ExitStack() as stack:
        stack.enter_context(patch.object(cs, "load_decision_streak_state", return_value=decision))
        stack.enter_context(patch.object(ss, "load_saver_state", side_effect=AssertionError("flag read downstream")))
        stack.enter_context(patch.object(orch, "run_cascade", return_value=(predictions, "stub")))
        stack.enter_context(patch.object(picks, "get_game_statuses_detailed", return_value={123: {"abstract": "P", "detailed": "Scheduled"}}))
        stack.enter_context(patch.object(sch, "_void_committed_games", return_value={}))
        stack.enter_context(patch.object(sch, "_now_et", return_value=NOW))
        stack.enter_context(patch.object(sch, "_trigger_live_forward_capture_on_lock", return_value=None))
        send = stack.enter_context(patch("bts.dm.send_dm", return_value="probe-transport-stub"))
        _, selection, _ = orch.run_and_pick(config, DAY)
        assert selection.pick_result is not None, selection
        daily = selection.pick_result.daily
        state = sch.SchedulerState(date=DAY, schedule_fetched_at="probe", games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None)
        delivered = sch._deliver_and_lock_pick(daily, config, p, state, DAY, "probe", selection=selection)
    rec = load_decision(DAY, p)
    row = dict(observation_saver=observation_saver, contradictory_flag=flag,
        selected_saver=selection.saver_available, action=selection.action,
        delivered=delivered, stub_send_count=send.call_count,
        recorded_saver=rec["saver_available"], pick_policy_decision=daily.policy_decision,
        flag_unchanged=flag_path.read_bytes() == flag_before)
    # Existing recovery record intentionally lacks state if the SelectionResult is gone.
    recovery = folder("recovery-" + observation_saver)
    sch._commit_decision_for_pick(recovery, DAY, daily, selection=None, delivery_status="delivered", state=state)
    row["recovery_saver"] = load_decision(DAY, recovery)["saver_available"]
    paths.append(row)
OUT["downstream_with_injected_row9_result"] = paths

# Scoring is the model/replay stream, not a consumer of either live saver authority.
scores = []
for model_saver, observed_saver in ((True, "used"), (False, "active")):
    p = folder("scoring-" + observed_saver)
    write(p, "contest_streak.manual.json", observation(observed_saver))
    flag_path = write(p, "saver_state.json", dict(season=2027, state="used"))
    (p / "streak.json").write_text(json.dumps(dict(season=2027, streak=12, saver_available=model_saver)))
    n = picks.update_streak([False], p, date=DAY)
    scores.append(dict(observation=observed_saver, model_saver=model_saver, scored_streak=n,
        model_after=json.loads((p / "streak.json").read_text()), flag_after=json.loads(flag_path.read_text())))
OUT["real_model_scoring"] = scores
p = folder("fetch-auto-earn-and-display")
write(p, "contest_streak.manual.json", observation("used"))
write(p, "saver_state.json", dict(season=2027, state="not_earned"))
ss.maybe_auto_earn_saver(p, best_streak=12, season=2027)
OUT["real_auto_earn_and_dashboard"] = dict(flag=ss.load_saver_state(p, season=2027).state,
    dashboard=asdict(web.saver_dashboard_context(p, now=NOW)),
    required_dated_policy_saver=ss.policy_saver_available(ss.SaverState("used", 2027), 12))

(HERE / "v2-results.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n")
print(json.dumps(OUT, indent=2, default=str))
