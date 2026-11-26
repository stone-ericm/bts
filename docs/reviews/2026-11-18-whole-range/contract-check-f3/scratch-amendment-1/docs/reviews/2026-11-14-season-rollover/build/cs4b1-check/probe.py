"""Contract review of main; future-contract deductions are kept out of these results."""
import sys
sys.dont_write_bytecode = True

def deny_network(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                 "socket.getnameinfo", "socket.connect", "socket.connect_ex", "socket.sendto"}:
        raise RuntimeError("audit: network forbidden: " + event)

sys.addaudithook(deny_network)
import ast
import contextlib
import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from click.testing import CliRunner
import httpx
import bts.cli as cli
import bts.contest_fetch as cf
import bts.contest_state as cs
import bts.leaderboard.auth as auth
import bts.saver_state as saver

spec = importlib.util.spec_from_file_location("earlier_contract_probe", REPO / "docs/reviews/2026-11-14-season-rollover/build/cs4b-check/probe.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.ROOT = HERE / "reused"
old.ROOT.mkdir(parents=True, exist_ok=True)
ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
BASE_DATETIME = dt.datetime
OUT = {}

def captured(name, profile, rounds, *, round_error=None, source_tick=False, auth_crosses=False, dry=False):
    picks = HERE / "cases" / name / "picks"
    picks.mkdir(parents=True, exist_ok=True)
    events, clock, earned, dms = [], [], [], []
    moments = ([BASE_DATETIME(2027, 12, 31, 23, 59, 59, 900000, tzinfo=ET),
                BASE_DATETIME(2028, 1, 1, 0, 0, 1, tzinfo=ET)] if source_tick or auth_crosses else [NOW])
    phase = [0]

    class Clock(BASE_DATETIME):
        @classmethod
        def now(cls, tz=None):
            index = phase[0] if auth_crosses else min(len(clock), len(moments) - 1)
            value = moments[index]
            clock.append(value.isoformat())
            events.append("clock")
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    def cookies():
        events.append("auth")
        if auth_crosses:
            phase[0] = 1
        return {}

    def fetched(*args):
        events.append("profile")
        return profile

    def mapped():
        events.append("rounds")
        if round_error:
            raise round_error
        return rounds

    atomic = cli._atomic_write_json
    opened = Path.open
    def write(path, data):
        events.append("write:" + path.name)
        return atomic(path, data)

    def opening(path, mode="r", *a, **kw):
        if path.name == "contest_ledger.jsonl" and mode == "a":
            events.append("ledger:append")
        return opened(path, mode, *a, **kw)

    def earn(*args, **kw):
        events.append("saver")
        earned.append(kw)

    with contextlib.ExitStack() as stack:
        for p in [patch.object(dt, "datetime", Clock),
                  patch.object(auth, "load_session_cookies", cookies),
                  patch.object(auth, "extract_uid", lambda _: "u"),
                  patch.object(auth, "fetch_login_session", lambda **kw: NS(user_id=1, username="offline", xsid="x")),
                  patch.object(cf, "fetch_profile", fetched), patch.object(cli, "_fetch_rounds", mapped),
                  patch.object(cli, "_atomic_write_json", write), patch.object(Path, "open", opening),
                  patch.object(saver, "maybe_auto_earn_saver", earn),
                  patch("bts.dm.send_dm", lambda *a, **kw: dms.append(a))]:
            stack.enter_context(p)
        result = CliRunner().invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks),
            "--dm-recipient", "offline.test"] + (["--dry-run"] if dry else []))
    return dict(exit=result.exit_code, output=result.output, exception=repr(result.exception), events=events,
                clock=clock, saver_calls=earned, dms=dms,
                files={str(p.relative_to(picks.parent)): p.read_text() for p in picks.parent.rglob("*") if p.is_file()})

def main():
    OUT["commands"] = {command: dict(exit=(r := CliRunner().invoke(cli.cli, ["season-contract", command])).exit_code,
                                    output=r.output) for command in ["status", "enable"]}
    old.matrix()
    old.consumers()
    OUT["reused_matrix"] = old.OUT["matrix"]
    OUT["reused_consumers"] = old.OUT["ledger_consumers"]
    OUT["reused_fetch"] = {name: old.fetch(name, **kw) for name, kw in [
        ("success", {}), ("dry_success", dict(dry=True)),
        ("dry_identity", dict(dry=True, prior=dict(old.auto(), username="other"))),
        ("best58", dict(best=58)), ("prior_nonobject", dict(prior=[])),
        ("prior_nonutf8", dict(prior=b"\xff")),
        ("observation_fail", dict(fail_at="contest_streak.json")),
        ("status_fail", dict(fail_at="contest_streak_fetch_status.json")),
        ("ledger_fail", dict(fail_at="ledger")), ("saver_fail", dict(fail_at="saver")),
    ]}
    profile = dict(activeStreak=7, seasonBestStreak=9, predictions=[dict(roundId=1, result="hit")])
    mapped = {1: dt.date(2027, 7, 14)}
    variants = [
        ("valid", profile, mapped, {}),
        ("missing_predictions", {k:v for k,v in profile.items() if k != "predictions"}, mapped, {}),
        ("null_predictions", dict(profile, predictions=None), mapped, {}),
        ("object_predictions", dict(profile, predictions={}), mapped, {}),
        ("nonobject_prediction", dict(profile, predictions=[1]), mapped, {}),
        ("empty_predictions", dict(profile, predictions=[]), mapped, {}),
        ("empty_rounds", profile, {}, {}), ("best57", dict(profile, seasonBestStreak=57), mapped, {}),
        ("best58", dict(profile, seasonBestStreak=58), mapped, {}),
        ("profile_null", None, mapped, {}),
        ("profile_null_rounds503", None, mapped, dict(round_error=httpx.HTTPStatusError("rounds503", request=httpx.Request("GET", "https://offline.invalid"), response=httpx.Response(503)))),
        ("predictions_null_rounds503", dict(profile, predictions=None), mapped, dict(round_error=httpx.HTTPStatusError("rounds503", request=httpx.Request("GET", "https://offline.invalid"), response=httpx.Response(503)))),
        ("bad_round_best58", dict(profile, seasonBestStreak=58, predictions=[dict(roundId="bad", result="hit")]), mapped, {}),
        ("unhashable_result", dict(profile, predictions=[dict(result=[])]), mapped, {}),
        ("year_tick_dec_source", profile, {1: dt.date(2027, 12, 31)}, dict(source_tick=True)),
        ("year_tick_jan_source", profile, {1: dt.date(2028, 1, 1)}, dict(source_tick=True)),
        ("auth_crosses_midnight", profile, {1: dt.date(2027, 12, 31)}, dict(auth_crosses=True)),
        ("undated_source", dict(profile, predictions=[dict(roundId=1, result=None)]), mapped, {}),
    ]
    OUT["captured"] = {name: captured(name, p, rounds, **kw) for name, p, rounds, kw in variants}
    OUT["direct_parser"] = []
    for filename in ["contest_streak.json", "contest_streak.manual.json"]:
        for label, fields in [
            ("v1", {}), ("v2_valid", dict(schema_version="bts_contest_streak_auto_v2", season=2027, activation_id="a"*32)),
            ("v2_bad", dict(schema_version="bts_contest_streak_auto_v2", season=True, activation_id="BAD")),
            ("v2_missing", dict(schema_version="bts_contest_streak_auto_v2")),
            ("unknown", dict(schema_version="other")), ("null_schema", dict(schema_version=None)),
        ]:
            path = HERE / "parser" / filename / label / filename
            old.write(path, dict(old.auto(), **fields))
            state = cs._parse_state_file(path, require_dated=True)
            OUT["direct_parser"].append(dict(file=filename, kind=label, streak=state.streak,
                best=cs.classify_best_streak(state, now=NOW, now_year=2027)))
    p = HERE / "filesystem"
    p.mkdir(exist_ok=True)
    (p / "dangling").symlink_to(p / "missing")
    OUT["dangling"] = dict(exists=(p / "dangling").exists(), lstat=bool((p / "dangling").lstat()))
    # Extract the exact real P05 ledger-consumption statements, without loading model artifacts.
    script = ast.parse((REPO / "scripts/audit/p05_tail_policy_audit.py").read_text())
    main_node = next(n for n in script.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    consume_nodes = [n for n in main_node.body if
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"ledger_last", "contest"} for t in n.targets))
        or (isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "line")][:3]
    consume_code = compile(ast.Module(body=consume_nodes, type_ignores=[]), "real-p05-ledger-fragment", "exec")
    projected = []
    picks = HERE / "p05"
    ledger = picks / "account_state/contest_ledger.jsonl"
    row = dict(recorded_at=NOW.isoformat(), active_streak=7, best_streak=9, source_date="2027-07-14",
               predictions=[dict(roundId=1, result="hit", streak=7)])
    for extra in ({}, dict(season=2027, switched=False, activation_id=None, settled_by_year={"2027":1}, settled_unmapped=0, unsettled=0)):
        old.write(ledger, dict(row, **extra))
        env = dict(picks=picks, json=json)
        exec(consume_code, env)
        projected.append(env["contest"])
    OUT["p05_projection_equal"] = projected[0] == projected[1]
    OUT["fetch_test_inventory"] = []
    for path in (REPO / "tests").rglob("*.py"):
        body = path.read_text()
        if not any(x in body for x in ["fetch-contest-streak", "build_observation"]):
            continue
        tree = ast.parse(body)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                text = ast.get_source_segment(body, node)
                if any(x in text for x in ["fetch-contest-streak", "_run_fetch", "_observe(", "build_observation"]):
                    OUT["fetch_test_inventory"].append(dict(file=str(path.relative_to(REPO)), line=node.lineno, name=node.name))
    target = HERE / "results.json"
    target.write_text(json.dumps(OUT, indent=2, default=str) + "\n")
    print(json.dumps(dict(result=str(target), matrix_cells=len(OUT["reused_matrix"]), matrix_calls=len(OUT["reused_matrix"])*5,
                         fetch_cases=len(OUT["reused_fetch"])+len(OUT["captured"]), parser_cases=len(OUT["direct_parser"]))))
    for k, v in OUT["captured"].items():
        print(k, v["exit"], v["events"], v["output"].strip())

if __name__ == "__main__":
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        main()
    (HERE / "probe-stderr.log").write_text(log.getvalue())
