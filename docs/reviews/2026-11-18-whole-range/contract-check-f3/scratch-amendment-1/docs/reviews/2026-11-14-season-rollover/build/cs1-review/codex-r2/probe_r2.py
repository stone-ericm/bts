"""Round-2 offline production-path probes. Run from the repo root; no sockets.

TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  .codex-review/cs1-review/probe_r2.py > .codex-review/cs1-review/probe-extra-r2.log 2>&1
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import io
import itertools
import json
import re
import socket
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
from click.testing import CliRunner
import bts.cli as cli
import bts.contest_fetch as fetch
import bts.contest_state as head
import bts.health.contest_state as health_module
import bts.leaderboard.auth as auth
import bts.orchestrator as orchestrator
import bts.picks as picks
import bts.strategy as strategy
import bts.web as web
from bts.health.contest_state import check
from bts.saver_state import load_saver_state

ROOT = Path.cwd()
OUT = ROOT / ".codex-review/cs1-review"
CASES = OUT / "r2-cases"
CASES.mkdir(exist_ok=True)
NOW = datetime(2026, 6, 18, 16, tzinfo=timezone.utc)
RESULTS = {}


def load_source(name, source):
    path = OUT / (name + ".py")
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("bts." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main = load_source("_r2_main_state", subprocess.check_output(["git", "show", "main:src/bts/contest_state.py"]))
main_cli = load_source("_r2_main_cli", subprocess.check_output(["git", "show", "main:src/bts/cli.py"]))
old_fetch = load_source("_r2_fetch_20260606", subprocess.check_output(["git", "show", "5746144d:src/bts/contest_fetch.py"]))


def plain(v):
    if dataclasses.is_dataclass(v):
        return plain(dataclasses.asdict(v))
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    return json.loads(json.dumps(v, default=str))


def capture(fn):
    try:
        return {"value": plain(fn())}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def folder(label):
    return Path(tempfile.mkdtemp(prefix=label + "-", dir=CASES))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def setup(label, contents):
    p = folder(label)
    put(p / "streak.json", {"streak": 4, "saver_available": False})
    auto_path = p / "account_state/contest_streak.json"
    auto_path.parent.mkdir(exist_ok=True)
    if isinstance(contents, bytes):
        auto_path.write_bytes(contents)
    else:
        put(auto_path, contents)
    put(p / "account_state/saver_state.json", {"state": "active", "season": 2026})
    return p


def web_notes():
    results = {}
    for kind, contents in [("row3_bounds", {"active_streak": 13, "best_streak": 10, "source_date": "2026-06-18"}),
                           ("broken_json", b"{broken"), ("non_object", b"[]"), ("utf8", b"\xff")]:
        results[kind] = {}
        for name, module in [("main", main), ("head", head)]:
            p = setup(kind + "-" + name, contents)
            with patch.object(head, "load_contest_streak_state", module.load_contest_streak_state), \
                 patch.object(health_module, "load_contest_streak_state", module.load_contest_streak_state), \
                 patch.object(health_module, "ContestStateError", module.ContestStateError):
                health = capture(lambda: check(p, expected=False, now=NOW))
                transition = capture(lambda: web.saver_transition_response(p, expected_prior="active", new_state="used",
                    same_origin=True, now=NOW))
            # A real do_POST invocation on memory streams avoids opening a socket.
            # Reset the flag so this is independent of the helper invocation.
            put(p / "account_state/saver_state.json", {"state": "active", "season": 2026})
            body = b"expected_prior=active&new_state=used"
            handler = web.Handler.__new__(web.Handler)
            handler.path = "/saver/transition"
            handler.headers = {"Content-Length": str(len(body)), "Host": "synthetic.invalid"}
            handler.client_address = ("127.0.0.1", 12345)
            handler.rfile = io.BytesIO(body)
            handler.wfile = io.BytesIO()
            codes, headers = [], []
            handler.send_response = lambda code: codes.append(code)
            handler.send_header = lambda key, val: headers.append([key, val])
            handler.end_headers = lambda: None
            with patch.object(web, "PICKS_DIR", p), \
                 patch.object(head, "load_contest_streak_state", module.load_contest_streak_state):
                posted = capture(handler.do_POST)
            loader, contest_loader = module.load_decision_streak_state, module.load_contest_streak_state
            with patch.object(web, "PICKS_DIR", p), patch.object(web, "fetch_bluesky_posts", return_value=[]), \
                 patch.object(head, "load_decision_streak_state", lambda *a, **k: loader(*a, **k, now=NOW)), \
                 patch.object(head, "load_contest_streak_state", contest_loader):
                ctx, err = web.load_dashboard_streak_context()
                html = web.render_page()
                (p / "dashboard.html").write_text(html)
                number = ctx.streak if ctx else web.load_streak()
                subtitle = web._streak_subtitle(ctx, err)
            results[kind][name] = {"path": str(p), "transition": transition, "health_expected_false": health,
                "post": posted, "response_codes": codes, "response_headers": headers,
                "response_body": handler.wfile.getvalue().decode(), "display_number": number,
                "subtitle": subtitle, "state_error": err, "saver_state": plain(load_saver_state(p, season=2026))}
    assert results["row3_bounds"]["main"]["transition"]["value"][0] == 200
    assert results["row3_bounds"]["head"]["transition"]["error"] == "ContestStateError"
    assert results["row3_bounds"]["main"]["response_codes"] == [303]
    assert results["row3_bounds"]["head"]["response_codes"] == []
    assert results["row3_bounds"]["head"]["display_number"] == 4
    assert results["row3_bounds"]["head"]["subtitle"] == "Streak State Error"
    assert results["row3_bounds"]["main"]["health_expected_false"]["value"] == []
    assert results["row3_bounds"]["head"]["health_expected_false"]["value"][0]["level"] == "CRITICAL"
    for kind in ["broken_json", "non_object"]:
        assert results[kind]["main"]["post"]["error"] == "ContestStateError"
        assert results[kind]["head"]["post"]["error"] == "ContestStateError"
    RESULTS["abc_web_health"] = results


def fetch_and_recovery():
    results = {}
    for kind, contents in [("utf8", b"\xff"), ("non_object", b"[]"), ("broken_json", b"{broken"),
                           ("damaged_recorded_at", {"active_streak": 7, "best_streak": 20, "recorded_at": "not-a-time"})]:
        results[kind] = {}
        for name, module, commands in [("main", main, main_cli), ("head", head, cli)]:
            p = setup("fetch-" + kind + "-" + name, contents)
            before = (p / "account_state/contest_streak.json").read_bytes()
            with patch.object(auth, "load_session_cookies", return_value={}), \
                 patch.object(auth, "extract_uid", return_value="synthetic"), \
                 patch.object(auth, "fetch_login_session", return_value=SimpleNamespace(user_id=1, username="synthetic", xsid="synthetic")), \
                 patch.object(fetch, "fetch_profile", return_value={"activeStreak": 8, "seasonBestStreak": 9, "predictions": []}) as transport, \
                 patch.object(commands, "_fetch_rounds", return_value={}):
                r = CliRunner().invoke(commands.cli, ["fetch-contest-streak", "--picks-dir", str(p)])
            result = {"path": str(p), "exit_code": r.exit_code, "error": type(r.exception).__name__ if r.exception else None,
                "output": r.output, "profile_calls": transport.call_count,
                "previous_bytes_preserved": before == (p / "account_state/contest_streak.json").read_bytes()}
            if kind in ["utf8", "non_object"]:
                assert r.exit_code != 0 and transport.call_count == 0 and result["previous_bytes_preserved"]
                manual = CliRunner().invoke(commands.cli, ["set-contest-streak", "--streak", "7", "--best-streak", "20",
                    "--source-date", "2026-06-18", "--override-expires-at", "2026-06-18T23:59:59Z", "--picks-dir", str(p)])
                result["manual_exit_code"] = manual.exit_code
                result["after_manual"] = capture(lambda: module.load_decision_streak_state(p, require_contest_state=True, now=NOW))
                assert manual.exit_code == 0 and result["after_manual"]["value"]["streak"] == 7
            else:
                assert r.exit_code == 0 and transport.call_count == 1
                result["after_fetch"] = capture(lambda: module.load_decision_streak_state(p, require_contest_state=True, now=NOW))
                assert result["after_fetch"]["value"]["streak"] == 8
            results[kind][name] = result
    RESULTS["d_fetch_recovery"] = results


def null_model():
    p = folder("null-model")
    put(p / "streak.json", {"streak": None})
    out = {}
    ctx = strategy.DecisionContext(primary_p=.99, second_p=None, has_diff_game=False,
        date="2026-06-18", allow_double=True, mdp=None)
    for name, module in [("main", main), ("head", head)]:
        state = module.load_decision_streak_state(p, require_contest_state=False, now=NOW)
        out[name] = {"decision": plain(state), "policy": capture(lambda: strategy.resolve_policy_decision(ctx, state.streak, state.saver_available))}
        assert state.streak is None and out[name]["policy"]["error"] == "TypeError"
    RESULTS["e_null_model"] = out


def all_result_types():
    results = {}
    values = [None, False, True, 0, -1, 1.25, float("nan"), float("inf"), float("-inf"),
              "", "hit", "miss", "void", "not_hit", "MISS", [], {}, ["miss"], {"miss": True}, [[[]]]]
    for i, value in enumerate(values):
        p = folder("scan-" + str(i))
        put(p / "2026-06-18.json", {"result": value})
        result = {}
        for name, module in [("main", main), ("head", head)]:
            result[name] = {"resolved": capture(lambda: module._resolved_pick_dates(p)),
                            "miss": capture(lambda: module._has_unconfirmed_miss(p, date(2026, 6, 17)))}
        expected = ["2026-06-18"] if isinstance(value, str) and value in ["hit", "miss", "void"] else []
        assert result["head"]["resolved"] == {"value": expected}
        assert result["head"]["miss"] == {"value": isinstance(value, str) and value == "miss"}
        results[repr(value)] = result
    RESULTS["scan_result_types"] = results


def decoder_limits():
    limit = sys.get_int_max_str_digits()
    values = {"deep_array": b"[" * 1200 + b"0" + b"]" * 1200}
    if limit:
        values["large_integer"] = b"9" * (limit + 1)
        values["large_result_integer"] = b'{"result":' + b"9" * (limit + 1) + b"}"
    # Check the smallest repair in an untracked module, never the tracked tree.
    tree = ast.parse((ROOT / "src/bts/contest_state.py").read_text())
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in {
                "_resolved_pick_dates", "_has_unconfirmed_miss", "_load_model_diagnostics"}:
            for node in ast.walk(function):
                if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
                    node.type.elts.append(ast.Name(id="ValueError", ctx=ast.Load()))
    candidate = load_source("_r2_candidate_state", ast.unparse(ast.fix_missing_locations(tree)).encode())
    result = {}
    for kind, raw in values.items():
        p = folder("decoder-" + kind)
        put(p / "account_state/contest_streak.json", {"active_streak": 7, "best_streak": 20, "source_date": "2026-06-16"})
        (p / "2026-06-18.json").write_bytes(raw)
        result[kind] = {"path": str(p), "size": len(raw)}
        for name, module in [("main", main), ("head", head), ("candidate", candidate)]:
            result[kind][name] = {"resolved": capture(lambda: module._resolved_pick_dates(p)),
                "miss": capture(lambda: module._has_unconfirmed_miss(p, date(2026, 6, 16))),
                "decision": capture(lambda: module.load_decision_streak_state(p, require_contest_state=True, now=NOW))}
    model_results = {}
    if limit:
        raw = b"9" * (limit + 1)
        assert json.loads(raw, parse_int=lambda text: len(text)) == limit + 1
        for observation, required in itertools.product(["manual", "auto", "none"], [False, True]):
            p = folder("decoder-model-" + observation)
            (p / "streak.json").write_bytes(raw)
            if observation != "none":
                obj = {"active_streak": 7, "best_streak": 20, "source_date": "2026-06-18"}
                if observation == "manual":
                    obj["override_expires_at"] = "2026-06-18T23:59:59Z"
                put(p / ("account_state/contest_streak.manual.json" if observation == "manual" else "account_state/contest_streak.json"), obj)
            states = {name: capture(lambda m=module: m.load_decision_streak_state(p, require_contest_state=required, now=NOW))
                      for name, module in [("main", main), ("head", head), ("candidate", candidate)]}
            assert states["head"]["error"] == "ValueError"
            if observation != "none":
                assert states["candidate"]["value"]["streak"] == 7
                assert states["candidate"]["value"]["model_streak"] is None
                assert states["candidate"]["value"]["model_saver_available"] is None
            else:
                assert states["candidate"]["error"] == "ContestStateError"
                assert ("required but missing" if required else "model state is corrupt") in states["candidate"]["message"]
            if observation == "manual" and required:
                import pandas as pd
                frame = pd.DataFrame([dict(batter_id=1, batter_name="Synthetic", team="NYM", game_pk=1,
                    p_game_hit=.99, pitcher_id=2, pitcher_name="Synthetic", lineup=1,
                    flags="", game_time="2026-06-18T23:10:00Z")])
                production = {}
                for name, module in [("head", head), ("candidate", candidate)]:
                    loader = module.load_decision_streak_state
                    with patch.object(head, "load_decision_streak_state", lambda *a, **k: loader(*a, **k, now=NOW)), \
                         patch.object(orchestrator, "run_cascade", return_value=(frame, "synthetic")), \
                         patch.object(picks, "get_game_statuses_detailed", return_value={1: {"abstract": "P", "detailed": "Scheduled"}}), \
                         patch.object(strategy, "_load_mdp", return_value=None):
                        production[name] = capture(lambda: orchestrator.run_and_pick({"orchestrator": {"picks_dir": str(p)},
                            "tiers": [], "scheduler": {"contest_state_required": True}}, "2026-06-18", require_detailed_statuses=False)[1])
                assert production["head"]["error"] == "ValueError"
                assert production["candidate"]["value"]["action"] == "single"
                states["real_run_and_pick"] = production
            model_results[f"{observation}/required={required}"] = {"path": str(p), "results": states}
    RESULTS["decoder_limits"] = {"integer_limit": limit, "cases": result, "model_cases": model_results,
                                  "candidate_fix": "add ValueError to handlers in both scans and _load_model_diagnostics"}


def timestamps():
    inputs = list(itertools.product([datetime(2026, 6, 6), datetime(2026, 6, 17, 23, 59, 59),
                                    datetime(2026, 11, 15, 5, 30), datetime(2026, 12, 31, 23, 59, 59)],
        [None, timezone.utc, timezone(timedelta(hours=-4)), timezone(timedelta(hours=-5)),
         timezone(timedelta(hours=5, minutes=30)), timezone(timedelta(hours=5, minutes=45)),
         timezone(timedelta(seconds=30))], [0, 1, 123000, 123456, 999999]))
    cases = []
    p = folder("timestamps")
    for original, zone, micros in inputs:
        original = original.replace(tzinfo=zone, microsecond=micros)
        current = fetch.build_observation({"activeStreak": 8, "seasonBestStreak": 9}, None, 1, "synthetic", original)
        old = old_fetch.build_observation({"activeStreak": 8, "seasonBestStreak": 9}, date(2026, 6, 6), 1, "synthetic", original)
        assert current["recorded_at"] == old["recorded_at"]
        parsed = head._parse_dt(current["recorded_at"])
        expected = (original if original.tzinfo else original.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        assert parsed == expected
        put(p / "account_state/contest_streak.json", current)
        assert head.load_decision_streak_state(p, require_contest_state=True, now=NOW).streak == 8
        cases.append({"input": str(original), "written": current["recorded_at"], "parsed": str(parsed)})
    # Run the actual May 29 manual producer body without its Click decorators.
    source = subprocess.check_output(["git", "show", "2890b2c3:src/bts/cli.py"], text=True)
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "set_contest_streak")
    function.decorator_list = []
    namespace = {"Path": Path, "json": json, "click": click}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "historical_manual_20260529", "exec"), namespace)
    historical_p = folder("manual-20260529")
    namespace["set_contest_streak"](7, 20, None, "2026-06-18", "synthetic", None, str(historical_p))
    historical = json.loads((historical_p / "account_state/contest_streak.manual.json").read_text())
    assert head._parse_dt(historical["recorded_at"]) is not None
    current_p = folder("manual-current")
    manual = CliRunner().invoke(cli.cli, ["set-contest-streak", "--streak", "7", "--best-streak", "20", "--picks-dir", str(current_p)])
    current = json.loads((current_p / "account_state/contest_streak.manual.json").read_text())
    assert manual.exit_code == 0 and head._parse_dt(current["recorded_at"]) is not None
    RESULTS["timestamps"] = {"count": len(cases), "python": sys.version, "cases": cases,
        "historical_manual": historical, "current_manual": current,
        "web_cli_fetch_source_diff_from_main": subprocess.check_output(["git", "diff", "main...HEAD", "--", "src/bts/cli.py", "src/bts/web.py", "src/bts/contest_fetch.py"], text=True)}


with patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK FORBIDDEN")):
    web_notes()
    fetch_and_recovery()
    null_model()
    all_result_types()
    decoder_limits()
    timestamps()
(OUT / "probe-extra-results-r2.json").write_text(json.dumps(RESULTS, indent=2, default=str) + "\n")
print(json.dumps({"web_cases": len(RESULTS["abc_web_health"]), "fetch_cases": len(RESULTS["d_fetch_recovery"]),
                  "scan_types": len(RESULTS["scan_result_types"]), "timestamps": RESULTS["timestamps"]["count"]}, indent=2))
