"""Offline adversarial CS1 probes; every loader reads real synthetic disk files.

Run from the repository root with TZ=America/New_York PYTHONDONTWRITEBYTECODE=1
.venv/bin/python .codex-review/cs1-review/probe.py. No production files are used.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path.cwd()
OUT = ROOT / ".codex-review/cs1-review"
CASES = OUT / "cases"
CASES.mkdir(exist_ok=True)
SCRATCH = OUT / "main-src"
if not SCRATCH.exists():
    shutil.copytree(ROOT / "src/bts", SCRATCH / "bts", ignore=shutil.ignore_patterns("__pycache__"))
(SCRATCH / "bts/contest_state.py").write_bytes(subprocess.check_output(
    ["git", "show", "main:src/bts/contest_state.py"], cwd=ROOT))

import bts.contest_state as head
from bts import picks
from bts.contest_fetch import build_observation
from bts.saver_state import transition_saver_state

spec = importlib.util.spec_from_file_location("bts._cs1_main", SCRATCH / "bts/contest_state.py")
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)
NOW = datetime(2026, 6, 17, 16, tzinfo=timezone.utc)
BAD = {"utf8": b"\xff", "array": b"[]", "json": b"{broken", "directory": None,
       "unreadable": b"{}"}
RESULTS = {}


def folder(name):
    return Path(tempfile.mkdtemp(prefix=name + "-", dir=CASES))


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def corrupt(path, kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    if BAD[kind] is None:
        path.mkdir()
    else:
        path.write_bytes(BAD[kind])
        if kind == "unreadable":
            path.chmod(0)


def auto(p, **extra):
    obj = build_observation({"activeStreak": 13, "seasonBestStreak": 20},
                            date(2026, 6, 17), 1, "synthetic", NOW)
    obj.update(extra)
    return put(p / "account_state/contest_streak.json", obj)


def manual(p, mode="active", **extra):
    obj = {"active_streak": 7, "best_streak": 20,
           "schema_version": "bts_contest_streak_manual_v2", "source": "manual_cli",
           "source_date": "2026-06-17", "recorded_at": NOW.isoformat()}
    if mode != "legacy":
        obj["override_expires_at"] = (NOW + timedelta(hours=1 if mode == "active" else -1)).isoformat()
    obj.update(extra)
    return put(p / "account_state/contest_streak.manual.json", obj)


def plain(obj):
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    return json.loads(json.dumps(obj, default=str))


def capture(fn):
    try:
        return {"value": plain(fn())}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def both(p, required=True):
    return {label: capture(lambda m=m: m.load_decision_streak_state(
        p, require_contest_state=required, now=NOW)) for label, m in [("main", main), ("head", head)]}


def snapshot(p):
    result = {}
    for f in p.rglob("*"):
        result[str(f.relative_to(p))] = (f.stat().st_mode, f.read_bytes() if f.is_file() else None)
    return result


def rows():
    out = {}
    for kind in BAD:
        p = folder("row1-" + kind)
        auto(p, source_date="2026-06-14")
        corrupt(p / "2026-06-16.json", kind)
        put(p / "2026-06-15.json", {"result": "miss"})
        out["row1-" + kind] = {
            label: {"resolved": capture(lambda m=m: m._resolved_pick_dates(p)),
                    "miss": capture(lambda m=m: m._has_unconfirmed_miss(p, date(2026, 6, 14))),
                    "decision": capture(lambda m=m: m.load_decision_streak_state(
                        p, require_contest_state=True, now=NOW))}
            for label, m in [("main", main), ("head", head)]}
        assert head._resolved_pick_dates(p) == [date(2026, 6, 15)]
        assert head._has_unconfirmed_miss(p, date(2026, 6, 14))
        assert head.load_decision_streak_state(p, now=NOW).status == "stale"
        if kind == "unreadable":
            assert out["row1-" + kind]["main"]["resolved"]["value"] == ["2026-06-15"]
            (p / "2026-06-16.json").chmod(0o600)
        for filename in ["contest_streak.json", "contest_streak.manual.json"]:
            p = folder("row2-" + kind)
            corrupt(p / "account_state" / filename, kind)
            out["row2-" + filename + "-" + kind] = both(p)
            assert out["row2-" + filename + "-" + kind]["head"]["error"] == "ContestStateError"
            if kind == "unreadable":
                assert out["row2-" + filename + "-" + kind]["main"]["error"] == "ContestStateError"
                (p / "account_state" / filename).chmod(0o600)
        for observation in ["manual", "auto"]:
            p = folder("row5-" + observation + "-" + kind)
            (manual if observation == "manual" else auto)(p)
            put(p / "streak.json", {"streak": 4, "saver_available": True})
            healthy = head.load_decision_streak_state(p, now=NOW)
            (p / "streak.json").unlink()
            corrupt(p / "streak.json", kind)
            before = snapshot(p) if kind != "unreadable" else None
            logs = []
            class Handler(logging.Handler):
                def emit(self, rec):
                    logs.append(rec.getMessage())
            handler = Handler()
            head.logger.addHandler(handler)
            try:
                out["row5-" + observation + "-" + kind] = both(p)
            finally:
                head.logger.removeHandler(handler)
            value = out["row5-" + observation + "-" + kind]["head"]["value"]
            assert value["model_streak"] is value["model_saver_available"] is None
            expected = plain(healthy)
            expected.update(model_streak=None, model_saver_available=None, model_state_error=value["model_state_error"])
            assert value == expected and len(logs) == 1
            assert before is None or snapshot(p) == before
            if kind == "unreadable":
                assert "PermissionError" in value["model_state_error"]
                (p / "streak.json").chmod(0o600)
        for required in [False, True]:
            p = folder("row7" if required else "row6")
            corrupt(p / "streak.json", kind)
            key = ("row7-" if required else "row6-") + kind
            out[key] = both(p, required)
            assert out[key]["head"]["error"] == "ContestStateError"
            assert ("required but missing" if required else "model state is corrupt") in out[key]["head"]["message"]
            if kind == "unreadable":
                (p / "streak.json").chmod(0o600)
    for active, best in [(13, 10), (58, 58), (13, 58)]:
        for mode in ["auto", "manual"]:
            p = folder("row3-" + mode)
            f = auto(p) if mode == "auto" else manual(p)
            valid = both(p)
            obj = json.loads(f.read_text())
            obj.update(active_streak=active, best_streak=best)
            put(f, obj)
            out[f"row3-{mode}-{active}-{best}"] = {"before_mutation": valid, "after_mutation": both(p)}
    for best in [None, False, "58", 58.0, [], {}]:
        p = folder("row3-near")
        auto(p, best_streak=best)
        out["row3-near-" + repr(best)] = both(p)
    for body in [None, 7, "x", True, {"result": []}, {"result": {}}]:
        p = folder("row1-near")
        auto(p, source_date="2026-06-14")
        put(p / "2026-06-16.json", body)
        out["row1-near-" + repr(body)] = both(p)
    for obj in [{}, {"unrelated": True}]:
        p = folder("row8")
        put(p / "streak.json", obj)
        out["row8-" + str(obj)] = both(p, False)
    RESULTS["rows"] = out


def dates_and_precedence():
    values = [None, "", "malformed", "2026-06-17", "2026-06-17T15:00:00Z", "2099-01-01T00:00:00Z"]
    out = {}
    from bts.health.contest_state import check
    for sd, asof, recorded in itertools.product(values, repeat=3):
        p = folder("metadata")
        obj = {"active_streak": 14, "best_streak": 18}
        for key, val in [("source_date", sd), ("as_of", asof), ("recorded_at", recorded)]:
            if val is not None:
                obj[key] = val
        put(p / "account_state/contest_streak.json", obj)
        res = both(p)
        res["optional"] = both(p, False)
        res["health"] = plain(check(p, expected=True, now=NOW))
        out[json.dumps([sd, asof, recorded])] = res
    RESULTS["dates"] = out
    out = {}
    for mode, bad in itertools.product(["active", "expired", "legacy", "corrupt"],
                                       ["utf8", "array", "json", "directory", "invalid", "undated", "empty_time"]):
        p = folder("precedence")
        if bad == "invalid":
            auto(p, best_streak=10)
        elif bad == "undated":
            put(p / "account_state/contest_streak.json", {"active_streak": 13})
        elif bad == "empty_time":
            put(p / "account_state/contest_streak.json", {"active_streak": 13, "recorded_at": ""})
        else:
            corrupt(p / "account_state/contest_streak.json", bad)
        if mode == "corrupt":
            corrupt(p / "account_state/contest_streak.manual.json", "utf8")
        else:
            manual(p, mode)
        out[f"{mode}/{bad}"] = {"required": both(p), "optional": both(p, False)}
        if mode == "active":
            assert both(p)["head"]["value"]["streak"] == 7
    RESULTS["precedence"] = out


def model_matrix():
    missing = object()
    values = [missing, None, False, True, 0, 4, -1, 58, "7", [], {}]
    flags = ["missing", "not_earned", "active", "used", "stale_season", "corrupt"]
    mismatches = []
    count = 0
    for observation, flag, streak, saver in itertools.product(["none", "auto", "manual"], flags, values, values):
        # Reuse a small number of directories: each complete pair is overwritten.
        p = CASES / f"matrix-{observation}-{flag}"
        p.mkdir(exist_ok=True)
        obj = {}
        if streak is not missing:
            obj["streak"] = streak
        if saver is not missing:
            obj["saver_available"] = saver
        put(p / "streak.json", obj)
        state_dir = p / "account_state"
        if observation != "none":
            (auto if observation == "auto" else manual)(p)
        if flag == "corrupt":
            put(state_dir / "saver_state.json", [])
        elif flag != "missing":
            put(state_dir / "saver_state.json", {"state": "active" if flag == "stale_season" else flag,
                "season": 2025 if flag == "stale_season" else 2026, "schema_version": "bts_saver_state_v1"})
        res = both(p, False)
        for v in res.values():
            if "value" in v:
                v["value"].pop("model_state_error", None)
        helpers = [capture(lambda f=f: f(p)) for f in
                   [picks.load_streak, picks.load_saver_available, picks.load_applied_dates]]
        # Picks module is byte-identical to main; also compare diagnostic values to its real reads.
        if res["main"] != res["head"]:
            mismatches.append({"path": str(p), "model": obj, "result": res})
        assert helpers[0] == capture(lambda: main.load_streak(p))
        assert helpers[1] == capture(lambda: main.load_saver_available(p))
        count += 1
    for observation in ["none", "auto", "manual"]:
        p = folder("missing-model-" + observation)
        if observation != "none":
            (auto if observation == "auto" else manual)(p)
        RESULTS["missing-model-" + observation] = both(p, False)
    RESULTS["model_matrix"] = {"count": count, "mismatches": mismatches,
                               "picks_diff": subprocess.check_output(["git", "diff", "main...HEAD", "--", "src/bts/picks.py"], text=True)}
    assert not mismatches


def production_paths():
    import pandas as pd
    import bts.cli as cli
    import bts.contest_fetch as fetch
    import bts.leaderboard.auth as auth
    import bts.orchestrator as orch
    import bts.scheduler as sch
    import bts.strategy as strategy
    import bts.web as web
    from bts.daily_decision import load_decision
    from bts.health.contest_state import check
    from click.testing import CliRunner

    frame = pd.DataFrame([dict(batter_id=1, batter_name="Synthetic", team="NYM", game_pk=1,
        p_game_hit=.99, pitcher_id=2, pitcher_name="Synthetic Pitcher", lineup=1,
        projected_lineup=False, flags="", game_time="2026-06-17T23:10:00Z")])

    def run_selection(p, module, required=True):
        config = {"orchestrator": {"picks_dir": str(p)}, "tiers": [],
                  "scheduler": {"contest_state_required": required}}
        loader = module.load_decision_streak_state
        with patch.object(head, "load_decision_streak_state", lambda *a, **k: loader(*a, **k, now=NOW)), \
             patch.object(orch, "run_cascade", return_value=(frame, "synthetic")), \
             patch.object(picks, "get_game_statuses_detailed", return_value={1: {"abstract": "P", "detailed": "Scheduled"}}), \
             patch.object(strategy, "_load_mdp", return_value=None):
            return orch.run_and_pick(config, "2026-06-17", require_detailed_statuses=False)[1]

    def dashboard(p, module):
        loader = module.load_decision_streak_state
        contest_loader = module.load_contest_streak_state
        with patch.object(head, "load_decision_streak_state", lambda *a, **k: loader(*a, **k, now=NOW)), \
             patch.object(head, "load_contest_streak_state", contest_loader), \
             patch.object(web, "PICKS_DIR", p), patch.object(web, "fetch_bluesky_posts", return_value=[]):
            state, error = web.load_dashboard_streak_context()
            html = web.render_page()
            (p / ("dashboard-main.html" if module is main else "dashboard-head.html")).write_text(html)
            return {"state": plain(state), "error": error, "subtitle": web._streak_subtitle(state, error),
                    "rendered": True}

    out = {}
    for name, sd, active, best in [("normal", date(2026, 6, 16), 8, 9), ("null_coverage", None, 8, 9),
                                   ("producer_above57", None, 58, 58)]:
        p = folder("fetch-" + name)
        profile = {"activeStreak": active, "seasonBestStreak": best,
                   "predictions": [] if sd is None else [{"roundId": 1, "result": "hit"}]}
        with patch.object(auth, "load_session_cookies", return_value={}), \
             patch.object(auth, "extract_uid", return_value="synthetic"), \
             patch.object(auth, "fetch_login_session", return_value=SimpleNamespace(xsid="synthetic", user_id=1, username="synthetic")), \
             patch.object(fetch, "fetch_profile", return_value=profile), \
             patch.object(cli, "_fetch_rounds", return_value={1: sd} if sd else {}):
            r = CliRunner().invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(p)])
        assert r.exit_code == 0, (r.output, r.exception)
        out[name] = {"writer": r.output, "stored": json.loads((p / "account_state/contest_streak.json").read_text()),
                     "loaders": both(p), "health": plain(check(p, expected=True, now=NOW)),
                     "selection": {label: capture(lambda m=m: run_selection(p, m)) for label, m in [("main", main), ("head", head)]},
                     "dashboards": {label: capture(lambda m=m: dashboard(p, m)) for label, m in [("main", main), ("head", head)]}}
        if name == "producer_above57":
            r = CliRunner().invoke(cli.cli, ["set-contest-streak", "--streak", "7", "--best-streak", "20",
                "--source-date", "2026-06-17", "--override-expires-at", "2026-06-17T23:59:59Z", "--picks-dir", str(p)])
            out[name]["manual_recovery"] = {"exit_code": r.exit_code, "selection": capture(lambda: run_selection(p, head))}

    for name, metadata, model in [("corrupt_model", {"source_date": "2026-06-17"}, b"\xff"),
                                  ("empty_timestamp", {"recorded_at": ""}, b"{}"),
                                  ("malformed_timestamp", {"recorded_at": "not-a-time"}, b"{}"),
                                  ("shadowed_as_of", {"source_date": "bad", "as_of": "2026-06-17"}, b"{}")]:
        p = folder("production-" + name)
        put(p / "account_state/contest_streak.json", {"active_streak": 7, "best_streak": 20, **metadata})
        (p / "streak.json").write_bytes(model)
        transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2026, source="probe")
        result = {"loaders": both(p), "selection": {label: capture(lambda m=m: run_selection(p, m))
                  for label, m in [("main", main), ("head", head)]}, "health": plain(check(p, expected=True, now=NOW))}
        with patch.object(web, "PICKS_DIR", p):
            result["dashboard"] = capture(lambda: web.load_dashboard_streak_context())
        result["dashboards"] = {label: capture(lambda m=m: dashboard(p, m)) for label, m in [("main", main), ("head", head)]}
        selected = capture(lambda: run_selection(p, head))
        if "value" in selected:
            sel = run_selection(p, head)
            assert sel.pick_result is not None
            result["dm_text"] = sch._format_pick_delivery_text(sel.pick_result.daily, sel.streak)
            state = sch.SchedulerState(date="2026-06-17", schedule_fetched_at="synthetic", games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False, pick_locked_at=None,
                result_status=None, next_wakeup=None)
            sch._commit_decision_for_pick(p, "2026-06-17", sel.pick_result.daily, selection=sel,
                                         delivery_status="delivered", state=state)
            result["decision_json"] = load_decision("2026-06-17", p)
        # Drive actual delivery handling as well; only the transport is captured.
        # For invalid timestamp cases the pick below is main's actual selection.
        daily_sel = run_selection(p, main if name != "corrupt_model" else head)
        delivery_config = {"orchestrator": {"picks_dir": str(p)},
                           "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                           "bluesky": {"dm_recipient": "synthetic.invalid"}}
        sent, failure_alerts = [], []
        loader = head.load_decision_streak_state
        delivery_state = sch.SchedulerState(date="2026-06-17", schedule_fetched_at="synthetic", games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False, pick_locked_at=None,
            result_status=None, next_wakeup=None)
        with patch.object(head, "load_decision_streak_state", lambda *a, **k: loader(*a, **k, now=NOW)), \
             patch.object(sch, "_now_et", return_value=NOW.astimezone(sch.ET)), \
             patch.object(sch, "_delivery_game_statuses", return_value={1: {"abstract": "P", "detailed": "Scheduled"}}), \
             patch.object(sch, "_trigger_live_forward_capture_on_lock"), \
             patch.object(sch, "_alert_contest_state_failure", side_effect=lambda c, e: failure_alerts.append(str(e))), \
             patch("bts.dm.send_dm", side_effect=lambda recipient, text: sent.append(text) or "synthetic-id"):
            result["head_delivery"] = capture(lambda: sch._deliver_and_lock_pick(
                daily_sel.pick_result.daily, delivery_config, p, delivery_state, "2026-06-17", "probe", selection=daily_sel))
        result["captured_dm"] = sent
        result["delivery_refusal_alerts"] = failure_alerts
        result["delivery_decision_json"] = load_decision("2026-06-17", p)
        if name in ["empty_timestamp", "malformed_timestamp"]:
            assert result["head_delivery"] == {"value": False} and not sent and failure_alerts
        else:
            assert result["head_delivery"] == {"value": True} and len(sent) == 1
        out[name] = result
    RESULTS["production"] = out


with patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK FORBIDDEN")):
    rows()
    dates_and_precedence()
    model_matrix()
    production_paths()
(OUT / "probe-results.json").write_text(json.dumps(RESULTS, indent=2, default=str) + "\n")
print(json.dumps({"rows": len(RESULTS["rows"]), "dates": len(RESULTS["dates"]),
    "precedence": len(RESULTS["precedence"]), "model_matrix": RESULTS["model_matrix"],
    "production": {k: {"main_selection": v.get("selection", {}).get("main", {}).get("error", "accepted"),
                       "head_selection": v.get("selection", {}).get("head", {}).get("error", "accepted")}
                   for k, v in RESULTS["production"].items()}}, indent=2))
