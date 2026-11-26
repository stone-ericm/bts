import builtins
from contextlib import contextmanager
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from review_runner import BASE, REVIEW, ROOT, run

ET = ZoneInfo("America/New_York")


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def reset(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


@contextmanager
def workdir(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def contest(root, tree):
    from bts.health.contest_state import check
    from tests.conftest import covered_automatic
    from tests.test_contest_dated_cs4a import clock, invoke, write

    summary = {}

    def case(name, current, settled, reading, expected, *, manual=None, revoke=False):
        directory = root / name
        directory.mkdir()
        now = dt.datetime.fromisoformat(current).replace(tzinfo=ET)
        for day in settled:
            dump(directory / f"{day}.json", {"result": "hit"})
        if manual == "legacy":
            write(directory, dict(active_streak=1, best_streak=1, source_date=None,
                                  recorded_at=reading, override_expires_at=(now + dt.timedelta(hours=1)).isoformat()))
        elif manual == "dated":
            with pytest.MonkeyPatch.context() as patch:
                clock(patch, now)
                result = invoke(directory, day=now.date().isoformat(), streak=0, best=0)
                assert result.exit_code == 0, result.output
        else:
            covered_automatic.__wrapped__()(directory, now, {
                "active_streak": 1, "best_streak": 1, "source_date": None, "recorded_at": reading,
            })
        if revoke:
            from bts.cli import cli
            with pytest.MonkeyPatch.context() as patch:
                clock(patch, now)
                response = CliRunner().invoke(cli, [
                    "season-contract", "revoke", "--season", str(now.year),
                    "--picks-dir", str(directory), "--reason", "offline review",
                ])
            assert response.exit_code == 0, response.output
        alerts = check(directory, expected=True, now=now)
        coverage = [a.level for a in alerts if "no source_date" in a.message]
        if revoke:
            assert alerts[-1].level == "CRITICAL" and "revoked" in alerts[-1].message
        observed = coverage[0] if coverage else "none"
        assert observed == (expected if tree == "head" or expected != "INFO" else "CRITICAL"), alerts
        summary[name] = [(a.source, a.level, a.message.split(" (")[0]) for a in alerts]

    opener = ["2027-03-24", "2027-03-25", "2027-03-26"]
    first = "2027-03-24T17:30:00Z"
    for offset, level in enumerate(("INFO", "INFO", "CRITICAL")):
        case(f"opening-{offset}", opener[offset] + "T22:15", opener[:offset + 1], first, level)
    case("third-fresh", "2027-03-26T22:15", opener, "2027-03-26T21:00:00Z", "CRITICAL")
    case("moved-first", "2027-04-20T22:15", ["2027-04-20"], "2027-04-20T17:30:00Z", "INFO")
    case("moved-second", "2027-04-21T22:15", ["2027-04-20", "2027-04-21"],
         "2027-04-20T17:30:00Z", "INFO")
    case("next-year", "2028-04-04T22:15", ["2028-04-03", "2028-04-04"],
         "2028-04-03T17:30:00Z", "INFO")
    case("36h", "2027-03-26T00:00", opener[:2], "2027-03-24T16:00:00Z", "INFO")
    case("over-36h", "2027-03-26T00:01", opener[:2], "2027-03-24T16:00:00Z", "CRITICAL")
    case("old-year", "2028-01-01T02:45", ["2028-01-01"], "2028-01-01T02:30:00Z", "CRITICAL")
    case("future-reading", "2027-03-24T22:15", opener[:1], "2027-03-25T03:00:00Z", "CRITICAL")
    case("future-settlement", "2027-03-24T22:15", opener[:2], first, "CRITICAL")
    case("no-settlement", "2027-03-24T22:15", ["2026-09-27"], first, "CRITICAL")
    case("legacy", "2027-03-24T22:15", opener[:1], first, "CRITICAL", manual="legacy")
    case("dated", "2027-03-24T22:15", opener[:1], first, "none", manual="dated")
    case("refusal-last", "2027-03-24T22:15", opener[:1], first, "INFO", revoke=True)
    case("july-jan-test-only", "2027-07-15T22:15", ["2027-01-02"],
         "2027-07-15T20:00:00Z", "INFO")
    case("july-one-season-settlement", "2027-07-15T22:15", opener[:1],
         "2027-07-15T20:00:00Z", "INFO")
    history = [(dt.date(2027, 3, 24) + dt.timedelta(days=i)).isoformat() for i in range(100)]
    case("july-months-of-picks", "2027-07-15T22:15", history, "2027-07-15T20:00:00Z", "CRITICAL")
    dump(root / "summary.json", summary)


def park(root, tree):
    from bts.health import park_drag_freshness as health
    from bts.features import park_drag_producer as producer
    from tests.test_park_drag_producer import TestRefresh, _FakeResp
    import pandas as pd

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            value = dt.datetime(2027, 3, 24, 12, tzinfo=ET)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    summary = {}
    fixture = root / "freshness"
    fixture.mkdir()
    (fixture / "park_drag_export.csv").write_text("fixture\n")

    def check(day, source="2026-09-27", generated="2027-03-24T12:00:00", **thresholds):
        dump(fixture / "park_drag_manifest.json",
             {"max_source_game_date": source, "generated_at": generated})
        alerts = health.check(fixture, today=dt.date.fromisoformat(day), thresholds=thresholds)
        summary[f"{day}/{source}/{generated}"] = [(a.level, a.message) for a in alerts]
        return [a.level for a in alerts]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(health, "datetime", Clock)
        original_import = builtins.__import__

        def without_lightgbm(name, *args, **kwargs):
            if name.split(".")[0] == "lightgbm":
                raise ImportError("LightGBM deliberately absent")
            return original_import(name, *args, **kwargs)

        patch.setattr(builtins, "__import__", without_lightgbm)
        expected = [[], [], [], [], ["WARN"], ["WARN"], ["WARN"], ["CRITICAL"]]
        days = ["2027-03-23", *(f"2027-03-{d}" for d in range(24, 31))]
        assert [check(day) for day in days] == (expected if tree == "head" else [["CRITICAL"]] * 8)
        assert check("2028-03-24", source="2027-09-27") == ["CRITICAL"]
        assert [check(f"2027-{month:02d}-15") for month in (10, 11, 12, 1, 2)] == [[]] * 5
        assert check("2027-04-04", source="2027-03-29") == ["CRITICAL"]
        assert check("2027-03-24", generated="2027-03-21T12:00:00") == (
            ["WARN"] if tree == "head" else ["CRITICAL", "WARN"])
        dump(fixture / "producer_status.json", {"ok": True})
        assert check("2027-03-26", source="2026-06-15") == ([] if tree == "head" else ["CRITICAL"])
        if tree == "head":
            from bts import strategy
            patch.setitem(strategy.SEASON_OPENER_DATES, 2027, "2027-04-20")
            assert [check(f"2027-04-{day}") for day in (19, 20, 21, 23, 26)] == [
                [], [], [], ["WARN"], ["CRITICAL"]]

        patch.setattr(producer, "_jitter_sleep", lambda: None)
        patch.setattr(producer, "_now_iso", lambda: "2027-03-24T12:00:00")
        patch.setattr(gzip.time, "time", lambda: 1_806_000_000)
        fixture_test = TestRefresh()
        for scenario, today, feed in (
            ("populated-new-game", dt.date(2026, 7, 7), fixture_test._fake_http()),
            ("populated-no-change", dt.date(2026, 7, 6), lambda *a, **k: _FakeResp(text="")),
            ("empty", dt.date(2027, 3, 24), None),
        ):
            directory = root / scenario
            fixture_test._seed(directory)
            pitches = directory / "producer/pitches_current.csv.gz"
            header = pd.read_csv(pitches).iloc[:0].to_csv(index=False)
            if feed is None:
                pitches.write_bytes(gzip.compress(header.encode(), mtime=0))
                feed = lambda *a, **k: _FakeResp(text=header)
            elif scenario == "populated-no-change":
                feed = lambda *a, **k: _FakeResp(text=header)
            results = []

            def capture_http(url, **kwargs):
                results.append(url)
                return feed(url, **kwargs)

            status = producer.refresh(directory, today=today, http_get=capture_http)
            assert status["ok"] is (tree == "head" or scenario != "empty"), status
            summary[scenario] = [status, results.copy()]
            if scenario == "empty":
                feed = fixture_test._fake_http("2027-03-24")
                status = producer.refresh(directory, today=dt.date(2027, 3, 25), http_get=capture_http)
                assert status["ok"] is (tree == "head"), status
                if tree == "head":
                    assert status["current_season_games"] == 1 and len(pd.read_csv(pitches)) == 30
                    assert any("game_date_lt=2027-03-24" in url for url in results)
                summary["empty-first-game"] = [status, results]
        directory = root / "both-empty"
        fixture_test._seed(directory)
        pitches = directory / "producer/pitches_current.csv.gz"
        static = directory / "producer/game_level_static.csv"
        header = pd.read_csv(pitches).iloc[:0].to_csv(index=False)
        pitches.write_bytes(gzip.compress(header.encode(), mtime=0))
        pd.read_csv(static).iloc[:0].to_csv(static, index=False)
        status = producer.refresh(directory, today=dt.date(2027, 3, 24),
                                  http_get=lambda *a, **k: _FakeResp(text=header))
        assert status["ok"] is (tree == "head")
        assert not (directory / "park_drag_export.csv").exists()
        summary["both-empty"] = status
    dump(root / "summary.json", summary)


def cron(root, tree):
    import httpx
    from bts import cli as main, contest_fetch, dm
    import tests.test_entry_intent as fixture
    import tests.test_contest_fetch as requests

    code_root = ROOT if tree == "head" else BASE
    fixture.SCRIPT = code_root / "scripts/cron-setup-hetzner.sh"
    box = fixture.box.__wrapped__(root)
    results = {}
    for intent, body in (
        ("enter", 'entry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'),
        ("research", 'entry_intent = "research"\npick_delivery = "private"\n'),
    ):
        box.config.write_text("[scheduler]\n" + body)
        response = box.run("show")
        assert response.returncode == 0, response.stderr
        results[intent] = response.stdout
    dump(root / "show.json", results)

    invoke = CliRunner.invoke
    calls = []

    def inspect(self, *args, **kwargs):
        response = invoke(self, *args, **kwargs)
        calls.append((response.exit_code, response.stderr))
        return response

    outcomes = {}
    for failure, category in requests.test_green_control_fetch_failure_without_recipient.pytestmark[0].args[1]:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CliRunner, "invoke", inspect)
            directory = root / failure
            requests.test_green_control_fetch_failure_without_recipient(directory, patch, failure, category)
        status = json.loads((directory / "health_state/contest_streak_fetch_status.json").read_text())
        assert status["last_error"] in calls[-1][1]
        outcomes[failure] = [*calls[-1], status, 0]
    for name, rounds, category in (
        ("empty-rounds", Mock(return_value={}), "actionable"),
        ("rounds-network", Mock(side_effect=httpx.ConnectError("offline rounds")), "transient"),
        ("prior-nonobject", Mock(return_value={}), None),
    ):
        directory = root / name
        picks = directory / "picks"
        with pytest.MonkeyPatch.context() as patch:
            requests._mock_auth_success(patch)
            patch.setattr(main, "_fetch_rounds", rounds)
            patch.setattr(contest_fetch, "fetch_profile",
                          lambda *_: {"activeStreak": 0, "seasonBestStreak": 0, "predictions": []})
            sender = Mock()
            patch.setattr(dm, "send_dm", sender)
            if category is None:
                dump(picks / "account_state/contest_streak.json", [])
            response = CliRunner().invoke(main.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
            sender.assert_not_called()
        path = directory / "health_state/contest_streak_fetch_status.json"
        if category is None:
            assert response.exit_code == 1 and isinstance(response.exception, AttributeError)
            assert not response.stderr and not path.exists()
            outcomes[name] = [response.exit_code, response.stderr, str(response.exception), 0]
        else:
            status = json.loads(path.read_text())
            assert response.exit_code == 2 and status["last_error"] in response.stderr
            assert status["last_error_category"] == category and status["last_alert_at"] is None
            outcomes[name] = [response.exit_code, response.stderr, status, 0]
    dump(root / "fetch.json", outcomes)


def pick_path(root, tree):
    from bts import daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert, contest_state, runner
    from tests.test_pick_chosen_on_cs5a import DAY, NOW, set_dated
    from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    original_poll = scheduler.run_result_polling
    with pytest.MonkeyPatch.context() as patch, workdir(root):
        config, directory, _, _ = world.__wrapped__(root, patch)
        prepare_run_day(patch)
        config["scheduler"]["entry_intent"] = "enter"
        config["orchestrator"]["models_dir"] = "models"
        config["health_checks"] = {"enabled": True}
        set_dated(directory, streak=30, best=30)
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
        # Use actual polling and scoring, with the game/boxscore ends captured.
        patch.setattr(scheduler, "run_result_polling", original_poll)
        patch.setattr(scheduler, "_watchdog_ping_sleep", lambda *a, **k: None)
        patch.setattr(scheduler, "poll_game_result", Mock(return_value="final"))
        patch.setattr(picks, "check_hit", Mock(return_value="hit"))
        patch.setattr(posting, "reply_to_bluesky", Mock(return_value="at://reply"))
        patch.setattr(runner, "get_self_pid", lambda: None)
        patch.setattr(runner, "read_systemd_nrestarts", lambda: None)
        for module in (picks, daily_decision, scheduler, contest_state, strategy):
            patch.setattr(module, "datetime", Clock)
        disk = shutil.disk_usage(root)
        patch.setattr(runner.disk_fill.shutil, "disk_usage",
                      lambda *_: type(disk)(total=1000, used=500, free=500))
        park = Path("data/external/park_drag")
        dump(park / "park_drag_manifest.json",
             {"max_source_game_date": "2027-06-30", "generated_at": "2027-07-15T18:35:00"})
        (park / "park_drag_export.csv").write_text("fake\n")
        scheduler.run_day(date=DAY, config=config)
        assert scheduler.load_state(DAY, directory).pick_locked
        daily = picks.load_pick(DAY, directory)
        assert daily.result == "hit" and daily.notification_sent and picks.check_hit.called
        assert dm.send_dm.call_count == 1 and alert.send_dm.call_count == 1
        assert "[park_drag_freshness]" in alert.send_dm.call_args.args[1]
        assert not posting.post_to_bluesky.called and not posting.reply_to_bluesky.called
        files = {"pick": directory / f"{DAY}.json"}
        files["decision"] = daily_decision.decision_path(DAY, directory)
        files["score"], = directory.glob("*streak*.json")
        for key, source in files.items():
            (root / f"{key}.json").write_bytes(source.read_bytes())
        dump(root / "sends.json", {
            "pick_dm": dm.send_dm.call_args_list,
            "health_dm": alert.send_dm.call_args_list,
            "public_posts": posting.post_to_bluesky.call_args_list,
            "public_replies": posting.reply_to_bluesky.call_args_list,
        })


def capture(root, tree):
    import pandas as pd
    from bts.cli import cli
    from bts.experiment import artifacts
    from bts.health.analytics_artifacts_missing import _check_capture
    from bts.model import predict
    from scripts import live_forward_capture_once as live
    from tests.experiment.test_artifacts import _write_pick_file

    day = "2027-03-24"
    head = "offline-capture-sha"
    candidate = "decision_weighted_lgbm_v0"
    summaries = {}
    for scenario in ("thin", "legacy-full", "retry", "settled", "pa", "changed",
                     "pa-before-export", "pa-during-swap", "verified-marker-loss"):
        location = root / scenario
        production = location / "prod"
        config = live.CaptureConfig(
            date=day, production_root=production, live_forward_root=location,
            python=Path("/unused/python"), candidate=candidate, artifact_root=Path("live"),
            data_dir=Path("processed"), picks_dir=Path("picks"), top_n=10, overwrite=False,
            fail_on_pending=False, auto_recapture_on_snapshot_drift=False,
        )
        artifact = production / "live" / day
        manifest = artifact / "manifest.json"
        verification = artifact / "verification.json"
        pick = production / "picks" / f"{day}.json"
        pick.parent.mkdir(parents=True)
        _write_pick_file(pick, date=day)
        original_pick = pick.read_bytes()
        pa_file = production / config.data_dir / "pa_2027.parquet"
        exports = []
        observations = []
        results = []
        count = 18 if scenario == "legacy-full" else 9
        inject = None

        def write_pa():
            pa_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": [day], "event": ["single"]}).to_parquet(pa_file, index=False)

        def change_pick(**changes):
            raw = json.loads(original_pick)
            raw["pick"].update(changes)
            dump(pick, raw)

        def pipeline(*args, **kwargs):
            observations.append({"scored": count, "processed_pa_visible": pa_file.exists()})
            return pd.DataFrame({
                "batter_id": [11 * (i + 1) for i in range(count)],
                "game_pk": [1001 + i for i in range(count)],
                "p_game_hit": [0.90 - 0.02 * i for i in range(count)],
            })

        def transport(config, args):
            if args[1] == "export-live-candidate-artifacts":
                exports.append(str(Path(args[args.index("--output-dir") + 1]).name))
                if inject == "pa-before-export":
                    write_pa()
                if inject == "changed":
                    change_pick(batter_id=9999)
            response = CliRunner().invoke(cli, args, catch_exceptions=False)
            return subprocess.CompletedProcess(args, response.exit_code, response.stdout, response.stderr)

        original_move = live.shutil.move

        def move(src, dst, *args, **kwargs):
            if inject == "pa-during-swap" and Path(src) == artifact:
                write_pa()
            return original_move(src, dst, *args, **kwargs)

        def tick():
            code, payload = live.capture_once(config)
            results.append([code, payload["status"], payload.get("verification_previously_passed")])
            return code, payload["status"]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(live, "git_head", lambda *_: head)
            patch.setattr(live, "run_bts", transport)
            patch.setattr(live.shutil, "move", move)
            patch.setattr(predict, "run_pipeline", pipeline)
            patch.setattr(artifacts, "current_git_commit", lambda: head)
            if scenario == "thin":
                assert tick() == ((0, "exported_verified") if tree == "head" else (1, "failed_verify"))
                before = manifest.read_bytes()
                saved = json.loads(before)
                assert saved["top_n"] == 10
                if tree == "head":
                    assert saved["effective_top_n"] == 9
                count = 18
                assert tick() == ((0, "existing_verified") if tree == "head" else (1, "failed_verify_existing"))
                assert manifest.read_bytes() == before and len(exports) == 1
            else:
                if scenario == "verified-marker-loss":
                    count = 18
                output = live.export_artifact(config, pick_path=pick, artifact_dir=artifact)
                assert output.returncode == 0, output
                data = json.loads(manifest.read_text())
                data.pop("effective_top_n", None)
                dump(manifest, data)
                before = manifest.read_bytes()
                if scenario == "legacy-full":
                    assert tick() == (0, "existing_verified")
                elif scenario == "verified-marker-loss":
                    assert tick() == (0, "existing_verified")
                    profile_name = data["profile_paths"]["production"][day]
                    profile_path = artifact / profile_name
                    profile_path.write_bytes(b"unreadable prior profile")
                    assert tick() == (1, "failed_verify_existing")
                    change_pick(batter_id=9999)
                    assert tick() == (1, "stale_pick_snapshot")
                    pick.write_bytes(original_pick)
                    count = 18
                    assert tick() == ((0, "recaptured_after_failed_verification") if tree == "head"
                                      else (1, "failed_verify_existing"))
                else:
                    assert tick() == (1, "failed_verify_existing")
                    count = 18
                    if scenario == "settled":
                        dump(pick, json.loads(original_pick) | {"result": "hit"})
                    elif scenario == "pa":
                        write_pa()
                    else:
                        inject = scenario
                    expected = {
                        "retry": (0, "recaptured_after_failed_verification"),
                        "settled": (1, "failed_recapture_post_resolution"),
                        "pa": (1, "failed_recapture_post_outcomes"),
                        "changed": (1, "failed_recapture_publication_check"),
                        "pa-before-export": (1, "failed_recapture_publication_check"),
                        "pa-during-swap": (0, "recaptured_after_failed_verification"),
                    }[scenario]
                    assert tick() == (expected if tree == "head" else (1, "failed_verify_existing"))
                    if tree == "head" and scenario == "pa-before-export":
                        assert any(item["processed_pa_visible"] for item in observations)
                    if tree == "head" and scenario == "pa-during-swap":
                        assert pa_file.exists()
                backups = sorted(artifact.parent.glob(f"{day}.stale_pick_snapshot.*"))
                if tree == "head" and scenario in {"retry", "pa-during-swap", "verified-marker-loss"}:
                    assert len(backups) == 1 and (backups[0] / "manifest.json").read_bytes() == before
                else:
                    assert not backups and manifest.read_bytes() == before
        report = json.loads(verification.read_text())
        health = _check_capture(pick.parent, day, {}, artifact.parent, None)
        if scenario in {"pa-during-swap", "verified-marker-loss"}:
            assert [alert.level for alert in health] == ([] if tree == "head" else ["CRITICAL"])
        summaries[scenario] = [results, exports, observations, report["ok"],
                               [c["name"] for c in report["checks"] if c["status"] != "pass"],
                               [alert.level for alert in health]]
    dump(root / "summary.json", summaries)


def shadow(root, tree):
    from bts import cli as main, daily_decision, picks, shadow_eval, skip_policy_shadow as policy, web
    from tests.test_shadow_eval import _daily, _pick
    from tests.test_skip_policy_shadow import _cand

    old_days = [(dt.date(2026, 8, 1) + dt.timedelta(days=i)).isoformat() for i in range(29)]
    new_day = "2027-03-24"
    old_day = "2026-09-27"
    now = [dt.datetime(2027, 7, 15, 18, tzinfo=ET)]
    original_datetime = dt.datetime

    class Clock(original_datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0].astimezone(tz) if tz else now[0].replace(tzinfo=None)

    def add(directory, day, *, shadow_result="hit"):
        picks.save_pick(_daily(day, _pick("Prod", 1), result="hit"), directory)
        picks.save_shadow_pick(_daily(day, _pick("Shadow", 2), result=shadow_result), directory)
        daily_decision.write_decision(day, directory, action="skip", source="mdp", primary=_cand(),
                                      streak=10, saver_available=False, now=now[0],
                                      delivery_status="not_applicable", scoreable=False)
        assert policy.record_skip_from_decision(day, directory, now=now[0])
        path = policy.decision_path(day, directory)
        payload = json.loads(path.read_text())
        payload["shadow_pick_result"] = "hit"
        dump(path, payload)

    def command(directory, label, args, extra=()):
        path = root / f"{label}.json"
        opts = {"shadow-status": "--output",
                "skip-policy-shadow-update": "--status-output",
                "check-results": "--shadow-status-output"}
        response = CliRunner().invoke(main.cli, [*args, "--picks-dir", str(directory),
                                                  opts[args[0]], str(path), *extra], catch_exceptions=False)
        assert response.exit_code == 0, response.output
        data = json.loads(path.read_text())
        (root / f"{label}.stdout").write_text(response.stdout)
        return data, response.stdout

    storage = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dt, "datetime", Clock)
        patch.setattr(shadow_eval, "datetime", Clock)
        patch.setattr(policy, "datetime", Clock)
        patch.setattr(picks, "datetime", Clock)
        patch.setattr(main, "_now_et", lambda: now[0])
        patch.setattr(shadow_eval, "_current_git_commit", lambda: "fixture")
        for name, days in (("mixed", old_days + [new_day]), ("prior", old_days), ("current", [new_day])):
            directory = root / name
            directory.mkdir()
            for day in days:
                add(directory, day)
        for directory_name in ("mixed", "prior", "current"):
            directory = root / directory_name
            for name, args, metric in (
                ("shadow", ["shadow-status"], "resolved_paired_days"),
                ("policy", ["skip-policy-shadow-update"], "divergent_days"),
                ("results", ["check-results", "--date", new_day], "resolved_paired_days"),
            ):
                status, _ = command(directory, f"{directory_name}-{name}", args)
                storage[f"{directory_name}-{name}"] = status["counts"][metric]
                if tree == "head":
                    if name != "results":
                        command(directory, f"{directory_name}-{name}-2026", args, ["--season", "2026"])
                    else:
                        command(directory, f"{directory_name}-{name}-2026", ["check-results", "--date", old_days[-1]])

        expected = {"mixed": 1, "prior": 0, "current": 1} if tree == "head" else {
            "mixed": 30, "prior": 29, "current": 1,
        }
        for name in ("shadow", "policy", "results"):
            assert all(storage[f"{folder}-{name}"] == count for folder, count in expected.items()), storage
        if tree == "head":
            command(root / "mixed", "policy-date-2026", ["skip-policy-shadow-update", "--date", old_days[-1]])
        backfill = shadow_eval.build_shadow_backfill_manifest(
            root / "mixed", n_bootstrap=20, hit_checker=lambda *_: "hit")
        assert [row["date"] for row in backfill["rows"]] == old_days + [new_day]

        for day in ("2027-01-15", "2027-02-15", "2027-03-23"):
            now[0] = original_datetime.fromisoformat(day + "T12:00:00-05:00")
            for name, args in (
                ("shadow", ["shadow-status"]),
                ("policy", ["skip-policy-shadow-update"]),
                ("results", ["check-results", "--date", day]),
            ):
                status, _ = command(root / "prior", f"preseason-{day}-{name}", args)
                if name == "policy" and tree == "head":
                    assert web.render_skip_policy_shadow_section(status) == ""
                if name != "policy" and tree == "head":
                    assert status["cycle_state"] == "no_shadow_files"
                    assert "Verify scheduler.shadow_model" in status["action_items"][0]

        directory = root / "contradiction"
        directory.mkdir()
        add(directory, old_day)
        daily_decision.write_decision(old_day, directory, action="single", source="mdp", primary=_cand(),
                                      streak=10, saver_available=False, now=now[0],
                                      delivery_status="delivered", scoreable=True)
        assert policy.find_aged_contradictions(directory, now=now[0]) == [old_day]
        status, stdout = command(directory, "aged-warning", ["skip-policy-shadow-update"])
        assert policy.decision_path(old_day, directory).exists()
        assert ("AGED CONTRADICTIONS" in stdout) is (tree == "base")
        assert status["counts"]["aged_superseded_records"] == ([old_day] if tree == "base" else [])
        if tree == "head":
            _, historical = command(directory, "aged-warning-2026",
                                    ["skip-policy-shadow-update"], ["--season", "2026"])
            assert "AGED CONTRADICTIONS" in historical

        now[0] = original_datetime.fromisoformat("2027-03-26T22:00:00-04:00")
        directory = root / "operations"
        directory.mkdir()
        add(directory, old_day)
        add(directory, new_day)
        add(directory, "2027-03-25")
        old_record = policy.decision_path(old_day, directory)
        dump(old_record, json.loads(old_record.read_text()) | {"shadow_pick_result": None})
        policy.decision_path(new_day, directory).unlink()
        daily_decision.write_decision("2027-03-25", directory, action="single", source="mdp",
                                      delivery_status="delivered", scoreable=True)
        patch.setattr(policy, "make_hit_checker", lambda: lambda *_: "hit")
        _, operations = command(directory, "policy-operations", ["skip-policy-shadow-update"])
        assert "Pruned 1" in operations and "Recorded 1" in operations and "Reconciled 2" in operations
        assert json.loads(old_record.read_text())["shadow_pick_result"] == "hit"
        assert not policy.decision_path("2027-03-25", directory).exists()

        day = "2027-03-26"
        picks.save_pick(_daily(day, _pick("Prod", 1)), directory)
        picks.save_shadow_pick(_daily(day, _pick("Shadow", 2)), directory)
        daily_decision.write_decision(day, directory, action="single", source="mdp",
                                      delivery_status="delivered", scoreable=True)
        patch.setattr(picks, "get_game_statuses_detailed", lambda *_: {})
        patch.setattr(picks, "check_hit", lambda *_, batter_name, **__: (
            "hit" if batter_name == "Prod" else "miss"))
        _, grading = command(directory, "check-results-grading", ["check-results", "--date", day])
        assert "HIT! Prod." in grading and "Shadow: Shadow — MISS" in grading
        assert picks.load_streak(directory, season=2027) == 1
        for name in (f"{day}.json", f"{day}.shadow.json", "streak.json"):
            (root / f"grading-{name}").write_bytes((directory / name).read_bytes())
    dump(root / "summary.json", storage)


def compare(family, parent):
    head = parent / "head"
    base = parent / "base"
    if family == "cron":
        h = json.loads((head / "show.json").read_text())
        b = json.loads((base / "show.json").read_text())
        scrub = lambda value, path: value.replace(str(path), "<fixture>")
        assert scrub(h["enter"], head) == scrub(b["enter"], base)
        recipient = " --dm-recipient stonehengee.bsky.social"
        lines = [line for line in b["research"].splitlines() if "bts fetch-contest-streak" in line]
        assert len(lines) == 4 and all(recipient in line for line in lines)
        expected = "\n".join(line.replace(recipient, "") if line in lines else line
                             for line in b["research"].splitlines()) + "\n"
        assert scrub(h["research"], head) == scrub(expected, base)
    elif family == "pick":
        for name in ("pick.json", "decision.json", "score.json", "sends.json"):
            assert (head / name).read_bytes() == (base / name).read_bytes(), name
    elif family == "park":
        for scenario in ("populated-new-game", "populated-no-change"):
            for name in ("park_drag_export.csv", "park_drag_manifest.json", "producer_status.json",
                         "producer/games_meta.csv"):
                if (head / scenario / name).exists():
                    assert (head / scenario / name).read_bytes() == (base / scenario / name).read_bytes(), name
            name = "producer/pitches_current.csv.gz"
            assert (head / scenario / name).read_bytes() == (base / scenario / name).read_bytes()
    elif family == "shadow":
        def normalize(path):
            data = json.loads(path.read_text())
            data.pop("generated_at", None)
            data.pop("picks_dir", None)
            text = json.dumps(data, sort_keys=True)
            return text.replace(str(head / "mixed"), "<picks>").replace(str(base / "prior"), "<picks>")

        for name in ("shadow", "policy", "results"):
            assert normalize(head / f"mixed-{name}-2026.json") == normalize(base / f"prior-{name}.json"), name
        for output in sorted(head.glob("grading-*.json")):
            assert output.read_bytes() == (base / output.name).read_bytes(), output


FAMILIES = {"contest": contest, "park": park, "cron": cron,
            "pick": pick_path, "capture": capture, "shadow": shadow}


def probes(family, tree=None):
    output = REVIEW / f"probe-{family}"
    if tree:
        import bts
        import tests
        source = ROOT if tree == "head" else BASE
        assert Path(bts.__file__).is_relative_to(source / "src"), bts.__file__
        tests.__path__.insert(0, str(ROOT / "tests"))
        FAMILIES[family](reset(output / tree), tree)
        print(f"{family} {tree}: assertions passed")
        return 0
    for label, source in (("base", BASE), ("head", ROOT)):
        code = run(source, "probe", family, label, cwd=source)
        if code:
            return code
    compare(family, output)
    for path in sorted(output.glob("*/summary.json")):
        print(path.relative_to(REVIEW), hashlib.sha256(path.read_bytes()).hexdigest()[:12])
    print(f"{family}: base and branch comparison passed")
    return 0
