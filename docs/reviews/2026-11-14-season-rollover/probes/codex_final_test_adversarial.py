import copy
import importlib.util
import io
import itertools
import json
import shutil
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.conftest import sample_game_feed
from bts.cli import cli
from bts.data.build import build_season

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parent


def baseline():
    spec = importlib.util.spec_from_file_location("g2_base_build", SCRATCH / "base_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_season


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_builder_differential(tmp_path, sample_game_feed):
    base = baseline()
    raw = tmp_path / "raw"
    season = raw / "2027"
    target = tmp_path / "proc" / "pa_2027.parquet"
    untyped = [{}, {"gameData": {}}, {"gameData": {"game": {}}},
               *[{"gameData": {"game": {"type": t}}} for t in (None, 7, True, "", "   ")]]
    fatal = [[], None, {"gameData": None}, {"gameData": {"game": None}},
             *[{"gameData": {"game": {"type": t}}} for t in ([], {})]]
    kinds = [("untyped", f) for f in untyped] + [("fatal", f) for f in fatal]
    kinds += [("excluded", {"gameData": {"game": {"type": t}}}) for t in ("S", "E", "X", " R ")]
    kinds += [("missing", None), ("empty", None), ("truncated", None), ("sidecar", None)]
    observations = []
    for (kind, feed), companion, existing in itertools.product(kinds, (None, "S", "R", "emptyR", "sevenR"), (False, True)):
        if raw.exists():
            shutil.rmtree(raw)
        if target.parent.exists():
            shutil.rmtree(target.parent)
        if kind != "missing":
            season.mkdir(parents=True)
        if kind in ("untyped", "fatal", "excluded"):
            (season / "100.json").write_text(json.dumps(feed))
        if kind == "truncated":
            (season / "100.json").write_text('{"gameData":')
        if kind == "sidecar":
            (season / "100_weather.json").write_text("broken JSON sidecar")
        if companion:
            season.mkdir(parents=True, exist_ok=True)
            other = copy.deepcopy(sample_game_feed)
            other["gameData"]["game"]["type"] = "S" if companion == "S" else "R"
            if companion == "emptyR":
                other["liveData"]["plays"]["allPlays"] = []
            if companion == "sevenR":
                other["gameData"]["game"]["doubleHeader"] = "S"
                for p in other["liveData"]["plays"]["allPlays"]:
                    p["about"]["inning"] = 7
            (season / "200.json").write_text(json.dumps(other))
        if existing:
            target.parent.mkdir()
            target.write_bytes(b"EXISTING PARQUET SENTINEL")
        initial = snapshot(tmp_path)
        results = []
        for fn in (base, build_season):
            err = io.StringIO()
            try:
                with redirect_stderr(err):
                    frame = fn(raw, target, 2027)
                result = ("none" if frame is None else "rows", frame, None)
            except Exception as exc:
                result = ("error", None, type(exc))
                assert snapshot(tmp_path) == initial
            results.append((*result, err.getvalue()))
            if target.exists() and not existing:
                target.unlink()
                target.parent.rmdir()
            elif existing:
                target.write_bytes(b"EXISTING PARQUET SENTINEL")
                # For checking the next call's failure, retain its actual restored mtime.
                initial = snapshot(tmp_path)
        old, new = results
        noop = kind in ("excluded", "missing", "empty", "sidecar") and companion in (None, "S")
        if noop:
            assert old[0] == "error" and new[0] == "none", (kind, companion, old, new)
        elif kind == "untyped" and companion in (None, "S"):
            assert old[0] == new[0] == "error" and new[2] is ValueError
        else:
            assert (old[0], old[2]) == (new[0], new[2]), (kind, companion, old, new)
            if old[0] == "rows":
                pd.testing.assert_frame_equal(old[1], new[1])
        cli_result = CliRunner().invoke(cli, ["data", "build", "--seasons", "2027",
            "--raw-dir", str(raw), "--out-dir", str(target.parent)])
        assert (cli_result.exit_code == 0) == (new[0] != "error"), (kind, companion, cli_result.output)
        observations.append((kind, companion, existing, old[0], new[0]))
    (SCRATCH / "builder_matrix.json").write_text(json.dumps(observations, indent=1))


def prepare_history(root, sample):
    for year in (2025, 2026):
        folder = root / "data" / "raw" / str(year)
        folder.mkdir(parents=True)
        for day in range(1, 17):
            feed = copy.deepcopy(sample)
            feed["gameData"]["game"].update(pk=year * 100 + day, season=str(year))
            feed["gameData"]["datetime"]["officialDate"] = f"{year}-06-{day:02}"
            (folder / f"{year * 100 + day}.json").write_text(json.dumps(feed))
        build_season(root / "data" / "raw", root / "data" / "processed" / f"pa_{year}.parquet", year)
    models = root / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        shutil.copyfile(ROOT / "data" / "models" / name, models / name)


def test_cold_pipeline_training_selection_delivery(tmp_path, monkeypatch, sample_game_feed):
    import bts.model.predict as predict
    import bts.data.pull as pull
    import bts.scheduler as sch
    import bts.picks as picks
    import bts.strategy as strategy
    from bts.daily_decision import load_decision
    prepare_history(tmp_path, sample_game_feed)
    monkeypatch.chdir(tmp_path)
    posted = {"value": False}
    day = {"value": "2027-03-24"}
    network_calls = []
    historical = copy.deepcopy(sample_game_feed)
    historical["gameData"]["game"].update(pk=9001, season="2027")
    historical["gameData"]["datetime"]["officialDate"] = "2027-03-24"
    spring = tmp_path / "data" / "raw" / "2027"
    spring.mkdir()
    nonregular = copy.deepcopy(historical)
    nonregular["gameData"]["game"]["type"] = "S"
    (spring / "8888.json").write_text(json.dumps(nonregular))

    def game(date, final=False):
        return {"gamePk": 9001 if date == "2027-03-24" else 9002, "gameType": "R",
                "officialDate": date, "gameDate": date + "T23:10:00Z",
                "status": {"detailedState": "Final" if final else "Scheduled", "abstractGameCode": "F" if final else "P"},
                "teams": {"away": {"team": {"id": 134}, "probablePitcher": {"id": 400001, "fullName": "Away Pitcher"}},
                          "home": {"team": {"id": 121}, "probablePitcher": {"id": 300001, "fullName": "Home Pitcher"}}}}

    def network(url, **kw):
        url = getattr(url, "full_url", url)
        network_calls.append(url)
        if "/schedule" in url:
            q = parse_qs(urlparse(url).query)
            if "teamId" in q:
                games = [game("2027-03-24", True)] if day["value"] == "2027-03-25" else []
            else:
                date = q["date"][0]
                games = [game(date, date < day["value"])] if date >= "2027-03-24" else []
            obj = {"dates": [{"games": games}]}
        elif "/feed/live" in url:
            if "9001" in url and day["value"] == "2027-03-25":
                obj = historical
            else:
                obj = copy.deepcopy(historical)
                obj["liveData"]["plays"]["allPlays"] = []
                for side, abbr in (("away", "PIT"), ("home", "NYM")):
                    obj["gameData"]["teams"][side]["abbreviation"] = abbr
                    if not posted["value"]:
                        for player in obj["liveData"]["boxscore"]["teams"][side]["players"].values():
                            player.pop("battingOrder", None)
        else:
            raise AssertionError(f"unexpected network call {url}")
        return io.BytesIO(json.dumps(obj).encode())

    for module, attr in ((predict, "urlopen"), (pull, "urlopen"), (pull, "retry_urlopen"),
                         (picks, "retry_urlopen"), (sch, "retry_urlopen")):
        monkeypatch.setattr(module, attr, network)
    delivered = []
    monkeypatch.setattr("bts.dm.send_dm", lambda recipient, text: delivered.append((recipient, text)) or "fake-dm-id")
    monkeypatch.setattr(sch, "_now_et", lambda: datetime.fromisoformat(day["value"] + "T18:00:00").replace(tzinfo=ZoneInfo("America/New_York")))
    # No training, lookup, selection, builder or persistence function is replaced.
    cache = tmp_path / "data" / "models" / "probe.pkl"
    result = predict.run_pipeline(day["value"], save_blend_path=cache)
    assert result.empty, "Opening Night cannot project a prior same-season lineup before one exists"
    assert not (tmp_path / "data" / "processed" / "pa_2027.parquet").exists()
    summary = {"pre_lineup_predictions": len(result), "days": []}
    for date in ("2027-03-24", "2027-03-25"):
        day["value"] = date
        posted["value"] = True
        if date.endswith("25"):
            (spring / "9001.json").write_text(json.dumps(historical))
        from bts.orchestrator import run_and_pick
        config = {"orchestrator": {"picks_dir": str(tmp_path / "data" / "picks")},
                  "tiers": [{"name": "local", "type": "local"}],
                  "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False},
                  "bluesky": {"dm_recipient": "fake-recipient"}}
        if date.endswith("24"):
            shutil.copyfile(cache, tmp_path / "data" / "models" / f"blend_{date}.pkl")
        strategy._mdp_cache = None
        df, sel, tier = run_and_pick(config, date)
        assert tier == "local" and len(df) == 2 and df.p_game_hit.notna().all()
        assert sel.source == "mdp" and sel.action == "single" and sel.pick_result is not None
        state = sch.SchedulerState(date, "", [], [], [], False, None, None, None)
        assert sch._deliver_and_lock_pick(sel.pick_result.daily, config, tmp_path / "data" / "picks",
                                          state, date, "probe", selection=sel)
        dec = load_decision(date, tmp_path / "data" / "picks")
        assert dec["scoreable"] and dec["delivery_status"] == "delivered" and state.pick_locked
        summary["days"].append({"date": date, "predictions": len(df), "top_p": float(df.p_game_hit.max()),
                                "action": sel.action, "source": sel.source, "decision": dec})
    assert len(delivered) == 2
    assert (tmp_path / "data" / "processed" / "pa_2027.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "data" / "processed" / "pa_2027.parquet")) == 2
    summary["network_calls"] = network_calls
    (SCRATCH / "cold_pipeline.json").write_text(json.dumps(summary, indent=1))

    # Existing feature code treats every *.json stem as a numeric game id, including weather sidecars.
    (spring / "9001_weather.json").write_text('{"surface_pressure": 1013}')
    with pytest.raises(ValueError, match="9001_weather"):
        predict.run_pipeline("2027-03-25", cached_blend=predict.load_blend(cache))


def test_next_day_prior_lineup_projection(tmp_path, monkeypatch, sample_game_feed):
    import bts.model.predict as p
    historical = copy.deepcopy(sample_game_feed)
    historical["gameData"]["teams"]["away"]["abbreviation"] = "PIT"
    historical["gameData"]["teams"]["home"]["abbreviation"] = "NYM"
    now_feed = copy.deepcopy(historical)
    now_feed["liveData"]["plays"]["allPlays"] = []
    for side in ("away", "home"):
        for player in now_feed["liveData"]["boxscore"]["teams"][side]["players"].values():
            player.pop("battingOrder", None)
    def network(url, **kw):
        if "/schedule" in url:
            prior = "teamId=" in url
            games = [{"gamePk": 9001 if prior else 9002, "gameType": "R", "gameDate": "2027-03-25T23:10:00Z",
                      "status": {"detailedState": "Final" if prior else "Scheduled"},
                      "teams": {"home": {}, "away": {}}}]
            obj = {"dates": [{"games": games}]}
        else:
            obj = historical if "/9001/" in url else now_feed
        return io.BytesIO(json.dumps(obj).encode())
    monkeypatch.setattr(p, "urlopen", network)
    slots = p._fetch_game_slots("2027-03-25")
    assert len(slots) == 2 and all(s["projected"] for s in slots)


def test_opening_day_late_lineup_has_no_pick_to_retry(tmp_path, monkeypatch, sample_game_feed):
    import bts.model.predict as p
    import bts.data.pull as pull
    import bts.scheduler as sch
    import bts.picks as picks
    from bts.daily_decision import load_decision
    from datetime import timedelta
    prepare_history(tmp_path, sample_game_feed)
    monkeypatch.chdir(tmp_path)
    clock = {"now": datetime.fromisoformat("2027-03-24T10:00:00-04:00")}
    calls = []
    date = "2027-03-24"
    def network(url, **kw):
        url = getattr(url, "full_url", url)
        calls.append((clock["now"].isoformat(), url))
        if "/schedule" in url:
            q = parse_qs(urlparse(url).query)
            games = []
            if q.get("date") == [date]:
                games = [{"gamePk": 9001, "gameType": "R", "gameDate": date + "T23:10:00Z",
                          "status": {"detailedState": "Scheduled", "abstractGameCode": "P"},
                          "teams": {"away": {"team": {"id": 134, "name": "Pirates"}, "probablePitcher": {"id": 400001}},
                                    "home": {"team": {"id": 121, "name": "Mets"}, "probablePitcher": {"id": 300001}}}}]
            obj = {"dates": [{"games": games}]}
        elif "/feed/live" in url:
            obj = copy.deepcopy(sample_game_feed)
            obj["liveData"]["plays"]["allPlays"] = []
            for side, abbr in (("away", "PIT"), ("home", "NYM")):
                obj["gameData"]["teams"][side]["abbreviation"] = abbr
                if clock["now"].strftime("%H:%M") < "18:25":
                    for player in obj["liveData"]["boxscore"]["teams"][side]["players"].values():
                        player.pop("battingOrder", None)
        else:
            raise AssertionError(url)
        return io.BytesIO(json.dumps(obj).encode())
    for module, attr in ((p, "urlopen"), (pull, "urlopen"), (pull, "retry_urlopen"),
                         (picks, "retry_urlopen"), (sch, "retry_urlopen")):
        monkeypatch.setattr(module, attr, network)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)
    monkeypatch.setattr(sch, "datetime", Clock)
    monkeypatch.setattr(sch, "_watchdog_ping_sleep", lambda seconds, *a, **kw: clock.update(now=clock["now"] + timedelta(seconds=seconds)))
    sent = []
    monkeypatch.setattr("bts.dm.send_dm", lambda *a: sent.append(a) or "fake")
    config = {"orchestrator": {"picks_dir": str(tmp_path / "data" / "picks"),
                               "heartbeat_path": str(tmp_path / "data" / ".heartbeat")},
              "tiers": [{"name": "local", "type": "local"}],
              "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60,
                            "live_forward_capture_on_lock": False},
              "bluesky": {"dm_recipient": "fake"}, "health_checks": {"enabled": False}}
    sch.run_day(date, config)
    state = sch.load_state(date, tmp_path / "data" / "picks")
    assert [r["time"][11:16] for r in state.runs_completed] == ["16:10", "17:10", "18:10"]
    assert not state.pick_locked and not sent
    assert load_decision(date, tmp_path / "data" / "picks") is None
    assert picks.load_pick(date, tmp_path / "data" / "picks") is None
    # No first-day projection: no pick exists to arm either fallback or missed-pick warning.
    assert not any(time[11:16] >= "18:25" and "/9001/feed/live" in url for time, url in calls)
    (SCRATCH / "late_lineup_day.json").write_text(json.dumps({"runs": state.runs_completed, "calls": calls}, indent=1))


def test_receipt_precedence_and_schedule_network(tmp_path, monkeypatch):
    import bts.scheduler as sch
    from bts.day_outcome import classify_day
    from bts.daily_decision import write_decision, load_decision_bytes
    from bts.entry_receipt import _sha, receipts_root
    from bts.receipt_io import publish
    date = "2027-03-24"
    picks = tmp_path / "picks"
    write_decision(date, picks, action="single", source="mdp", scoreable=True, delivery_status="delivered",
                   primary={"batter_id": 1, "batter_name": "A", "game_pk": 9001, "p_game_hit": .8})
    dec, raw = load_decision_bytes(date, picks)
    sha = _sha(raw)
    def receipt(kind):
        return {"schema": "bts_pick_entry_receipt_v1", "outcome": "observed", "et_date": date, "season": 2027,
                "degraded": [], "account": {"username": "owner"}, "selection": {"commit": {"decision_sha256": sha}},
                "observation": {"before_cutoff": kind != "late", "response_completed_at": date + "T23:06:00Z"},
                "verifier": {"reason": "no_pick" if kind == "absent" else "match"},
                "qualification": {"all_confirmed": kind in ("full", "late"),
                                  "slots": [{"state": "unit_unverified" if kind == "batter" else
                                                       "confirmed" if kind in ("full", "late") else "missing"}]}}
    expected = {("absent",): "not_entered", ("late", "absent"): "entered_late",
                ("batter", "late", "absent"): "entered_game_unverified",
                ("full", "batter", "late", "absent"): "entered"}
    for kinds, state in expected.items():
        for order in itertools.permutations(kinds):
            assert classify_day(date=date, games=1, decision=dec, decision_sha256=sha,
                receipts=[receipt(k) for k in order], intent="enter", expected_username="owner").state == state
    publish(receipts_root(picks) / date / "batter.json", json.dumps(receipt("batter")).encode())
    config = tmp_path / "config.toml"
    config.write_text('[scheduler]\nentry_intent="enter"\npick_delivery="dm"\n')
    alerts = []
    monkeypatch.setattr("bts.dm.send_dm", lambda *args: alerts.append(args))
    states = {"Postponed": "no_games", "Cancelled": "no_games", "Canceled": "no_games", "Suspended": "entered_game_unverified",
              "Delayed Start": "entered_game_unverified", "Final": "entered_game_unverified"}
    for detailed, state in states.items():
        payload = {"dates": [{"games": [{"gameType": "R", "status": {"detailedState": detailed}},
                                         {"gameType": "S", "status": {"detailedState": "Final"}}]}]}
        monkeypatch.setattr(sch, "retry_urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
        res = CliRunner().invoke(cli, ["day-outcome", "--date", date, "--config", str(config),
             "--picks-dir", str(picks), "--expected-username", "owner", "--dm-recipient", "fake"])
        assert res.exit_code == 0 and json.loads(res.output)["state"] == state, res.output
    assert not alerts
