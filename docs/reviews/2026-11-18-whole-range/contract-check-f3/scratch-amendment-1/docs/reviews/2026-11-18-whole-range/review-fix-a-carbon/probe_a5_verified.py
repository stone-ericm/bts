"""Adversarial probe for A5 (fetch_schedule body shapes, callers, model reader, churn)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
import pytest

from bts import cli, day_outcome, picks, scheduler
from bts.data import lineup_collect

ROOT = Path(__file__).resolve().parents[2]
DAY = "2027-03-24"


def _load_check_heartbeat():
    spec = spec_from_file_location("_wr_a_check_hb", ROOT / "scripts" / "check_heartbeat.py")
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_offline(tmp_root: Path, mp: pytest.MonkeyPatch):
    mpl_dir = tmp_root / "_mpl"
    mpl_dir.mkdir(exist_ok=True)
    mp.setenv("MPLCONFIGDIR", str(mpl_dir))
    fixture = ROOT / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("_wr_a_sched_offline", fixture)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mp.setattr(mod, "ROOT", ROOT)
    mp.setattr(mod, "HERE", tmp_root)
    return mod


def run_a5_probes() -> dict:
    shapes = [
        ("error_object", b'{"message":"upstream failure","messageNumber":1}', True),
        ("dates_missing", b'{"copyright":"MLB"}', True),
        ("dates_null", b'{"dates":null}', True),
        ("dates_string", b'{"dates":"2027-03-24"}', True),
        ("dates_entry_not_object_int", b'{"dates":[123]}', True),
        ("dates_entry_not_object_list", b'{"dates":[[]]}', True),
        ("games_missing", b'{"dates":[{"date":"2027-03-24"}]}', True),
        ("games_null", b'{"dates":[{"date":"2027-03-24","games":null}]}', True),
        ("games_string", b'{"dates":[{"date":"2027-03-24","games":"none"}]}', True),
        ("game_not_object", b'{"dates":[{"games":["bad"]}]}', True),
        ("game_not_object_after_spring", b'{"dates":[{"games":[{"gameType":"S"},null]}]}', True),
        ("invalid_json", b'{"dates":[', True),
        ("non_utf8", b'\xff\xfe', True),
        ("empty_body", b'', True),
        ("deeply_nested", b'{"a":' * 2000 + b'1' + b'}' * 2000, True),
        ("top_level_array", b'[]', True),
        ("off_day_empty_dates", b'{"dates":[]}', False),
        ("empty_games_list", b'{"dates":[{"date":"2027-03-24","games":[]}]}', False),
        (
            "mixed_regular_and_nonregular",
            json.dumps({
                "dates": [{
                    "date": DAY,
                    "games": [
                        {"gamePk": 1, "gameType": "S"},
                        {"gamePk": 2, "gameType": "A"},
                        {"gamePk": 3, "gameType": "E"},
                        {"gamePk": 4, "gameType": "P"},
                        {"gamePk": 5, "gameType": "R"},
                    ],
                }]
            }).encode(),
            False,
        ),
    ]

    shape_results = []
    with pytest.MonkeyPatch.context() as mp:
        for name, raw, should_raise in shapes:
            mp.setattr(scheduler, "retry_urlopen", lambda *a, _raw=raw, **k: BytesIO(_raw))
            if should_raise:
                with pytest.raises(ValueError) as exc_info:
                    scheduler.fetch_schedule(DAY)
                shape_results.append({"case": name, "raised": "ValueError", "message": str(exc_info.value)[:160]})
            else:
                res = scheduler.fetch_schedule(DAY)
                shape_results.append({"case": name, "raised": None, "game_pks": [g["gamePk"] for g in res]})

    assert shape_results[-3]["game_pks"] == []
    assert shape_results[-2]["game_pks"] == []
    assert shape_results[-1]["game_pks"] == [5]

    caller_results = {}
    with TemporaryDirectory(prefix="wr_a5_") as tmp:
        tmp_path = Path(tmp)
        with pytest.MonkeyPatch.context() as outer_mp:
            offline_mod = _load_offline(tmp_path, outer_mp)

            # Caller 1: scheduler.run_day on persistent malformed response (4 systemd restarts 30s apart)
            with offline_mod.offline(offline_mod.instant(DAY, 10, 0)) as (world, patch):
                patch.setattr(
                    scheduler,
                    "retry_urlopen",
                    lambda *a, **k: BytesIO(b'{"message":"upstream gateway error"}'),
                )
                hb_mod = _load_check_heartbeat()
                samples = []
                t0 = datetime(2027, 3, 24, 13, 58, tzinfo=timezone.utc)
                _, _, samples = hb_mod.assess_churn(samples, 0, t0)

                dms_per_restart = []
                for _ in range(1, 5):
                    with pytest.raises(ValueError, match="dates"):
                        scheduler.run_day(DAY, world.config)
                    hb_stale, hb_reason = hb_mod.is_stale(
                        Path(world.config["orchestrator"]["heartbeat_path"]),
                        now=world.now.astimezone(timezone.utc),
                    )
                    assert hb_stale is False and hb_reason == "fresh running"
                    dms_per_restart.append(len(world.sends("dm")) + len(world.sends("health-dm")))
                    world.advance(30)

                churn, churn_reason, _ = hb_mod.assess_churn(
                    samples, 4, t0 + timedelta(minutes=5)
                )
                assert churn is True and "NRestarts +4 within 20 min" in churn_reason
                assert dms_per_restart == [0, 0, 0, 0]
                caller_results["run_day_and_systemd_churn"] = {
                    "dms_per_restart": dms_per_restart,
                    "heartbeat_fresh_each_cycle": True,
                    "churn_detected_at_5min": churn,
                    "churn_reason": churn_reason,
                }

            # Caller 2: scheduler._next_day_wakeup catches ValueError and hands off in 15 min
            with offline_mod.offline(offline_mod.instant(DAY, 22, 0)) as (world, patch):
                patch.setattr(
                    scheduler, "retry_urlopen", lambda *a, **k: BytesIO(b'{"message":"bad"}')
                )
                wake = scheduler._next_day_wakeup(DAY, world.config["scheduler"])
                assert wake == world.now + timedelta(minutes=15)
                caller_results["_next_day_wakeup"] = "caught_handoff_15min"

            # Caller 3: lineup_collect.collect_for_date & CLI bts data collect-lineup-times
            with offline_mod.offline(offline_mod.instant(DAY, 12, 0)) as (world, patch):
                out_dir = tmp_path / "lineups"
                out_dir.mkdir()
                patch.setattr(
                    scheduler, "retry_urlopen", lambda *a, **k: BytesIO(b'{"message":"bad"}')
                )
                with pytest.raises(ValueError, match="dates"):
                    lineup_collect.collect_for_date(DAY, out_dir)
                assert not (out_dir / f"{DAY}.jsonl").exists()
                r = CliRunner().invoke(
                    cli.cli, ["data", "collect-lineup-times", "--date", DAY, "--out-dir", str(out_dir)]
                )
                assert r.exit_code != 0 and isinstance(r.exception, ValueError)
                assert not (out_dir / f"{DAY}.jsonl").exists()
                caller_results["lineup_collect"] = "propagates_ValueError_no_jsonl_written"

            # Caller 4: day_outcome.gather & bts day-outcome CLI
            with offline_mod.offline(offline_mod.instant("2027-03-25", 8, 30)) as (world, patch):
                patch.setattr(
                    scheduler, "retry_urlopen", lambda *a, **k: BytesIO(b'{"message":"bad"}')
                )
                out = day_outcome.gather(DAY, world.picks, world.config, expected_username="u")
                assert out.state == "schedule_unverifiable" and not out.ok
                r = CliRunner().invoke(
                    cli.cli,
                    [
                        "day-outcome", "--date", DAY,
                        "--config", str(world.config_path),
                        "--picks-dir", str(world.picks),
                        "--dm-recipient", "offline.invalid",
                    ],
                    catch_exceptions=False,
                )
                assert r.exit_code == 1
                assert len(world.sends("dm")) == 1
                assert "SCHEDULE UNVERIFIABLE" in world.sends("dm")[0]
                caller_results["day_outcome"] = "schedule_unverifiable_exit_1_and_dm"

            # Independent reader 5: scheduler delivery-status guard (_void_committed_games)
            with offline_mod.offline(offline_mod.instant(DAY, 18, 25)) as (world, patch):
                world.dated(0)
                world.predictions = lambda: world.frame((0.99, 0.98))
                res = scheduler.run_single_check(DAY, [101, 102], set(), world.config, 0.03)
                daily = res["pick_result"].daily
                state = scheduler.SchedulerState(DAY, world.now.isoformat(), [], [], [], False, None, None, None)
                for raw_status in (b'{"message":"bad"}', b'{"dates":null}', b'{broken'):
                    patch.setattr(picks, "retry_urlopen", lambda *a, _r=raw_status, **k: BytesIO(_r))
                    assert scheduler._void_committed_games(daily, DAY) == {}
                assert scheduler._deliver_and_lock_pick(
                    daily, world.config, world.picks, state, DAY, "lineup", selection=res["selection"]
                ) is True
                caller_results["delivery_status_guard"] = "fails_open_delivers_unverified"

            # Independent reader 6: model.predict._fetch_game_slots
            from bts.model import predict as predict_mod

            model_cases = {}
            for label, body_bytes, expect_exc in [
                ("error_object", b'{"message":"bad"}', None),
                ("empty_dates", b'{"dates":[]}', None),
                ("dates_null", b'{"dates":null}', TypeError),
            ]:
                with pytest.MonkeyPatch.context() as pmp:
                    pmp.setattr(predict_mod, "urlopen", lambda *a, _b=body_bytes, **k: BytesIO(_b))
                    if expect_exc is not None:
                        with pytest.raises(expect_exc):
                            predict_mod._fetch_game_slots(DAY)
                        model_cases[label] = f"raises_{expect_exc.__name__}"
                    else:
                        slots = predict_mod._fetch_game_slots(DAY)
                        assert slots == []
                        model_cases[label] = "returns_empty_slots"
            caller_results["model_predict_fetch_game_slots"] = model_cases

    return {"shapes": shape_results, "callers": caller_results}


def test_probe_a5_verified():
    res = run_a5_probes()
    assert len(res["shapes"]) == 19


if __name__ == "__main__":
    out = run_a5_probes()
    (Path(__file__).resolve().parent / "probe_a5_summary.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))
