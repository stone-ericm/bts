import importlib.util
from pathlib import Path
from unittest.mock import patch


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "shadow_predict_once.py"
    spec = importlib.util.spec_from_file_location("shadow_predict_once", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shadow_unit_sources_env_without_hardcoding_model_flags():
    unit_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "systemd"
        / "bts-shadow-prediction.service"
    )
    body = unit_path.read_text()

    assert "EnvironmentFile=/home/bts/projects/bts/.env" in body
    assert (
        'Environment="PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin"'
        in body
    )
    assert "BTS_LGBM_DETERMINISTIC" not in body
    assert "BTS_LGBM_RANDOM_STATE" not in body
    assert "OMP_NUM_THREADS" not in body


def _write_config(tmp_path, picks_dir):
    config_path = tmp_path / "orchestrator.toml"
    config_path.write_text(
        "\n".join([
            "[orchestrator]",
            f'picks_dir = "{picks_dir.as_posix()}"',
            f'data_dir = "{(tmp_path / "data").as_posix()}"',
            f'models_dir = "{(tmp_path / "models").as_posix()}"',
            f'heartbeat_path = "{(tmp_path / ".heartbeat").as_posix()}"',
            "",
        ])
    )
    return config_path


def _save_state(picks_dir, *, analytics_jobs=None):
    from bts.scheduler import SchedulerState, save_state

    state = SchedulerState(
        date="2026-05-16",
        schedule_fetched_at="2026-05-16T10:00:00-04:00",
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=True,
        pick_locked_at="2026-05-16T12:00:00-04:00",
        result_status=None,
        next_wakeup=None,
        analytics_jobs=analytics_jobs or {},
    )
    save_state(state, picks_dir)


def _save_locked_pick(picks_dir, *, delivered=True):
    from bts.picks import DailyPick, Pick, save_pick

    pick = Pick(
        batter_name="Prod Pick",
        batter_id=1,
        team="TB",
        lineup_position=1,
        pitcher_name="Pitcher",
        pitcher_id=2,
        p_game_hit=0.72,
        flags=[],
        projected_lineup=False,
        game_pk=100,
        game_time="2026-05-16T23:10:00Z",
    )
    daily = DailyPick(
        date="2026-05-16",
        run_time="2026-05-16T16:00:00+00:00",
        pick=pick,
        double_down=None,
        runner_up=None,
        notification_sent=delivered,
        notification_channel="bluesky_dm",
        notification_id="dm-1" if delivered else None,
    )
    save_pick(daily, picks_dir)


def test_shadow_worker_calls_existing_runner_with_dispatched_allowed(tmp_path):
    from bts.scheduler import load_state

    module = _load_script()
    picks_dir = tmp_path / "picks"
    _save_state(
        picks_dir,
        analytics_jobs={"shadow": {"status": "dispatched", "reason": "trigger_queued"}},
    )
    _save_locked_pick(picks_dir)
    config_path = _write_config(tmp_path, picks_dir)

    with patch("bts.scheduler._run_shadow_prediction") as mock_shadow:
        exit_code = module.run_once(
            config_path,
            "2026-05-16",
            "bts-shadow-prediction.service",
        )

    assert exit_code == 0
    mock_shadow.assert_called_once()
    args, kwargs = mock_shadow.call_args
    assert args[1:] == ("2026-05-16", "Prod Pick")
    assert kwargs["allow_prior_dispatched"] is True
    assert kwargs["attempt_reason"] == "shadow_unit_attempt"
    assert kwargs["unit"] == "bts-shadow-prediction.service"
    assert load_state("2026-05-16", picks_dir) is not None


def test_shadow_worker_accepts_private_locked_state(tmp_path):
    module = _load_script()
    picks_dir = tmp_path / "picks"
    _save_state(picks_dir)
    _save_locked_pick(picks_dir, delivered=False)
    config_path = _write_config(tmp_path, picks_dir)

    with patch("bts.scheduler._run_shadow_prediction") as mock_shadow:
        exit_code = module.run_once(
            config_path,
            "2026-05-16",
            "bts-shadow-prediction.service",
        )

    assert exit_code == 0
    mock_shadow.assert_called_once()


def test_shadow_worker_marks_missing_pick_failed(tmp_path):
    from bts.scheduler import load_state

    module = _load_script()
    picks_dir = tmp_path / "picks"
    _save_state(picks_dir)
    config_path = _write_config(tmp_path, picks_dir)

    exit_code = module.run_once(
        config_path,
        "2026-05-16",
        "bts-shadow-prediction.service",
    )

    assert exit_code == 1
    state = load_state("2026-05-16", picks_dir)
    assert state.analytics_jobs["shadow"]["status"] == "failed"
    assert state.analytics_jobs["shadow"]["reason"] == "production_pick_missing"
    assert state.analytics_jobs["shadow"]["unit"] == "bts-shadow-prediction.service"
