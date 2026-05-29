import json


def _write_pick(path, result):
    path.write_text(json.dumps({
        "date": path.stem,
        "run_time": f"{path.stem}T12:00:00+00:00",
        "result": result,
        "pick": {},
        "double_down": None,
    }))


def test_model_state_used_when_no_contest_state(tmp_path):
    from bts.contest_state import load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({
        "streak": 4,
        "saver_available": True,
    }))

    state = load_decision_streak_state(tmp_path)

    assert state.streak == 4
    assert state.saver_available is True
    assert state.allow_double is True
    assert state.status == "model_only"


def test_required_contest_state_missing_raises(tmp_path):
    from bts.contest_state import ContestStateError, load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({
        "streak": 4,
        "saver_available": True,
    }))

    try:
        load_decision_streak_state(tmp_path, require_contest_state=True)
    except ContestStateError as exc:
        assert "required but missing" in str(exc)
    else:
        raise AssertionError("expected required contest state to fail closed")


def test_fresh_contest_state_drives_live_decision(tmp_path):
    from bts.contest_state import load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({
        "streak": 4,
        "saver_available": True,
    }))
    _write_pick(tmp_path / "2026-05-28.json", "hit")
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7,
        "best_streak": 7,
        "source": "manual_screenshot",
        "source_date": "2026-05-29",
    }))

    state = load_decision_streak_state(tmp_path)

    assert state.streak == 7
    assert state.model_streak == 4
    assert state.saver_available is False  # unknown contest saver is conservative
    assert state.allow_double is True
    assert state.status == "fresh"


def test_stale_contest_state_freezes_higher_streak_and_disables_double(tmp_path):
    from bts.contest_state import load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({
        "streak": 5,
        "saver_available": True,
    }))
    _write_pick(tmp_path / "2026-05-28.json", "hit")
    _write_pick(tmp_path / "2026-05-29.json", "hit")
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7,
        "source": "manual_screenshot",
        "source_date": "2026-05-28",
        "saver_available": True,
    }))

    state = load_decision_streak_state(tmp_path)

    assert state.streak == 7
    assert state.saver_available is False
    assert state.contest_saver_available is True
    assert state.allow_double is False
    assert state.status == "stale"


def test_stale_contest_state_never_lowers_model_streak(tmp_path):
    from bts.contest_state import load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({
        "streak": 12,
        "saver_available": True,
    }))
    _write_pick(tmp_path / "2026-05-29.json", "hit")
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7,
        "source_date": "2026-05-28",
    }))

    state = load_decision_streak_state(tmp_path)

    assert state.streak == 12
    assert state.allow_double is False
