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


# --- Task 4: expiring-override precedence (auto default) ---
import datetime as _dt

_NOW = _dt.datetime(2026, 6, 6, 18, 0, 0, tzinfo=_dt.timezone.utc)


def _write_manual(state_dir, streak, *, expires_at=None, source_date="2026-06-06"):
    d = {"active_streak": streak, "best_streak": max(streak, 9),
         "source": "manual_cli", "source_date": source_date}
    if expires_at is not None:
        d["override_expires_at"] = expires_at
    (state_dir / "contest_streak.manual.json").write_text(json.dumps(d))


def _write_auto(state_dir, streak, source_date="2026-06-06"):
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": streak,
        "best_streak": max(streak, 9), "source": "mlb_bts_profile", "source_date": source_date}))


def test_auto_wins_over_legacy_manual(tmp_path):
    from bts.contest_state import load_contest_streak_state
    sd = tmp_path / "account_state"; sd.mkdir()
    _write_manual(sd, 7)                       # legacy, no override_expires_at
    _write_auto(sd, 0)
    st = load_contest_streak_state(tmp_path, now=_NOW)
    assert st.streak == 0 and st.path.name == "contest_streak.json"


def test_unexpired_override_wins_over_auto(tmp_path):
    from bts.contest_state import load_contest_streak_state
    sd = tmp_path / "account_state"; sd.mkdir()
    _write_manual(sd, 11, expires_at="2026-06-07T18:00:00Z")   # unexpired
    _write_auto(sd, 0)
    st = load_contest_streak_state(tmp_path, now=_NOW)
    assert st.streak == 11 and st.path.name == "contest_streak.manual.json"


def test_expired_override_ignored_auto_used(tmp_path):
    from bts.contest_state import load_contest_streak_state
    sd = tmp_path / "account_state"; sd.mkdir()
    _write_manual(sd, 11, expires_at="2026-06-05T18:00:00Z")   # expired before _NOW
    _write_auto(sd, 0)
    st = load_contest_streak_state(tmp_path, now=_NOW)
    assert st.streak == 0 and st.path.name == "contest_streak.json"


def test_legacy_manual_fallback_when_no_auto(tmp_path):
    from bts.contest_state import load_contest_streak_state
    sd = tmp_path / "account_state"; sd.mkdir()
    _write_manual(sd, 0)                       # legacy hotfix, no auto yet
    st = load_contest_streak_state(tmp_path, now=_NOW)
    assert st is not None and st.streak == 0 and st.path.name == "contest_streak.manual.json"


def test_state_file_rejects_bool_streak(tmp_path):
    from bts.contest_state import load_contest_streak_state, ContestStateError
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "active_streak": True, "best_streak": 9, "source": "x", "source_date": "2026-06-06"}))
    try:
        load_contest_streak_state(tmp_path)
        assert False, "bool streak should be rejected"
    except ContestStateError:
        pass
