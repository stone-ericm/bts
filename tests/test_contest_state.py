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


def test_lagged_contest_uses_contest_value_keeps_doubles(tmp_path):
    # model 9 > contest 7 so streak==7 also proves no max() inflation (old max(9,7)=9); 1 pick behind -> lagged.
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 9, "saver_available": True}))
    _write_pick(tmp_path / "2026-05-28.json", "hit")
    _write_pick(tmp_path / "2026-05-29.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7, "source": "manual_screenshot",
        "source_date": "2026-05-28", "saver_available": True}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 7
    assert state.allow_double is True
    assert state.status == "lagged"
    assert state.saver_available is True   # contest saver is known (True) here


def test_lagged_contest_model_does_not_inflate(tmp_path):
    # Was test_stale_..._never_lowers_model_streak: model 12, 1 pick behind -> lagged; must NOT raise the streak.
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 12, "saver_available": True}))
    _write_pick(tmp_path / "2026-05-29.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7, "source_date": "2026-05-28"}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 7           # contest value; model 12 cannot inflate it
    assert state.model_streak == 12
    assert state.allow_double is True
    assert state.status == "lagged"


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


# --- Hardening: missing source_date must be conservative, not fail-open ---

def test_missing_source_date_is_conservative_not_fail_open(tmp_path):
    """Contest file with NO source_date and NO resolved picks must NOT fail open
    to 'fresh'. Without a source_date we cannot verify freshness, so the safe
    answer is stale (last-confirmed); doubles stay enabled (Phase 1)."""
    from bts.contest_state import load_contest_streak_state, contest_state_is_fresh
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 3,
        "best_streak": 9, "source": "mlb_bts_profile"}))  # no source_date
    st = load_contest_streak_state(tmp_path)
    assert st.source_date is None
    assert contest_state_is_fresh(st, tmp_path) is False


def test_missing_source_date_is_stale_but_keeps_doubles(tmp_path):
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 3, "saver_available": True}))
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 3,
        "best_streak": 9, "source": "mlb_bts_profile"}))  # no source_date
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 3
    assert state.allow_double is True
    assert state.status == "stale"


def test_model_never_inflates_decision_streak(tmp_path):
    """The 2026-06-17 incident: model replay = 10 (missed 6/11 entry), real MLB = 8.
    The decision streak must be 8, NOT max(10, 8)=10, and doubles must stay enabled."""
    from bts.contest_state import load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({"streak": 10, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-16.json", "hit")          # latest resolved local pick
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 8,
        "best_streak": 9, "source": "mlb_bts_profile", "source_date": "2026-06-15"}))

    state = load_decision_streak_state(tmp_path)

    assert state.streak == 8            # real MLB streak, NOT the inflated model 10
    assert state.model_streak == 10
    assert state.allow_double is True   # no staleness freeze
    assert state.status == "lagged"     # source_date 06-15 is 1 settled pick behind 06-16


def test_two_pick_gap_is_stale(tmp_path):
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 9, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-16.json", "hit")
    _write_pick(tmp_path / "2026-06-17.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "active_streak": 8, "best_streak": 9, "source": "mlb_bts_profile",
        "source_date": "2026-06-15"}))   # 2 settled picks behind -> stale
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 8 and state.status == "stale" and state.allow_double is True


def test_unconfirmed_local_miss_marks_stale(tmp_path):
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 0, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-17.json", "miss")   # local reset MLB hasn't posted
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "active_streak": 8, "best_streak": 9, "source": "mlb_bts_profile",
        "source_date": "2026-06-16"}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 8            # last confirmed; Phase 1 doesn't lower it
    assert state.status == "stale"      # but flags the stale-high risk


def test_unexpired_override_wins_over_auto_through_decision(tmp_path):
    # Phase 1 P1.4: an UNEXPIRED operator override beats the auto file AND drives the
    # decision (8), proving precedence flows through load_decision_streak_state.
    import datetime as _dt2
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 10, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-16.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({            # auto says 5
        "active_streak": 5, "best_streak": 9, "source": "mlb_bts_profile",
        "source_date": "2026-06-16"}))
    (sd / "contest_streak.manual.json").write_text(json.dumps({     # override says 8, unexpired
        "active_streak": 8, "best_streak": 9, "source": "manual_cli",
        "source_date": "2026-06-16", "override_expires_at": "2026-06-18T00:00:00Z"}))
    now = _dt2.datetime(2026, 6, 17, 12, 0, tzinfo=_dt2.timezone.utc)
    state = load_decision_streak_state(tmp_path, now=now)
    assert state.streak == 8            # override (8) wins over auto (5)
    assert state.source == "contest" and state.allow_double is True
    assert state.status == "fresh"      # source_date 06-16 covers latest pick 06-16


def test_stale_unknown_saver_is_conservative_even_when_streaks_agree(tmp_path):
    # Saver: equal streaks don't prove the local model saver matches the real account
    # when stale (manual entry can diverge), so stay conservative -> False.
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 8, "saver_available": True}))
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 8,
        "best_streak": 9, "source": "mlb_bts_profile"}))  # no source_date -> stale, saver unknown
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 8 and state.status == "stale"
    assert state.saver_available is False   # conservative, NOT the model's True
