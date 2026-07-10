"""Tests for the Streak Saver manual flag (bts.saver_state)."""
import json
from datetime import date

from bts.saver_state import load_saver_state, SaverState, season_for


def _write(picks_dir, obj):
    d = picks_dir / "account_state"
    d.mkdir(parents=True, exist_ok=True)
    (d / "saver_state.json").write_text(json.dumps(obj))


def test_missing_file_is_uninitialized(tmp_path):
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "uninitialized" and s.is_available is False


def test_valid_active_for_matching_season(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "active", "source": "manual_init"})
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "active" and s.is_available is True


def test_wrong_season_is_uninitialized_not_not_earned(tmp_path):
    _write(tmp_path, {"season": 2025, "state": "active"})
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "uninitialized"   # stale -> fail-closed, NOT not_earned
    assert s.season == 2025             # but the stale season is preserved (health distinguishes)


def test_invalid_state_or_bad_json_is_uninitialized(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "bogus"})
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"
    (tmp_path / "account_state" / "saver_state.json").write_text("{not json")
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"
    for bad in ("[]", "123", '"active"',                                  # not an object
                '{"season": 2026, "state": []}', '{"season": 2026, "state": {}}'):  # unhashable state
        (tmp_path / "account_state" / "saver_state.json").write_text(bad)
        assert load_saver_state(tmp_path, season=2026).state == "uninitialized"


def test_not_earned_and_used_not_available(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "not_earned"})
    assert load_saver_state(tmp_path, season=2026).is_available is False
    _write(tmp_path, {"season": 2026, "state": "used"})
    assert load_saver_state(tmp_path, season=2026).is_available is False


def test_season_for_uses_source_date_year_else_now(tmp_path):
    assert season_for(date(2026, 6, 18), now_year=2027) == 2026
    assert season_for(None, now_year=2027) == 2027


# --- transitions + fetch-path auto-earn ---

from bts.saver_state import transition_saver_state, maybe_auto_earn_saver


def test_transition_guarded_by_expected_prior(tmp_path):
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active", season=2026, source="t")
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="used", season=2026, source="t") is True
    assert load_saver_state(tmp_path, season=2026).state == "used"
    # wrong expected_prior -> no-op
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="used", season=2026, source="t") is False


def test_invalid_transition_rejected_unless_forced(tmp_path):
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active", season=2026, source="t")
    # active -> not_earned is NOT an allowed transition (guards a scripted/cross-page POST)
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="not_earned", season=2026, source="t") is False
    assert load_saver_state(tmp_path, season=2026).state == "active"
    # ...but --force (CLI break-glass) may override
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="not_earned", season=2026, source="t", force=True) is True


def test_auto_earn_inits_not_earned_below_10(tmp_path):
    maybe_auto_earn_saver(tmp_path, best_streak=8, season=2026)
    assert load_saver_state(tmp_path, season=2026).state == "not_earned"


def test_auto_earn_promotes_not_earned_to_active_at_10(tmp_path):
    maybe_auto_earn_saver(tmp_path, best_streak=8, season=2026)    # -> not_earned
    maybe_auto_earn_saver(tmp_path, best_streak=10, season=2026)   # -> active
    assert load_saver_state(tmp_path, season=2026).state == "active"


def test_auto_earn_will_not_init_active_from_uninitialized_at_10(tmp_path):
    # fail-closed: a fresh file at best_streak>=10 must NOT become active automatically
    maybe_auto_earn_saver(tmp_path, best_streak=12, season=2026)
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"


def test_auto_earn_never_overwrites_used(tmp_path):
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="used", season=2026, source="t")
    maybe_auto_earn_saver(tmp_path, best_streak=14, season=2026)
    assert load_saver_state(tmp_path, season=2026).state == "used"


def test_concurrent_transitions_from_same_prior_serialize(tmp_path):
    # The lock makes the expected_prior guard race-safe: of many concurrent writers from the
    # same prior, EXACTLY ONE wins (no lost update) -- the rest re-read the new state -> False.
    import threading
    (tmp_path / "account_state").mkdir(parents=True)
    results = []

    def worker(new_state):
        results.append(transition_saver_state(tmp_path, expected_prior="uninitialized",
                                              new_state=new_state, season=2026, source="t"))

    threads = [threading.Thread(target=worker, args=(s,))
               for s in (["active", "used", "not_earned"] * 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 1


# --- F7 (2026-07-09 audit, accepted-risk + detective control): every
# transition attempt through the guarded writer lands in an append-only
# audit trail (inside data/picks -> covered by the F5 ops backup), with
# the requesting peer recorded when the mutation came over the network.

def test_transition_appends_audit_line_with_peer(tmp_path):
    import json as _json
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active",
                           season=2026, source="t")
    transition_saver_state(tmp_path, expected_prior="active", new_state="used",
                           season=2026, source="dashboard", peer="100.64.1.2")
    log = tmp_path / "account_state" / "saver_transitions.jsonl"
    lines = [_json.loads(l) for l in log.read_text().splitlines()]
    assert lines[-1]["source"] == "dashboard"
    assert lines[-1]["peer"] == "100.64.1.2"
    assert lines[-1]["outcome"] == "written"
    assert lines[-1]["expected_prior"] == "active"
    assert lines[-1]["new_state"] == "used"
    assert lines[-1]["ts"]


def test_rejected_transition_is_also_audited(tmp_path):
    import json as _json
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active",
                           season=2026, source="t")
    ok = transition_saver_state(tmp_path, expected_prior="used", new_state="active",
                                season=2026, source="dashboard", peer="100.64.9.9")
    assert ok is False
    log = tmp_path / "account_state" / "saver_transitions.jsonl"
    lines = [_json.loads(l) for l in log.read_text().splitlines()]
    assert lines[-1]["outcome"] == "rejected_state_mismatch"
    assert lines[-1]["peer"] == "100.64.9.9"


def test_disallowed_transition_not_audited_as_written(tmp_path):
    import json as _json
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active",
                           season=2026, source="t")
    ok = transition_saver_state(tmp_path, expected_prior="active", new_state="not_earned",
                                season=2026, source="dashboard", peer="100.64.9.9")
    assert ok is False
    log = tmp_path / "account_state" / "saver_transitions.jsonl"
    lines = [_json.loads(l) for l in log.read_text().splitlines()]
    assert lines[-1]["outcome"] == "rejected_disallowed"
