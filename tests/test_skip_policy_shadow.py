"""Tests for the skip-policy shadow logger (bts/skip_policy_shadow.py).

Ground truth is a decision MARKER the live pick path writes at each genuine MDP skip
(`record_mdp_skip_decision`, from strategy.select_pick), recording the EXECUTABLE declined
candidate. The shadow reads markers, logs the counterfactual "pick-the-band" single, and
reconciles the realized outcome — accumulating the band hit rate vs the ~0.744 breakeven
(docs/audit/2026-06-20-skip-policy-shadow.md). Pure logic + injected hit checker; no MLB API.
"""
import json
from datetime import datetime, timezone

from bts.skip_policy_shadow import (
    record_mdp_skip_decision,
    load_skip_decision,
    skip_decision_path,
    build_divergent_record,
    record_skip_from_marker,
    record_pending_skips,
    prune_superseded,
    reconcile_decision,
    reconcile_pending,
    build_skip_policy_shadow_status,
    write_status,
    decision_path,
)

UTC = timezone.utc


def _cand(bid=1, p=0.78, *, name="Batter", team="NYM", game_pk=100, pitcher="Pitcher"):
    return {"batter_id": bid, "batter_name": name, "team": team,
            "game_pk": game_pk, "pitcher_name": pitcher, "p_game_hit": p}


# ---- decision marker (the cascade-written ground-truth seam) ----

def test_record_and_load_skip_marker(tmp_path):
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10,
                             saver_available=True, now=datetime(2026, 6, 18, tzinfo=UTC))
    assert skip_decision_path("2026-06-18", tmp_path).exists()
    m = load_skip_decision("2026-06-18", tmp_path)
    assert m["schema_version"] == "bts_mdp_skip_decision_v1"
    assert m["action"] == "skip"
    assert m["streak"] == 10
    assert m["candidate"]["batter_id"] == 1
    assert m["candidate"]["p_game_hit"] == 0.75


def test_load_skip_marker_missing_is_none(tmp_path):
    assert load_skip_decision("2026-01-01", tmp_path) is None


# ---- build_divergent_record / record_skip_from_marker ----

def test_build_divergent_record_uses_marker_candidate():
    marker = {"candidate": _cand(7, 0.74, name="Arraez"), "streak": 10, "saver_available": True}
    r = build_divergent_record("2026-06-19", marker)
    assert r["deployed_action"] == "skip"
    assert r["shadow_action"] == "single"
    assert r["divergent"] is True
    assert r["rank1"]["batter_id"] == 7
    assert r["rank1"]["batter_name"] == "Arraez"
    assert r["shadow_pick_result"] is None


def test_record_skip_from_marker_writes_decision(tmp_path):
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10)
    rec = record_skip_from_marker("2026-06-18", tmp_path)
    assert rec is not None
    saved = json.loads(decision_path("2026-06-18", tmp_path).read_text())
    assert saved["divergent"] is True
    assert saved["rank1"]["batter_id"] == 1
    assert saved["schema_version"] == "bts_skip_policy_shadow_v1"


def test_record_skip_from_marker_none_without_marker(tmp_path):
    assert record_skip_from_marker("2026-06-18", tmp_path) is None
    assert not decision_path("2026-06-18", tmp_path).exists()


def _daily():
    """A minimal DailyPick-like object (has a pick); delivery is decided by the injected delivered_fn."""
    return type("D", (), {"pick": object(), "double_down": None})()


def test_skip_dropped_when_pick_delivered(tmp_path):
    # marker present, but production DURABLY DELIVERED a pick -> final = pick -> not a divergence.
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10)
    rec = record_skip_from_marker("2026-06-18", tmp_path,
                                  pick_loader=lambda d, p: _daily(), delivered_fn=lambda daily: True)
    assert rec is None
    assert not decision_path("2026-06-18", tmp_path).exists()


def test_skip_recorded_when_pick_not_delivered(tmp_path):
    # an UNDELIVERED (preview/provisional) pick file exists, but it was never delivered -> skip wins.
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10)
    rec = record_skip_from_marker("2026-06-18", tmp_path,
                                  pick_loader=lambda d, p: _daily(), delivered_fn=lambda daily: False)
    assert rec is not None
    assert decision_path("2026-06-18", tmp_path).exists()


def test_record_skip_from_marker_idempotent_preserves_resolved(tmp_path):
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10)
    record_skip_from_marker("2026-06-18", tmp_path)
    p = decision_path("2026-06-18", tmp_path)
    rec = json.loads(p.read_text()); rec["shadow_pick_result"] = "hit"; p.write_text(json.dumps(rec))
    assert record_skip_from_marker("2026-06-18", tmp_path) is None       # don't clobber
    assert json.loads(p.read_text())["shadow_pick_result"] == "hit"


def test_prune_removes_when_pick_delivered(tmp_path):
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10)
    record_skip_from_marker("2026-06-18", tmp_path)                      # recorded (nothing delivered yet)
    removed = prune_superseded(tmp_path, pick_loader=lambda d, p: _daily(), delivered_fn=lambda daily: True)
    assert removed == ["2026-06-18"]
    assert not decision_path("2026-06-18", tmp_path).exists()


def test_prune_keeps_when_pick_not_delivered(tmp_path):
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10)
    record_skip_from_marker("2026-06-18", tmp_path)
    kept = prune_superseded(tmp_path, pick_loader=lambda d, p: _daily(), delivered_fn=lambda daily: False)
    assert kept == []
    assert decision_path("2026-06-18", tmp_path).exists()


# ---- record_pending_skips (backfill from markers) ----

def test_record_pending_skips_records_only_missing_in_window(tmp_path):
    record_mdp_skip_decision("2026-06-17", tmp_path, candidate=_cand(1, 0.75), streak=10)
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(2, 0.78), streak=10)
    record_skip_from_marker("2026-06-18", tmp_path)          # 06-18 already recorded
    recorded = record_pending_skips(tmp_path, lookback_days=10, now=datetime(2026, 6, 18, tzinfo=UTC))
    assert recorded == ["2026-06-17"]
    assert decision_path("2026-06-17", tmp_path).exists()


def test_record_pending_skips_respects_lookback(tmp_path):
    record_mdp_skip_decision("2026-05-01", tmp_path, candidate=_cand(1, 0.75), streak=10)
    recorded = record_pending_skips(tmp_path, lookback_days=10, now=datetime(2026, 6, 18, tzinfo=UTC))
    assert recorded == []                                    # too old
    assert not decision_path("2026-05-01", tmp_path).exists()


# ---- reconcile ----

def _rec(divergent, result, date="2026-06-18"):
    return {"divergent": divergent, "shadow_pick_result": result, "date": date, "rank1": _cand()}


def test_reconcile_pending_stays_pending():
    rec = _rec(True, None)
    assert reconcile_decision(rec, hit_checker=lambda r1: None,
                              now=datetime(2026, 6, 18, tzinfo=UTC)) is False
    assert rec["shadow_pick_result"] is None


def test_reconcile_stale_pending_voids():
    rec = _rec(True, None, date="2026-06-18")
    assert reconcile_decision(rec, hit_checker=lambda r1: None,
                              now=datetime(2026, 6, 25, tzinfo=UTC), stale_after_days=3) is True
    assert rec["shadow_pick_result"] == "void"


def test_reconcile_sets_hit_and_is_idempotent():
    rec = _rec(True, None)
    assert reconcile_decision(rec, hit_checker=lambda r1: "hit",
                              now=datetime(2026, 6, 18, tzinfo=UTC)) is True
    assert rec["shadow_pick_result"] == "hit"
    assert reconcile_decision(rec, hit_checker=lambda r1: "miss",
                              now=datetime(2026, 6, 18, tzinfo=UTC)) is False  # idempotent


def test_reconcile_permanent_error_voids_when_stale():
    rec = _rec(True, None, date="2026-06-18")

    def boom(r1):
        raise RuntimeError("permanent 404")

    assert reconcile_decision(rec, hit_checker=boom,
                              now=datetime(2026, 6, 25, tzinfo=UTC), stale_after_days=3) is True
    assert rec["shadow_pick_result"] == "void"


def test_reconcile_transient_error_recent_stays_pending():
    rec = _rec(True, None, date="2026-06-18")

    def boom(r1):
        raise RuntimeError("transient")

    assert reconcile_decision(rec, hit_checker=boom,
                              now=datetime(2026, 6, 18, tzinfo=UTC)) is False
    assert rec["shadow_pick_result"] is None


def test_reconcile_pending_continues_on_error(tmp_path):
    record_mdp_skip_decision("2026-06-17", tmp_path, candidate=_cand(1, 0.75), streak=10)
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(2, 0.78), streak=10)
    record_skip_from_marker("2026-06-17", tmp_path)
    record_skip_from_marker("2026-06-18", tmp_path)

    def flaky(r1):
        if r1["batter_id"] == 1:
            raise RuntimeError("transient")
        return "hit"

    n = reconcile_pending(tmp_path, hit_checker=flaky, now=datetime(2026, 6, 18, tzinfo=UTC))
    assert n == 1
    assert json.loads(decision_path("2026-06-17", tmp_path).read_text())["shadow_pick_result"] is None
    assert json.loads(decision_path("2026-06-18", tmp_path).read_text())["shadow_pick_result"] == "hit"


# ---- status ----

def _drec(result):
    return {"divergent": True, "shadow_pick_result": result, "date": "2026-06-18", "rank1": _cand()}


def test_status_counts_rate_and_void_excluded():
    recs = [_drec("hit"), _drec("miss"), _drec("hit"), _drec(None), _drec("void")]
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744, min_divergent_days=2)
    assert s["schema_version"] == "bts_skip_policy_shadow_status_v1"
    assert s["counts"]["divergent_days"] == 5
    assert s["counts"]["pending"] == 1
    assert s["counts"]["void"] == 1
    assert s["shadow_band_hit_rate"]["resolved"] == 3   # void excluded
    assert s["shadow_band_hit_rate"]["hits"] == 2
    assert abs(s["shadow_band_hit_rate"]["rate"] - 2 / 3) < 1e-9


def test_verdict_insufficient_then_below_then_above():
    assert build_skip_policy_shadow_status([_drec("hit")] * 5, min_divergent_days=30
                                           )["shadow_band_hit_rate"]["verdict"] == "insufficient_n"
    below = [_drec("hit")] * 10 + [_drec("miss")] * 10      # 0.50; Wilson hi < 0.744
    assert build_skip_policy_shadow_status(below, breakeven_p=0.744, min_divergent_days=10
                                           )["shadow_band_hit_rate"]["verdict"] == "below_breakeven"
    above = [_drec("hit")] * 19 + [_drec("miss")]           # 0.95; Wilson lo > 0.744
    assert build_skip_policy_shadow_status(above, breakeven_p=0.744, min_divergent_days=10
                                           )["shadow_band_hit_rate"]["verdict"] == "above_breakeven"


# ---- end-to-end: cascade marker -> record -> reconcile -> status ----

def test_end_to_end_marker_to_status(tmp_path):
    record_mdp_skip_decision("2026-06-18", tmp_path, candidate=_cand(1, 0.75), streak=10,
                             now=datetime(2026, 6, 18, tzinfo=UTC))
    record_pending_skips(tmp_path, lookback_days=10, now=datetime(2026, 6, 19, tzinfo=UTC))
    reconcile_pending(tmp_path, hit_checker=lambda r1: "miss", now=datetime(2026, 6, 19, tzinfo=UTC))
    out = tmp_path / "status.json"
    write_status(tmp_path, out, generated_at="2026-06-20T00:00:00Z", git_commit="abc")
    s = json.loads(out.read_text())
    assert s["counts"]["divergent_days"] == 1
    assert s["shadow_band_hit_rate"]["resolved"] == 1
    assert s["shadow_band_hit_rate"]["hits"] == 0
