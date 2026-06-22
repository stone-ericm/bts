"""Tests for the skip-policy shadow logger (bts/skip_policy_shadow.py).

Ground truth is `decision.json` (bts_daily_decision_v1) written by the scheduler at each
genuine MDP skip (action="skip", source="mdp"). The shadow reads those records, logs the
counterfactual "pick-the-band" single, and reconciles the realized outcome — accumulating the
band hit rate vs the ~0.744 breakeven (docs/audit/2026-06-20-skip-policy-shadow.md).
Pure logic + injected hit checker; no MLB API.
"""
import json
from datetime import datetime, timezone

from bts.daily_decision import write_decision
from bts.skip_policy_shadow import (
    build_divergent_record,
    record_skip_from_decision,
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


def _write_mdp_skip(date, tmp_path, bid=1, p=0.75, streak=10):
    """Helper: write an MDP skip decision.json for a date."""
    write_decision(date, tmp_path, action="skip", source="mdp", primary=_cand(bid, p),
                   streak=streak, delivery_status="not_applicable", scoreable=False)


# ---- build_divergent_record ----

def test_build_divergent_record_uses_decision_primary():
    dec = {"primary": _cand(7, 0.74, name="Arraez"), "streak": 10, "saver_available": True}
    r = build_divergent_record("2026-06-19", dec)
    assert r["deployed_action"] == "skip"
    assert r["shadow_action"] == "single"
    assert r["divergent"] is True
    assert r["rank1"]["batter_id"] == 7
    assert r["rank1"]["batter_name"] == "Arraez"
    assert r["shadow_pick_result"] is None


# ---- record_skip_from_decision ----

def test_records_mdp_skip_decision(tmp_path):
    write_decision("2026-06-18", tmp_path, action="skip", source="mdp", primary=_cand(1, 0.75),
                   streak=10, delivery_status="not_applicable", scoreable=False)
    rec = record_skip_from_decision("2026-06-18", tmp_path)
    assert rec is not None and rec["divergent"] is True and rec["rank1"]["batter_id"] == 1


def test_ignores_pick_and_heuristic_skip(tmp_path):
    write_decision("2026-06-19", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)
    assert record_skip_from_decision("2026-06-19", tmp_path) is None
    write_decision("2026-06-20", tmp_path, action="skip", source="heuristic", primary=_cand(),
                   delivery_status="not_applicable", scoreable=False)
    assert record_skip_from_decision("2026-06-20", tmp_path) is None


def test_record_skip_from_decision_none_without_decision(tmp_path):
    assert record_skip_from_decision("2026-06-18", tmp_path) is None
    assert not decision_path("2026-06-18", tmp_path).exists()


def test_record_skip_from_decision_writes_shadow_record(tmp_path):
    _write_mdp_skip("2026-06-18", tmp_path, bid=1, p=0.75, streak=10)
    rec = record_skip_from_decision("2026-06-18", tmp_path)
    assert rec is not None
    saved = json.loads(decision_path("2026-06-18", tmp_path).read_text())
    assert saved["divergent"] is True
    assert saved["rank1"]["batter_id"] == 1
    assert saved["schema_version"] == "bts_skip_policy_shadow_v1"


def test_record_skip_from_decision_idempotent_preserves_resolved(tmp_path):
    _write_mdp_skip("2026-06-18", tmp_path)
    record_skip_from_decision("2026-06-18", tmp_path)
    p = decision_path("2026-06-18", tmp_path)
    rec = json.loads(p.read_text()); rec["shadow_pick_result"] = "hit"; p.write_text(json.dumps(rec))
    assert record_skip_from_decision("2026-06-18", tmp_path) is None       # don't clobber
    assert json.loads(p.read_text())["shadow_pick_result"] == "hit"


# ---- prune_superseded ----

def test_prune_drops_record_when_decision_no_longer_mdp_skip(tmp_path):
    write_decision("2026-06-18", tmp_path, action="skip", source="mdp", primary=_cand(),
                   delivery_status="not_applicable", scoreable=False)
    record_skip_from_decision("2026-06-18", tmp_path)
    # decision flips to a committed pick (e.g. late delivery)
    write_decision("2026-06-18", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)
    assert prune_superseded(tmp_path) == ["2026-06-18"]
    assert not decision_path("2026-06-18", tmp_path).exists()


def test_prune_keeps_genuine_mdp_skip(tmp_path):
    _write_mdp_skip("2026-06-18", tmp_path)
    record_skip_from_decision("2026-06-18", tmp_path)
    kept = prune_superseded(tmp_path)
    assert kept == []
    assert decision_path("2026-06-18", tmp_path).exists()


# ---- record_pending_skips ----

def test_record_pending_skips_records_only_missing_in_window(tmp_path):
    _write_mdp_skip("2026-06-17", tmp_path, bid=1)
    _write_mdp_skip("2026-06-18", tmp_path, bid=2)
    record_skip_from_decision("2026-06-18", tmp_path)          # 06-18 already recorded
    recorded = record_pending_skips(tmp_path, lookback_days=10, now=datetime(2026, 6, 18, tzinfo=UTC))
    assert recorded == ["2026-06-17"]
    assert decision_path("2026-06-17", tmp_path).exists()


def test_record_pending_skips_respects_lookback(tmp_path):
    _write_mdp_skip("2026-05-01", tmp_path, bid=1)
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
    _write_mdp_skip("2026-06-17", tmp_path, bid=1)
    _write_mdp_skip("2026-06-18", tmp_path, bid=2)
    record_skip_from_decision("2026-06-17", tmp_path)
    record_skip_from_decision("2026-06-18", tmp_path)

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


# ---- end-to-end: decision.json -> record -> reconcile -> status ----

def test_end_to_end_decision_to_status(tmp_path):
    write_decision("2026-06-18", tmp_path, action="skip", source="mdp", primary=_cand(1, 0.75),
                   streak=10, delivery_status="not_applicable", scoreable=False,
                   now=datetime(2026, 6, 18, tzinfo=UTC))
    record_pending_skips(tmp_path, lookback_days=10, now=datetime(2026, 6, 19, tzinfo=UTC))
    reconcile_pending(tmp_path, hit_checker=lambda r1: "miss", now=datetime(2026, 6, 19, tzinfo=UTC))
    out = tmp_path / "status.json"
    write_status(tmp_path, out, generated_at="2026-06-20T00:00:00Z", git_commit="abc")
    s = json.loads(out.read_text())
    assert s["counts"]["divergent_days"] == 1
    assert s["shadow_band_hit_rate"]["resolved"] == 1
    assert s["shadow_band_hit_rate"]["hits"] == 0
