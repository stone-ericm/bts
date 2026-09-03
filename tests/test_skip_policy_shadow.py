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
    # decision flips to a committed pick (e.g. late delivery) — a SAME-DAY
    # event, so prune sees it inside the age fence (round-2 R5)
    write_decision("2026-06-18", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)
    assert prune_superseded(
        tmp_path, now=datetime(2026, 6, 18, 23, 0, tzinfo=UTC),
    ) == ["2026-06-18"]
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
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744)
    assert s["schema_version"] == "bts_skip_policy_shadow_status_v2"
    assert s["counts"]["divergent_days"] == 5
    assert s["counts"]["pending"] == 1
    assert s["counts"]["void"] == 1
    assert s["shadow_band_hit_rate"]["resolved"] == 3   # void excluded
    assert s["shadow_band_hit_rate"]["hits"] == 2
    assert abs(s["shadow_band_hit_rate"]["rate"] - 2 / 3) < 1e-9


def test_make_hit_checker_passes_through_void(monkeypatch):
    """The shadow's realized-outcome checker must surface a suspended-game 'void' (not
    collapse it to 'miss'), and call check_hit with the status contract. The shadow then
    excludes voids from the band hit rate.
    """
    import bts.picks
    from bts.skip_policy_shadow import make_hit_checker
    seen = {}

    def fake_check_hit(game_pk, batter_id, batter_name=None, date=None, team=None,
                       *, return_status=False):
        seen["return_status"] = return_status
        return "void"

    monkeypatch.setattr(bts.picks, "check_hit", fake_check_hit)
    checker = make_hit_checker()
    result = checker({"game_pk": 1, "batter_id": 2, "batter_name": "B", "team": "X"})
    assert result == "void"
    assert seen["return_status"] is True


# ---- F10 (2026-07-09 audit): pre-registered checkpoint verdicts. The old
# nightly Wilson look inflated the chance of an eventual chance crossing;
# now the verdict is only evaluated at n ∈ CHECKPOINTS, each look at a
# Bonferroni-split alpha, computed deterministically from the FIRST c
# resolved records in date order (stateless: every rebuild replays the same
# looks). A decisive look is terminal — later data can't un-decide it.

def _dated_recs(results):
    """Divergent records with distinct increasing dates so 'first c' is stable."""
    from datetime import timedelta
    base = datetime(2026, 4, 1, tzinfo=UTC)
    return [
        {"divergent": True, "shadow_pick_result": r,
         "date": (base + timedelta(days=i)).strftime("%Y-%m-%d"), "rank1": _cand()}
        for i, r in enumerate(results)
    ]


def test_checkpoint_verdict_insufficient_below_first_checkpoint():
    s = build_skip_policy_shadow_status(_dated_recs(["hit"] * 29))
    assert s["shadow_band_hit_rate"]["verdict"] == "insufficient_n"
    assert s["shadow_band_hit_rate"]["verdict_basis"]["checkpoint"] is None


def test_checkpoint_verdict_decisive_at_first_look():
    below = _dated_recs(["hit"] * 10 + ["miss"] * 20)   # 10/30, hi ~= .55 < .744 at z=2.394
    s = build_skip_policy_shadow_status(below, breakeven_p=0.744)
    assert s["shadow_band_hit_rate"]["verdict"] == "below_breakeven"
    basis = s["shadow_band_hit_rate"]["verdict_basis"]
    assert basis["checkpoint"] == 30
    assert basis["n_used"] == 30 and basis["hits_used"] == 10

    above = _dated_recs(["hit"] * 29 + ["miss"])        # 29/30, lo ~= .79 > .744 at z=2.394
    s = build_skip_policy_shadow_status(above, breakeven_p=0.744)
    assert s["shadow_band_hit_rate"]["verdict"] == "above_breakeven"


def test_checkpoint_decisive_verdict_is_terminal():
    # decisive ABOVE at n=30, then 30 straight misses: the pre-registered
    # look already fired — the verdict must not flip.
    recs = _dated_recs(["hit"] * 29 + ["miss"] + ["miss"] * 30)
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744)
    assert s["shadow_band_hit_rate"]["verdict"] == "above_breakeven"
    assert s["shadow_band_hit_rate"]["verdict_basis"]["checkpoint"] == 30


def test_checkpoint_straddle_advances_to_next_look():
    # 21/30 (~.70) straddles at the first look; the next 30 all miss →
    # 21/60 (.35) is decisively below at the SECOND pre-registered look.
    recs = _dated_recs(["hit"] * 21 + ["miss"] * 9 + ["miss"] * 30)
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744)
    assert s["shadow_band_hit_rate"]["verdict"] == "below_breakeven"
    assert s["shadow_band_hit_rate"]["verdict_basis"]["checkpoint"] == 60


def test_between_checkpoints_no_new_look():
    # n=45: the only look so far was at 30 (straddle). The extra 15 misses
    # feed the monitoring CI but must NOT constitute a new formal look.
    recs = _dated_recs(["hit"] * 21 + ["miss"] * 9 + ["miss"] * 15)
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744)
    assert s["shadow_band_hit_rate"]["verdict"] == "straddles_breakeven"
    basis = s["shadow_band_hit_rate"]["verdict_basis"]
    assert basis["checkpoint"] == 30
    assert basis["n_used"] == 30
    # monitoring stats still reflect ALL resolved records
    assert s["shadow_band_hit_rate"]["resolved"] == 45
    assert abs(s["shadow_band_hit_rate"]["rate"] - 21 / 45) < 1e-9


def test_checkpoints_use_first_c_in_date_order():
    # same multiset, different order: early hits vs early misses flip the
    # first-30 window, so the checkpoint result must follow date order.
    early_hits = _dated_recs(["hit"] * 29 + ["miss"] + ["miss"] * 15)
    early_misses = _dated_recs(["miss"] * 15 + ["hit"] * 29 + ["miss"])
    s_hits = build_skip_policy_shadow_status(early_hits, breakeven_p=0.744)
    s_misses = build_skip_policy_shadow_status(early_misses, breakeven_p=0.744)
    assert s_hits["shadow_band_hit_rate"]["verdict"] == "above_breakeven"
    assert s_misses["shadow_band_hit_rate"]["verdict"] != "above_breakeven"


def test_status_schema_v2():
    s = build_skip_policy_shadow_status(_dated_recs(["hit"] * 5))
    assert s["schema_version"] == "bts_skip_policy_shadow_status_v2"
    assert s["initiative"]["checkpoints"] == [30, 60, 90]


def test_checkpoint_membership_excludes_mutable_recent_records():
    # Codex review L3: records younger than the void-staleness window can
    # still change (a pending record resolving late reshuffles the first-c
    # window). Checkpoint membership is therefore restricted to records old
    # enough that their fate is sealed; recent ones feed monitoring only.
    from datetime import date, timedelta
    recs = _dated_recs(["hit"] * 10 + ["miss"] * 20)          # decisively below at c=30
    last_date = max(r["date"] for r in recs)
    # 'today' is the day after the last record: the newest records are
    # inside the mutability window -> not yet checkpoint-eligible
    as_of = date.fromisoformat(last_date) + timedelta(days=1)
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744, as_of=as_of)
    assert s["shadow_band_hit_rate"]["verdict"] == "insufficient_n"
    basis = s["shadow_band_hit_rate"]["verdict_basis"]
    assert basis["eligible_n"] < 30
    # monitoring still sees everything
    assert s["shadow_band_hit_rate"]["resolved"] == 30

    # once every record has aged past the window, the look fires
    as_of = date.fromisoformat(last_date) + timedelta(days=10)
    s = build_skip_policy_shadow_status(recs, breakeven_p=0.744, as_of=as_of)
    assert s["shadow_band_hit_rate"]["verdict"] == "below_breakeven"
    assert s["shadow_band_hit_rate"]["verdict_basis"]["eligible_n"] == 30


# ---- F10: regime fingerprint stored per record (future stratification;
# the breakeven came from one model era — records must be attributable).

def test_record_carries_regime_fingerprint(tmp_path):
    _write_mdp_skip("2026-06-18", tmp_path, bid=1)
    (tmp_path / "2026-06-18.json").write_text(json.dumps({
        "date": "2026-06-18",
        "run_time": "2026-06-18T15:00:00+00:00",
        "policy_npz_sha256": "polsha256",
        "feature_env_hash": "envhash123",
    }))
    rec = record_skip_from_decision("2026-06-18", tmp_path)
    assert rec["regime"]["policy_npz_sha256"] == "polsha256"
    assert rec["regime"]["feature_env_hash"] == "envhash123"
    # Codex review L5: the pick file is mutable and a skip cycle doesn't
    # re-save it, so the stamp is best-effort — run_time is recorded so a
    # future stratification can detect a stale-provenance stamp.
    assert rec["regime"]["pick_run_time"] == "2026-06-18T15:00:00+00:00"


def test_record_regime_none_when_pick_file_absent(tmp_path):
    _write_mdp_skip("2026-06-18", tmp_path, bid=1)
    rec = record_skip_from_decision("2026-06-18", tmp_path)
    assert rec["regime"] is None


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


def test_prune_superseded_never_touches_aged_records(tmp_path):
    # Codex round-2 R5: the same-day-flip claim must be ENFORCED — an old
    # record whose decision.json later vanishes/corrupts (backdated recovery
    # run, disk repair) must NOT be pruned, or a fired checkpoint's
    # membership could change. Age-fence: only records younger than the
    # eligibility window are prunable.
    from bts.skip_policy_shadow import prune_superseded
    _write_mdp_skip("2026-05-01", tmp_path, bid=1)
    record_skip_from_decision("2026-05-01", tmp_path)
    # decision.json vanishes long after the fact
    (tmp_path / "2026-05-01" / "decision.json").unlink()

    removed = prune_superseded(tmp_path, now=datetime(2026, 7, 10, tzinfo=UTC))
    assert removed == []
    assert decision_path("2026-05-01", tmp_path).exists()

    # a RECENT superseded record still prunes (the real same-day case)
    _write_mdp_skip("2026-07-09", tmp_path, bid=2)
    record_skip_from_decision("2026-07-09", tmp_path)
    (tmp_path / "2026-07-09" / "decision.json").unlink()
    removed = prune_superseded(tmp_path, now=datetime(2026, 7, 10, tzinfo=UTC))
    assert removed == ["2026-07-09"]


def test_prune_fence_and_eligibility_never_overlap(tmp_path):
    # Boundary (round-3 pre-fix): at calendar age exactly
    # CHECKPOINT_ELIGIBLE_AFTER_DAYS a record must NOT be prunable — the
    # eligibility cutoff (date <= as_of - 4d) admits it that same night, and
    # a record that is both eligible and prunable reopens the R5 hole.
    from datetime import date
    from bts.skip_policy_shadow import CHECKPOINT_ELIGIBLE_AFTER_DAYS, prune_superseded
    _write_mdp_skip("2026-07-06", tmp_path, bid=3)
    record_skip_from_decision("2026-07-06", tmp_path)
    (tmp_path / "2026-07-06" / "decision.json").unlink()
    # age exactly 4 calendar days (eligible tonight): must be fenced
    now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    assert (date(2026, 7, 10) - date(2026, 7, 6)).days == CHECKPOINT_ELIGIBLE_AFTER_DAYS
    removed = prune_superseded(tmp_path, now=now)
    assert removed == []
    assert decision_path("2026-07-06", tmp_path).exists()


def test_aged_superseded_contradiction_is_flagged_not_pruned(tmp_path):
    # Round-3 finding (both partial runs converged on it): an AGED record
    # whose decision.json later flips to non-skip (historical backfill /
    # recovery run) escapes pruning by design — fired-look membership is
    # frozen. But the contradiction must be FLAGGED, not silently carried:
    # the status now reports aged records whose authoritative decision no
    # longer says mdp-skip.
    from bts.skip_policy_shadow import find_aged_contradictions
    _write_mdp_skip("2026-05-01", tmp_path, bid=1)
    record_skip_from_decision("2026-05-01", tmp_path)
    # authoritative decision later rewritten to a delivered single
    write_decision("2026-05-01", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)

    now = datetime(2026, 7, 10, tzinfo=UTC)
    assert prune_superseded(tmp_path, now=now) == []          # frozen membership
    assert find_aged_contradictions(tmp_path, now=now) == ["2026-05-01"]

    out = tmp_path / "status.json"
    write_status(tmp_path, out, generated_at="2026-07-10T00:00:00Z", now=now)
    s = json.loads(out.read_text())
    assert s["counts"]["aged_superseded_records"] == ["2026-05-01"]

    # a RECENT flip is handled by prune, not flagged
    _write_mdp_skip("2026-07-09", tmp_path, bid=2)
    record_skip_from_decision("2026-07-09", tmp_path)
    write_decision("2026-07-09", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)
    assert find_aged_contradictions(tmp_path, now=now) == ["2026-05-01"]


def test_aged_contradictions_excluded_from_checkpoint_eligibility(tmp_path):
    # Round-3 F3: annotation isn't enough — a contradicted aged record is a
    # reclassification (void-equivalent) and must leave the eligible sequence,
    # not be frozen into a future look's first-30.
    from datetime import date, timedelta
    base = date(2026, 4, 1)
    for i in range(30):
        d = (base + timedelta(days=i)).isoformat()
        _write_mdp_skip(d, tmp_path, bid=100 + i)
        record_skip_from_decision(d, tmp_path)
        p = decision_path(d, tmp_path)
        rec = json.loads(p.read_text()); rec["shadow_pick_result"] = "miss"
        p.write_text(json.dumps(rec))
    # one record's authoritative decision later flips to a delivered single
    write_decision("2026-04-05", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)

    out = tmp_path / "status.json"
    now = datetime(2026, 7, 1, tzinfo=UTC)
    write_status(tmp_path, out, generated_at="2026-07-01T00:00:00Z", now=now)
    s = json.loads(out.read_text())
    assert s["counts"]["aged_superseded_records"] == ["2026-04-05"]
    # 30 records minus the contradicted one -> below the first checkpoint
    assert s["shadow_band_hit_rate"]["verdict_basis"]["eligible_n"] == 29
    assert s["shadow_band_hit_rate"]["verdict"] == "insufficient_n"


# ---- tail objective (2026-09-03): season-best-unbeatable skips are a different rule ----

def test_tail_objective_skip_is_not_recorded(tmp_path):
    write_decision("2026-09-19", tmp_path, action="skip", source="mdp", primary=_cand(1, 0.74),
                   streak=0, delivery_status="not_applicable", scoreable=False,
                   objective="emax_season_best", best_streak=18, best_status="trusted",
                   effective_best=18)
    assert record_skip_from_decision("2026-09-19", tmp_path) is None


def test_reach57_objective_skip_is_still_recorded(tmp_path):
    write_decision("2026-08-19", tmp_path, action="skip", source="mdp", primary=_cand(1, 0.74),
                   streak=10, delivery_status="not_applicable", scoreable=False,
                   objective="reach57")
    assert record_skip_from_decision("2026-08-19", tmp_path) is not None
