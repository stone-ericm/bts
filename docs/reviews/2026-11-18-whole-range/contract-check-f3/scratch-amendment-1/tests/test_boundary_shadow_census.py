"""Tests for the boundary-shadow one-step table-intent disagreement census.

Registration: docs/superpowers/specs/2026-08-09-boundary-shadow-measurement.md
(r1). Mechanism phase only — the artifact must never contain outcomes.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.audit.boundary_shadow_census import (
    CensusHalt, asof_contest, attribution, load_ledger, quintile_boundaries,
    resolve_state, run_census,
)


# --- boundary construction ---

def test_quintile_boundaries_linear_interpolation():
    ps = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    got = quintile_boundaries(ps)
    import pandas as pd
    want = [float(pd.Series(ps).quantile(q, interpolation="linear"))
            for q in (0.2, 0.4, 0.6, 0.8)]
    assert got == pytest.approx(want)
    assert all(b2 > b1 for b1, b2 in zip(got, got[1:]))


def test_quintile_boundaries_duplicate_collapse_is_invalid():
    assert quintile_boundaries([0.75] * 10) is None


def test_quintile_boundaries_halt_on_bad_input():
    with pytest.raises(CensusHalt):
        quintile_boundaries([0.7, float("nan"), 0.8])
    with pytest.raises(CensusHalt):
        quintile_boundaries([0.7, 1.4])


# --- as-of ledger join + state resolution ---

def _ledger_file(tmp_path, rows):
    p = tmp_path / "contest_ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_asof_contest_latest_at_or_before():
    rows = [
        {"recorded_at": "2026-07-01T14:00:00Z", "active_streak": 5, "source_date": "2026-06-30"},
        {"recorded_at": "2026-07-02T14:00:00Z", "active_streak": 6, "source_date": "2026-07-01"},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        led = load_ledger(_ledger_file(Path(d), rows))
    ts = datetime(2026, 7, 2, 13, 0, tzinfo=timezone.utc)
    assert asof_contest(led, ts).active_streak == 5
    ts2 = datetime(2026, 7, 2, 14, 0, tzinfo=timezone.utc)
    assert asof_contest(led, ts2).active_streak == 6
    assert asof_contest(led, datetime(2026, 6, 1, tzinfo=timezone.utc)) is None


def test_resolve_state_recorded_beats_ledger(tmp_path):
    led = load_ledger(_ledger_file(tmp_path, [
        {"recorded_at": "2026-07-01T14:00:00Z", "active_streak": 5, "source_date": "2026-06-30"},
    ]))
    rec = {"streak": 8, "saver_available": True, "finalized_at": "2026-07-02T01:00:00Z"}
    src, streak, saver = resolve_state(rec, led, saver_active=True)
    assert (src, streak, saver) == ("recorded", 8, True)


def test_resolve_state_ledger_asof_for_state_null_rows(tmp_path):
    led = load_ledger(_ledger_file(tmp_path, [
        {"recorded_at": "2026-07-01T14:00:00Z", "active_streak": 5, "source_date": "2026-06-30"},
    ]))
    rec = {"streak": None, "saver_available": None, "finalized_at": "2026-07-01T22:00:00Z"}
    src, streak, saver = resolve_state(rec, led, saver_active=True)
    assert (src, streak, saver) == ("ledger_asof", 5, True)


def test_resolve_state_unknown_before_first_observation(tmp_path):
    led = load_ledger(_ledger_file(tmp_path, [
        {"recorded_at": "2026-07-01T14:00:00Z", "active_streak": 5, "source_date": "2026-06-30"},
    ]))
    rec = {"streak": None, "saver_available": None, "finalized_at": "2026-06-20T22:00:00Z"}
    src, streak, saver = resolve_state(rec, led, saver_active=True)
    assert src == "unknown" and streak is None


# --- clamp attribution ---

def test_attribution_parity_and_clamp_and_halt():
    assert attribution("single", "single") == "parity"
    assert attribution("double", "single") == "clamped_double_downgrade"
    with pytest.raises(CensusHalt):
        attribution("skip", "single")
    with pytest.raises(CensusHalt):
        attribution("single", "double")


# --- end-to-end on a tiny fixture ---

def _policy_npz(tmp_path):
    # 58 streaks x 181 days x 2 saver x 5 bins; action codes 0/1/2.
    table = np.zeros((58, 181, 2, 5), dtype=np.int8)
    table[:, :, :, :] = 1                # single everywhere...
    table[0:3, :, :, 3:] = 2             # ...double at low streak in top bins
    table[8:, :, :, 0] = 0               # ...skip at streak>=8 in Q0
    path = tmp_path / "policy.npz"
    np.savez(path, policy_table=table,
             boundaries=np.array([0.796, 0.811, 0.825, 0.841]),
             season_length=180)
    return path


def _decision(tmp_path, date, **kw):
    rec = {
        "schema_version": "bts_daily_decision_v1", "date": date,
        "action": kw.get("action", "single"), "source": kw.get("source", "mdp"),
        "primary": {"batter_id": 1, "batter_name": "X", "team": "NYM",
                    "game_pk": 9, "p_game_hit": kw.get("p", 0.80)},
        "double_down": None,
        "streak": kw.get("streak"), "saver_available": kw.get("saver"),
        "delivery_status": kw.get("delivery_status", "delivered"),
        "scoreable": kw.get("scoreable", True),
        "finalized_at": kw.get("finalized_at", f"{date}T22:00:00Z"),
    }
    d = tmp_path / "picks" / date
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(json.dumps(rec))
    return rec


def _fixture(tmp_path):
    picks = tmp_path / "picks"
    picks.mkdir(exist_ok=True)
    # skip day with recorded state (streak 8, Q0 -> table says skip)
    _decision(tmp_path, "2026-07-01", action="skip", p=0.78, streak=8, saver=True,
              delivery_status="not_applicable", scoreable=False,
              finalized_at="2026-07-02T01:20:00Z")
    # commit day, state-null (ledger recovers streak 2; p .80 -> bin1 -> single)
    _decision(tmp_path, "2026-07-03", action="single", p=0.80,
              finalized_at="2026-07-03T22:00:00Z")
    ledger = _ledger_file(tmp_path / "picks", [
        {"recorded_at": "2026-07-01T14:00:00Z", "active_streak": 8, "source_date": "2026-06-30"},
        {"recorded_at": "2026-07-03T14:00:00Z", "active_streak": 2, "source_date": "2026-07-02"},
    ])
    return picks, ledger, _policy_npz(tmp_path)


def test_run_census_end_to_end(tmp_path):
    picks, ledger, policy = _fixture(tmp_path)
    out = tmp_path / "census.json"
    art = run_census(
        picks_dir=picks, ledger_path=ledger, policy_path=policy,
        saver_active=True, start="2026-06-23", end="2026-08-09",
        as_of="2026-08-09", output=out,
    )
    assert out.exists() and not (tmp_path / "census.json.tmp").exists()
    assert art["schema_version"] == "bts_boundary_shadow_census_v1"
    assert art["role"].startswith("mechanism census")
    assert "outcome" not in json.dumps(art).lower()
    rows = art["rows"]
    assert len(rows) == 2
    by_date = {r["date"]: r for r in rows}
    assert by_date["2026-07-01"]["state_source"] == "recorded"
    assert by_date["2026-07-03"]["state_source"] == "ledger_asof"
    assert by_date["2026-07-01"]["parity"] == "parity"
    assert art["gates"]["gate_a"]["checked"] == 1
    assert art["gates"]["gate_a"]["mismatches"] == []
    # boundary sets present with provenance
    assert "primary" in art["boundaries"] and art["boundaries"]["primary"]["values"]


def test_run_census_gate_a_halts_on_ledger_disagreement(tmp_path):
    picks, ledger, policy = _fixture(tmp_path)
    # poison the ledger: as-of value for the recorded skip disagrees (9 != 8)
    ledger.write_text(json.dumps(
        {"recorded_at": "2026-07-01T14:00:00Z", "active_streak": 9,
         "source_date": "2026-06-30"}) + "\n")
    with pytest.raises(CensusHalt):
        run_census(picks_dir=picks, ledger_path=ledger, policy_path=policy,
                   saver_active=True, start="2026-06-23", end="2026-08-09",
                   as_of="2026-08-09", output=tmp_path / "c.json")
