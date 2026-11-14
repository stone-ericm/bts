"""Round-2 adversarial probe for S1 v3, S2 v2, and S7."""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from bts.contest_fetch import build_observation, derive_source_date
from bts.contest_state import (
    ContestStateError,
     ContestStreakState,
    _contest_season_state,
    _has_unconfirmed_miss,
    _policy_saver_available,
    _resolved_pick_dates,
    load_decision_streak_state,
)
from bts.daily_decision import load_decision, write_decision
from bts.health.contest_state import check as health_check
from bts.leaderboard.scraper import parse_rounds_lookup
from bts.picks import load_streak, save_streak, update_streak
from bts.saver_state import load_saver_state, transition_saver_state
from bts.simulate.mdp import transition_outcomes
from bts.strategy import DecisionContext, _load_mdp, resolve_policy_decision
from bts.web import render_saver_section, saver_dashboard_context

ET = ZoneInfo("America/New_York")


def test_r1_closures() -> None:
    print("=== 1. Verify Round-1 Streak/Saver Findings (B1, B6, B7, a, b, c, d) ===")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        # (a) Malformed/non-dict JSON and unhashable result in pick files
        for idx, payload in enumerate(["{bad", "[]", "null", "7", '"str"', '{"result": []}', '{"result": {}}']):
            (p / f"2026-09-0{idx + 1}.json").write_text(payload)
        assert _resolved_pick_dates(p) == []
        assert _has_unconfirmed_miss(p, date(2026, 8, 1)) is False
        print("  (a) malformed/non-dict/unhashable pick files skipped cleanly: CLOSED")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        (sd / "saver_state.json").write_text(json.dumps({"season": 2026, "state": "active"}))
        (sd / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 12,
            "best_streak": 14,
            "source_date": "2026-09-26",
            "recorded_at": "2027-03-24T12:00:00Z",
        }))
        now = datetime(2027, 3, 24, 16, 0, tzinfo=timezone.utc)
        st = load_decision_streak_state(p, now=now, require_contest_state=True)
        ctx = saver_dashboard_context(p, now=now)
        assert (st.streak, st.status) == (0, "season_start")
        assert ctx.state == "uninitialized"  # 2027 saver is uninitialized, not 2026's 'active'
        print("  B1/(b) saver season uses now_year in decision, web, health, and cli fetch: CLOSED")

    # B6 part 1: Opening Night settled pick with source_date=None and active_streak=1
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        (p / "2027-03-24.json").write_text('{"result": "hit"}')
        (sd / "contest_streak.json").write_text(json.dumps({
            "active_streak": 1,
            "recorded_at": "2027-03-25T06:10:00Z",
            "source_date": None,
        }))
        alerts = health_check(p, expected=True, now=datetime(2027, 3, 25, 12, 0, tzinfo=timezone.utc))
        crits = [a for a in alerts if a.level == "CRITICAL"]
        assert crits == []
        print("  B6 (part 1, <=2 settled picks with source_date=None): CLOSED")

    # B6 part 2: Unexpired manual override across season boundary in health_check!
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        (sd / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 1,
            "source": "manual_cli",
            "source_date": "2026-09-26",
            "recorded_at": "2026-12-31T23:30:00-05:00",
            "override_expires_at": "2027-03-26T00:00:00Z",
        }))
        (p / "2027-03-24.json").write_text('{"result": "hit"}')
        now = datetime(2027, 3, 25, 15, 0, tzinfo=timezone.utc)
        st = load_decision_streak_state(p, now=now, require_contest_state=True)
        alerts = health_check(p, expected=True, now=now)
        crits = [a for a in alerts if a.level == "CRITICAL"]
        print(f"  B6 (part 2, unexpired manual override): decision=({st.streak}, {st.source}, {st.status}), "
              f"health CRITICALs={[c.message for c in crits]}")

    # B7: update_streak in picks.py (untouched file)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        save_streak(9, p, saver_available=False, applied_date="2026-09-26")
        (p / "2027-03-24.json").write_text('{"result": "hit"}')
        new_s = update_streak([True], p, date="2027-03-24")
        st = load_decision_streak_state(p, now=datetime(2027, 3, 25, 1, 5, tzinfo=timezone.utc))
        print(f"  B7 (untouched picks.py update_streak): streak.json={new_s}, model_only decision.streak={st.streak}")


def test_s1_v3_matrix_and_first_7_days() -> None:
    print("\n=== 2. S1 v3 Full State Matrix & 7-Day Season Start Walk ===")
    # Check uncovered states (future years)
    c_future_rec_old_src = ContestStreakState(
        streak=15, saver_available=None, source="mlb_bts_profile",
        source_date=date(2025, 9, 26), recorded_at="2027-01-01T06:00:00Z",
        path=Path("contest_streak.json"),
    )
    # Suppose now is 2026-12-31 23:00 ET (now_year=2026), recorded_year=2027 (> now_year), source_year=2025 (< now_year)
    st_uncovered = _contest_season_state(c_future_rec_old_src, now_year=2026, settled_this_year=True)
    print(f"  Uncovered cell (recorded_year=2027 > now_year=2026, source_year=2025 < now_year): {st_uncovered!r}")

    # Real MLB rounds.json behavior from tests/leaderboard/fixtures/rounds.json:
    # 2026 started at roundId=823; 2027 rounds.json will only map 2027 roundIds (e.g. 1010+).
    rounds_2027 = parse_rounds_lookup({
        "rounds": [
            {"id": 1010 + i, "date": f"2027-03-{24 + i:02d}T08:00:00-04:00", "status": "pending"}
            for i in range(7)
        ]
    })
    unreset_profile_2026 = {
        "activeStreak": 12,
        "seasonBestStreak": 18,
        "predictions": [{"roundId": 1008, "result": "hit"}],  # last settled round from Sept 2026
    }
    derived_src = derive_source_date(unreset_profile_2026["predictions"], rounds_2027)
    print(f"  Un-reset 2026 profile against 2027 rounds.json -> derive_source_date = {derived_src!r}")

    # Walk Days 1..4 on un-reset profile + 2027 rounds.json when (a) contest lags on Day 2 or research mode / missed app entry,
    # OR (b) fetch fails on Days 2..7 after succeeding on Day 1 morning!
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        obs = build_observation(
            unreset_profile_2026, derived_src, 1, "stonehengee",
            datetime(2027, 3, 24, 14, 30, tzinfo=timezone.utc),
        )
        (sd / "contest_streak.json").write_text(json.dumps(obs))
        # Day 1 pre-settlement
        d1 = load_decision_streak_state(p, now=datetime(2027, 3, 24, 18, 0, tzinfo=timezone.utc), require_contest_state=True)
        # Day 1 settles locally at 22:15 ET (even as a miss or void!)
        (p / "2027-03-24.json").write_text('{"result": "miss"}')
        # Day 2 (2027-03-25): either MLB profile hasn't settled/reset yet, or fetch failed on Day 2..7
        d2 = load_decision_streak_state(p, now=datetime(2027, 3, 25, 16, 0, tzinfo=timezone.utc), require_contest_state=True)
        h2 = [a for a in health_check(p, expected=True, now=datetime(2027, 3, 25, 16, 0, tzinfo=timezone.utc)) if a.level == "CRITICAL"]
        print(f"  Walk W_unreset_null_src: Day 1 streak={d1.streak} ({d1.status}) -> "
              f"after local Day-1 MISS, Day 2 streak={d2.streak} ({d2.status}), CRITICALs={len(h2)}!")

    # Walk Days 1..4 when un-reset profile STILL resolves source_date=2026-09-26 (Row 2 -> Row 4):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        (sd / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 12,
            "best_streak": 18,
            "source_date": "2026-09-26",
            "recorded_at": "2027-03-25T14:30:00Z",
        }))
        d1 = load_decision_streak_state(p, now=datetime(2027, 3, 24, 18, 0, tzinfo=timezone.utc), require_contest_state=True)
        # Day 1 settles locally (e.g. hit, or rained-out void, or operator missed app entry; or normal 1-day contest lag)
        (p / "2027-03-24.json").write_text('{"result": "hit"}')
        try:
            load_decision_streak_state(p, now=datetime(2027, 3, 25, 16, 0, tzinfo=timezone.utc), require_contest_state=True)
            d2_err = None
        except ContestStateError as e:
            d2_err = str(e)
        print(f"  Walk W_unreset_2026_src: Day 1 streak={d1.streak} ({d1.status}) -> "
              f"Day 2 require_contest_state=True raises ContestStateError({d2_err!r})")


def test_s7_and_callers_and_policy_diffs() -> None:
    print("\n=== 3. S7 Saver Semantics, Caller Disagreements, & Policy Action Diffs ===")
    # 3a. Verify transition_outcomes semantics
    for s in (0, 8, 9, 10, 12, 15, 16):
        t_true = transition_outcomes("single", s, True, p_hit=0.75, p_both=0.56)
        t_false = transition_outcomes("single", s, False, p_hit=0.75, p_both=0.56)
        if s in (8, 10, 16):
            print(f"  MDP transition single@{s}: saver=True -> miss=({t_true[1].next_streak}, {t_true[1].saver_available}), "
                  f"saver=False -> miss=({t_false[1].next_streak}, {t_false[1].saver_available})")

    # 3b. Cold-start / mid-season account with uninitialized saver_state.json and NO local settled picks
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        (sd / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 12,
            "best_streak": 14,
            "source_date": "2027-06-18",
            "recorded_at": "2027-06-19T14:30:00Z",
        }))
        now = datetime(2027, 6, 19, 16, 0, tzinfo=timezone.utc)
        st = load_decision_streak_state(p, now=now, require_contest_state=True)
        web_ctx = saver_dashboard_context(p, now=now)
        h_alerts = health_check(p, expected=True, now=now)
        write_decision("2027-06-19", p, action="double", source="mdp",
                       streak=st.streak, saver_available=st.saver_available,
                       delivery_status="delivered", scoreable=True, now=now)
        dec_rec = load_decision("2027-06-19", p)
        print(f"  Mid-season cold box (streak=12, best=14, saver=uninitialized, 0 local picks):")
        print(f"    decision.saver_available = {st.saver_available} (PASSES SAVER=TRUE IN 10-15 ZONE!)")
        print(f"    decision.json saver_available = {dec_rec['saver_available']}")
        print(f"    dashboard state = {web_ctx.state!r} ({render_saver_section(web_ctx)[:72]}...)")
        print(f"    health alerts = {[(a.level, a.message) for a in h_alerts]}")

    # 3c. Opening Day Row 2 false WARN in health_check (uses raw state.streak=12 instead of decision_streak=0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        (sd / "saver_state.json").write_text(json.dumps({"season": 2026, "state": "active"}))
        (sd / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 12,
            "best_streak": 18,
            "source_date": "2026-09-26",
            "recorded_at": "2027-03-24T12:00:00Z",
        }))
        now = datetime(2027, 3, 24, 16, 0, tzinfo=timezone.utc)
        st = load_decision_streak_state(p, now=now, require_contest_state=True)
        h_alerts = health_check(p, expected=True, now=now)
        print(f"  Opening Day Row 2 (file active_streak=12, decision_streak={st.streak}, saver={st.saver_available}):")
        print(f"    health alerts = {[(a.level, a.message) for a in h_alerts]}")

    # 3d. Streak 10..15 with saver.state == 'not_earned' (missed WARN in health_check!)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sd = p / "account_state"
        sd.mkdir()
        transition_saver_state(p, expected_prior="uninitialized", new_state="not_earned", season=2027, source="auto_earn")
        (sd / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 11,
            "best_streak": 11,
            "source_date": None,  # rounds lag -> auto_earn did not promote!
            "recorded_at": "2027-04-05T12:00:00Z",
        }))
        (p / "2027-04-04.json").write_text('{"result": "hit"}')
        now = datetime(2027, 4, 5, 16, 0, tzinfo=timezone.utc)
        st = load_decision_streak_state(p, now=now, require_contest_state=True)
        h_alerts = health_check(p, expected=True, now=now)
        print(f"  Streak 11 with saver='not_earned': decision.saver_available={st.saver_available}, "
              f"saver WARNs={[a.message for a in h_alerts if 'Streak Saver' in a.message]}")

    # 3e. Policy action changes across streaks 0..15 on tracked mdp_policy.npz & mdp_tail_policy.npz!
    mdp = _load_mdp()
    pt = mdp["policy_table"]
    bounds = mdp["boundaries"]
    print(f"  Tracked mdp_policy.npz shape={pt.shape}, boundaries={[round(b, 4) for b in bounds]}")
    actions = ("skip", "single", "double")
    for s in range(16):
        diffs_by_bin = []
        for q in range(pt.shape[3]):
            diff_days = [
                (d, actions[pt[s, d, 0, q]], actions[pt[s, d, 1, q]])
                for d in range(1, pt.shape[1])
                if pt[s, d, 0, q] != pt[s, d, 1, q]
            ]
            if diff_days:
                diffs_by_bin.append((q, len(diff_days), diff_days[0], diff_days[-1]))
        if diffs_by_bin:
            print(f"    streak={s:2d}: {diffs_by_bin}")

    tail = mdp["tail"]
    tpt = tail.policy_table  # [s, m, d, sv, q]
    tail_diffs_below_10 = int(np.sum(tpt[:10, :, 1:, 0, :] != tpt[:10, :, 1:, 1, :]))
    tail_diffs_10_15 = int(np.sum(tpt[10:16, :, 1:, 0, :] != tpt[10:16, :, 1:, 1, :]))
    print(f"  Tracked mdp_tail_policy.npz diffs (saver=0 vs 1): streak<10={tail_diffs_below_10}, streak 10..15={tail_diffs_10_15}")


if __name__ == "__main__":
    test_r1_closures()
    test_s1_v3_matrix_and_first_7_days()
    test_s7_and_callers_and_policy_diffs()
