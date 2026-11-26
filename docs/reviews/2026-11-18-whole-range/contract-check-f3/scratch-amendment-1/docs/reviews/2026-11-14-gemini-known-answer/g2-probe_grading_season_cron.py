"""Adversarial probes for 48f6f01 grading edge cases, 29e10bf season end, and 1137cbe cron."""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts.health.tail_policy import check as tail_policy_check
from bts.picks import (
    DailyPick,
    Pick,
    _boxscore_grade,
    _grade_pick_pre_suspension,
    check_hit,
    effective_daily_result,
    grade_pick_in_feed,
    resolve_daily_slot_results,
)
from bts.reconcile_receipt import ReconcileReceipt
from bts.skip_policy_shadow import make_hit_checker, reconcile_decision
from bts.strategy import _days_remaining, season_end_date

ET = ZoneInfo("America/New_York")


def _box_feed(game_pk: int, batter_id: int, batter_name: str, stats: dict,
              away_players: dict | None = None, abstract: str = "F", detailed: str = "Final",
              resume_dt: str | None = None, plays: list | None = None) -> dict:
    away = away_players if away_players is not None else {}
    home = {
        f"ID{batter_id}": {
            "person": {"id": batter_id, "fullName": batter_name},
            "stats": {"batting": stats},
        }
    }
    dt = {}
    if resume_dt:
        dt["resumeDateTime"] = resume_dt
    return {
        "gameData": {
            "game": {"pk": game_pk},
            "status": {"abstractGameCode": abstract, "detailedState": detailed},
            "datetime": dt,
        },
        "liveData": {
            "boxscore": {"teams": {"away": {"players": away}, "home": {"players": home}}},
            "plays": {"allPlays": plays or []},
        },
    }


def probe_grading_edge_cases():
    print("[PROBE 5A grading edge cases + rules 6A-6C]")
    # 1. All-walks + HBP + SH + IBB (AB=0, SF=0) -> void
    f_pass = _box_feed(1, 592450, "Aaron Judge", {"hits": 0, "atBats": 0, "sacFlies": 0, "baseOnBalls": 3, "hitByPitch": 1, "sacBunts": 1})
    # 2. 0-for-0 with 1 sacFly -> miss
    f_sf = _box_feed(1, 592450, "Aaron Judge", {"hits": 0, "atBats": 0, "sacFlies": 1})
    # 3. Rostered DNP ({}) -> void
    f_dnp = _box_feed(1, 592450, "Aaron Judge", {})
    # 4. Name collision: away team has different batter with same name who went 0-for-4, home has target batter_id=592450 who walked (0 AB, void) -> check ID vs name order!
    f_same_name = _box_feed(
        1, 592450, "Will Smith",
        {"hits": 0, "atBats": 0, "sacFlies": 0, "baseOnBalls": 3},
        away_players={"ID669257": {"person": {"id": 669257, "fullName": "Will Smith"}, "stats": {"batting": {"hits": 0, "atBats": 4, "sacFlies": 0}}}},
    )
    print(f"  walks+HBP+SH={grade_pick_in_feed(f_pass, 592450, 'Aaron Judge')}")
    print(f"  sacFly={grade_pick_in_feed(f_sf, 592450, 'Aaron Judge')}")
    print(f"  DNP={grade_pick_in_feed(f_dnp, 592450, 'Aaron Judge')}")
    print(f"  same-name collision (ID592450 in home walked, ID669257 in away 0-4): {grade_pick_in_feed(f_same_name, 592450, 'Will Smith')}")

    # 5. Suspended game with resumeDateTime where pre-suspension play lacks startTime -> missed hit vs void!
    f_susp_missing_ts = _box_feed(
        1, 592450, "Aaron Judge", {"hits": 1, "atBats": 2, "sacFlies": 0},
        resume_dt="2027-06-02T17:00:00Z",
        plays=[{"about": {}, "matchup": {"batter": {"id": 592450, "fullName": "Aaron Judge"}}, "result": {"eventType": "single"}}],
    )
    # 6. Suspended game marked abstractGameCode="F", detailedState="Suspended: Rain", resumeDateTime=None (e.g. not yet scheduled)
    f_susp_no_resumedt = _box_feed(
        1, 592450, "Aaron Judge", {"hits": 0, "atBats": 2, "sacFlies": 0},
        abstract="F", detailed="Suspended: Rain", resume_dt=None,
    )
    print(f"  suspended pre-play startTime=None (1-2 pre-susp): {grade_pick_in_feed(f_susp_missing_ts, 592450, 'Aaron Judge')}")
    print(f"  abstract=F detailed='Suspended: Rain' resumeDateTime=None (0-2): {grade_pick_in_feed(f_susp_no_resumedt, 592450, 'Aaron Judge')}")

    # 7. ReconcileReceipt basis label for normal-game 6A/6B void:
    rec = ReconcileReceipt(clock=lambda: datetime(2027, 5, 2, 2, 0, tzinfo=ET), lookback_days=2)
    slot = {"role": "pick", "batter_id": 592450, "batter_name": "Aaron Judge", "game_pk": 1, "result": "void", "schedule_void_state": None}
    basis, pk = rec._qualify(slot, [("url", f_pass)], status=None)
    print(f"  ReconcileReceipt._qualify on normal-game walk-only Pass: basis={basis!r}, game_pk={pk}")

    # 8. skip_policy_shadow.reconcile_decision when make_hit_checker returns "void"
    rec_shadow = {
        "date": "2027-05-01",
        "rank1": {"game_pk": 1, "batter_id": 592450, "batter_name": "Aaron Judge", "team": "NYY"},
    }
    with patch("bts.picks.check_hit", return_value="void"):
        checker = make_hit_checker()
        changed = reconcile_decision(rec_shadow, hit_checker=checker, now=datetime(2027, 5, 2, 2, 0, tzinfo=ET))
    print(f"  skip_policy_shadow reconcile on 'void': changed={changed}, resolved_status={rec_shadow.get('resolved_status')}, outcome={rec_shadow.get('outcome')}")


def probe_season_end_and_cron():
    print("[PROBE 5B season_end_date + tail_policy + cron]")
    for ds in ("2026-09-29", "2027-01-01", "2027-03-24", "2027-09-26", "2027-09-27", "2028-04-01"):
        print(f"  {ds}: season_end={season_end_date(ds)}, days_remaining={_days_remaining(ds)}")
    # Test cron show output with fake HOME config
    with tempfile.TemporaryDirectory() as tmp:
        toml_path = Path(tmp) / ".bts-orchestrator.toml"
        toml_path.write_text('[scheduler]\nentry_intent = "enter"\n')
        env = {"HOME": tmp, "HEALTHCHECKS_PING_URL": "https://hc-ping.com/fake", "PATH": "/usr/bin:/bin:/usr/local/bin"}
        res = subprocess.run(
            ["bash", "scripts/cron-setup-hetzner.sh", "show"],
            env=env, capture_output=True, text=True, check=False,
        )
        print(f"  cron-setup show exit={res.returncode}")
        for line in res.stdout.splitlines():
            if any(k in line for k in ("data build", "day-outcome", "check-pick-entered")):
                print(f"    CRON: {line}")


if __name__ == "__main__":
    probe_grading_edge_cases()
    probe_season_end_and_cron()
