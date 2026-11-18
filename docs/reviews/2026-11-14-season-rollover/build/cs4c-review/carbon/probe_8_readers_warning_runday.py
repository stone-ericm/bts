"""Adversarial Probe 8: Completes 20+ pick readers parity + Item 5 warning checks + Item 6 real run_day suite."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REVIEW_DIR))

from netguard import forbid_network
import probe_5_complete_suite as p5

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"


def run_readers_check() -> None:
    from bts import contest_state, daily_decision, entry_receipt, picks, scheduler, web
    from bts.experiment import artifacts as exp_artifacts
    from bts.health import (
        fallback_defer, late_delivery, mdp_policy_alignment,
        pitcher_sparsity, post_failure, postponed_pick, predicted_vs_realized,
        projected_lineup, same_team_corr,
    )
    from bts.state import export as state_export, regenerate as state_regen
    from scripts.audit.season_ledger.sources import pick_files as ledger_picks

    old_rec = {
        "schema": SCHEMA,
        "recorded_at": "2027-07-14T22:35:00-04:00",
        "age_hours": 20.0,
        "stale": True,
        "reason": "old",
    }
    fresh_rec = {
        "schema": SCHEMA,
        "recorded_at": "2027-07-15T16:35:00-04:00",
        "age_hours": 2.0,
        "stale": False,
        "reason": None,
    }
    p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                    0.82, [], False, 101, f"{DAY}T23:10:00Z")
    dp = picks.attach_provenance(picks.DailyPick(
        DAY, NOW.isoformat(), p1, None, None,
        notification_sent=True, notification_channel="bluesky_dm",
        notification_id="dm-1", delivered_at=NOW.isoformat(),
        result="hit", slot_results={"primary": "hit"},
    ))
    base_doc = {k: v for k, v in asdict(dp).items() if k != "automatic_reading"}
    null_doc = dict(base_doc, automatic_reading=None)
    fresh_doc = dict(base_doc, automatic_reading=fresh_rec)
    old_doc = dict(base_doc, automatic_reading=old_rec)

    policy_npz = REPO_ROOT / "data" / "models" / "mdp_policy.npz"
    projected_outputs = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for vname, vdoc in [("absent", base_doc), ("null", null_doc), ("fresh", fresh_doc), ("stale", old_doc)]:
            vdir = root / f"readers_{vname}"
            vdir.mkdir()
            raw_v = json.dumps(vdoc, indent=2).encode("utf-8")
            vpath = vdir / f"{DAY}.json"
            vpath.write_bytes(raw_v)
            (vdir / f"{DAY}.shadow.json").write_bytes(raw_v)
            daily_decision.write_decision(
                DAY, vdir, action="single", source="heuristic",
                primary=scheduler._row_from_daily(p1),
                delivery_status="delivered", scoreable=True, now=NOW,
            )
            r_dates = contest_state._resolved_pick_dates(vdir)
            r_latest = contest_state.latest_resolved_pick_date(vdir)
            r_gap = contest_state.resolved_pick_settlement_gap(vdir, date(2027, 7, 14))
            r_miss = contest_state._has_unconfirmed_miss(vdir, date(2027, 7, 14))
            r_replay = picks._replay_season_streak(vdir, 2027, "2027-07-16")
            r_hist = state_export._pick_to_historical(json.loads(raw_v))
            r_regen = state_regen._hist_to_pick_file(r_hist)
            assert "automatic_reading" not in r_hist and "automatic_reading" not in r_regen

            dp_loaded, held = picks.load_pick_bytes(DAY, vdir)
            rec_obj = entry_receipt.EntryReceipt(
                et_date=DAY, started_at=NOW, clock=lambda: NOW,
                expected_username=None, picks_dir=vdir,
            )
            rec_obj.selection(dp_loaded, held, None, None)
            rec_sel = dict(rec_obj.record()["selection"])
            assert rec_sel.pop("pick_file_sha256") == hashlib.sha256(raw_v).hexdigest()

            led_row, = ledger_picks.parse_pick_file(vpath.name, raw_v).rows
            assert led_row["content_sha256"] == hashlib.sha256(raw_v).hexdigest()
            assert json.loads(led_row["record_raw_json"]) == vdoc
            led_proj = {k: v for k, v in led_row.items() if k not in {"obs_id", "content_sha256", "record_raw_json"}}

            exp_snap = exp_artifacts.load_production_pick_snapshot(vpath, expected_date=DAY)
            assert exp_snap["source_sha256"] == hashlib.sha256(raw_v).hexdigest()
            exp_proj = {k: v for k, v in exp_snap.items() if k not in {"source_sha256", "production_pick_json"}}

            web_picks = web.load_all_picks(vdir)
            web_proj = {k: v for k, v in web_picks[0].items() if k != "automatic_reading"}

            today_d = date(2027, 7, 15)
            h_checks = {
                "post_failure": post_failure.check(vdir, today=today_d, now=NOW + timedelta(hours=5)),
                "late_delivery": late_delivery.check(vdir, today=today_d),
                "fallback_defer": fallback_defer.check(vdir, today=today_d),
                "postponed_pick": postponed_pick.check(vdir, today=today_d),
                "projected_lineup": projected_lineup.check(vdir, today=today_d),
                "same_team_corr": same_team_corr.check(vdir, today=today_d),
                "pitcher_sparsity": pitcher_sparsity.check(vdir, today=today_d),
                "predicted_vs_realized": predicted_vs_realized.check(vdir, today=today_d),
                "mdp_policy_alignment": mdp_policy_alignment.check(
                    vdir, policy_path=policy_npz, today=today_d
                ),
            }
            projected_outputs.append({
                "r_dates": r_dates, "r_latest": r_latest, "r_gap": r_gap, "r_miss": r_miss,
                "r_replay": r_replay, "r_hist": r_hist, "r_regen": r_regen,
                "rec_sel": rec_sel, "led_proj": led_proj, "exp_proj": exp_proj,
                "web_proj": web_proj, "h_checks": h_checks,
            })

    for other in projected_outputs[1:]:
        assert other == projected_outputs[0]
    print("[PASS] Item 4 all 20+ pick readers verified across absent, null, fresh, and stale records.")


def main() -> None:
    with forbid_network(allow_loopback=False):
        from bts import contest_state, picks
        with patch.object(contest_state, "datetime", p5.DayClock), patch.object(picks, "datetime", p5.DayClock):
            run_readers_check()
        p5.run_item_5_warning_checks()
        p5.run_item_6_real_run_day_suite()
    print("PROBE_8_ALL_PASSED")


if __name__ == "__main__":
    main()
