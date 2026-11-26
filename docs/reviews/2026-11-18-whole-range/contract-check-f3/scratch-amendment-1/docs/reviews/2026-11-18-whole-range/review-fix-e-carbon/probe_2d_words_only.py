"""Probe 2d: Complete words-only check on stale_meta, refusal_meta, and effective_pick_bar
(popping `body_sha256` alongside `updated_at` from status files since `body_sha256` hashes the text).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_SRC = REPO / ".carbon-review/wr-e/base_63669c42/src"
HEAD_SRC = REPO / "src"

WORKER_CODE = r"""
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import bts
from bts import contest_state, picks, scheduler, strategy
from bts.health import alert, late_delivery

ET = ZoneInfo("America/New_York")
mode = sys.argv[1]
expected_root = Path(sys.argv[2]).resolve()
actual_file = Path(bts.__file__).resolve()
assert actual_file.is_relative_to(expected_root), f"{actual_file} not under {expected_root}"
out = {"bts_file": str(actual_file)}

def mk_pick(name, bid, team, gpk, p=0.82):
    return picks.Pick(name, bid, team, 1, "P1", 901, p, [], False, gpk, "2027-07-15T23:05:00Z", "BOS")

stale_meta = []
for route in ("delivery_old", "morning_old", "delivery_interrupt", "morning_interrupt", "cross_dedup", "retry_after_fail", "jan1_boundary"):
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        health_dir = Path(tmp) / "health_state"
        cfg = {"bluesky": {"dm_recipient": "op.bsky.social"}, "orchestrator": {"picks_dir": str(picks_dir)}}
        now = datetime(2027, 12, 31, 23, 55, tzinfo=ET) if route == "jan1_boundary" else datetime(2027, 7, 15, 10, 0, tzinfo=ET)
        ds = contest_state.DecisionStreakState(
            10, False, False, "contest", "fresh", None, None,
            observation_kind="automatic", contest_recorded_at="2027-07-14T10:00:00+00:00",
        )
        rec = contest_state.automatic_reading_record(ds, now)
        sent_dms = []
        calls = [0]
        def fake_send(rcpt, body, _route=route, _calls=calls, _sent=sent_dms):
            _calls[0] += 1
            if _route in ("delivery_interrupt", "morning_interrupt"):
                raise KeyboardInterrupt("sigint in dm")
            if _route == "retry_after_fail" and _calls[0] == 1:
                raise RuntimeError("transient network failure")
            _sent.append((rcpt, body))
            return "msg-1"
        with patch.object(contest_state, "load_decision_streak_state", lambda *a, **kw: ds), \
             patch.object(scheduler, "_now_et", lambda: now), \
             patch.object(alert, "send_dm", fake_send), \
             redirect_stderr(io.StringIO()):
            if route == "delivery_old":
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                status = json.loads((health_dir / "contest_stale_reading_dm_delivery_status.json").read_text())
                status.pop("updated_at", None)
                status.pop("body_sha256", None)
                stale_meta.append({"route": route, "sent_count": len(sent_dms), "status": status})
            elif route == "morning_old":
                scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                status = json.loads((health_dir / "contest_stale_reading_dm_delivery_status.json").read_text())
                status.pop("updated_at", None)
                status.pop("body_sha256", None)
                stale_meta.append({"route": route, "sent_count": len(sent_dms), "status": status})
            elif route == "delivery_interrupt":
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "contained": True})
            elif route == "morning_interrupt":
                try:
                    scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                    contained = True
                except KeyboardInterrupt:
                    contained = False
                stale_meta.append({"route": route, "contained": contained})
            elif route == "cross_dedup":
                scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "sent_count": len(sent_dms)})
            elif route == "retry_after_fail":
                scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "calls": calls[0], "sent_count": len(sent_dms)})
            elif route == "jan1_boundary":
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                now_jan1 = datetime(2028, 1, 1, 0, 5, tzinfo=ET)
                with patch.object(scheduler, "_now_et", lambda: now_jan1):
                    scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "sent_count": len(sent_dms), "d1_has_dec31": "2027-12-31" in sent_dms[0][1], "d2_has_jan1": "2028-01-01" in sent_dms[1][1]})
out["stale_meta"] = stale_meta

refusal_meta = []
for rkind in ("late", "void"):
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        now = datetime(2027, 7, 15, 19, 10, tzinfo=ET)
        dp = picks.DailyPick(
            date="2027-07-15", run_time=now.isoformat(),
            pick=mk_pick("Steven Kwan", 101, "CLE", 70001, 0.82),
            double_down=None, runner_up=None, bluesky_posted=False, bluesky_uri=None,
        )
        picks.save_pick(dp, picks_dir)
        st_obj = scheduler.SchedulerState(
            date="2027-07-15", schedule_fetched_at=now.isoformat(),
            games=[], confirmed_game_pks=[], runs_completed=[],
            pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        captured = []
        with patch.object(scheduler, "_now_et", lambda: now), \
             patch.object(alert, "send_dm", lambda r, b, _c=captured: (_c.append((r, b)) or "id-1")), \
             redirect_stderr(io.StringIO()):
            if rkind == "late":
                scheduler._refuse_late_delivery(dp, {"bluesky": {"dm_recipient": "op"}}, picks_dir, st_obj, "2027-07-15", "fallback", now - timedelta(minutes=10), now)
            else:
                scheduler._refuse_void_game_delivery(dp, {"bluesky": {"dm_recipient": "op"}}, picks_dir, st_obj, "2027-07-15", "fallback", {70001: "Postponed"})
        eod_alerts = [(a.level, a.source, a.message) for a in late_delivery.check(picks_dir, today=date(2027, 7, 15))]
        status_doc = json.loads((Path(tmp) / "health_state/health_dm_delivery_status.json").read_text())
        status_doc.pop("updated_at", None)
        status_doc.pop("body_sha256", None)
        saved_st = scheduler.load_state("2027-07-15", picks_dir)
        archive_doc = json.loads((picks_dir / "2027-07-15" / saved_st.delivery_refusals[0]["archive"]).read_text())
        refusal_meta.append({
            "kind": rkind,
            "pick_gone": picks.load_pick("2027-07-15", picks_dir) is None,
            "pick_locked": saved_st.pick_locked,
            "refusal_record": saved_st.delivery_refusals,
            "archive_refusal": archive_doc["refused_delivery"],
            "status_doc": status_doc,
            "eod_alerts": eod_alerts,
        })
out["refusal_meta"] = refusal_meta

mdp = strategy._load_mdp()
end = date(2027, 9, 27)
bars = []
regimes_seen = set()
for days in range(0, 186):
    day_str = (end - timedelta(days=days)).isoformat()
    for streak in range(57):
        regime, _ = strategy._regime(mdp, streak, day_str)
        regimes_seen.add(regime)
        best_variants = [(streak, "trusted")]
        if regime == strategy.OBJECTIVE_TAIL:
            best_variants.extend([(56, "trusted"), (56, "untrusted"), (None, None)])
        for saver in (False, True):
            for b_strk, b_trust in best_variants:
                val = strategy.effective_pick_bar(streak, day_str, saver, mdp, b_strk, b_trust)
                bars.append(val)
out["bar_count"] = len(bars)
out["regimes_seen"] = sorted(regimes_seen)
out["bars"] = bars
print(json.dumps(out))
"""


def run_worker(mode: str, src_dir: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_dir}:{REPO}"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", WORKER_CODE, mode, str(src_dir)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=45, check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    base = run_worker("base", BASE_SRC)
    head = run_worker("head", HEAD_SRC)
    assert base["bts_file"] != head["bts_file"]
    assert base["stale_meta"] == head["stale_meta"], f"{base['stale_meta']} != {head['stale_meta']}"
    assert base["refusal_meta"] == head["refusal_meta"], f"{base['refusal_meta']} != {head['refusal_meta']}"
    assert base["regimes_seen"] == ["reach57", "tail"] == head["regimes_seen"]
    assert base["bars"] == head["bars"]
    print("4. Stale warning metadata/dedup/retry/interrupts matched:", len(head["stale_meta"]), "routes.")
    print("5. Refusal side effects and EOD late_delivery matched:", len(head["refusal_meta"]), "kinds.")
    print(f"6. effective_pick_bar matched across {head['bar_count']} states in both regimes.")


if __name__ == "__main__":
    main()
