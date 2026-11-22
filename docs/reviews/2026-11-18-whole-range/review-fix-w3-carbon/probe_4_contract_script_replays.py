"""Probe 4: Replay archived contract-check-w3, contract-check-w3/r2, and slice-3-codex scenarios against HEAD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
HEAD_SRC = REPO / "src"
DAY = "2027-03-24"


def load_offline(tmp_root: Path):
    source = REPO / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("probe4_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_case(case: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HEAD_SRC)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--worker", case],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-1500:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE4_JSON ")]
    return json.loads(lines[-1][len("PROBE4_JSON "):])


class FifoDeadline(BaseException):
    pass


def worker_main(case: str) -> dict:
    from bts import contest_state, daily_decision, picks, scheduler, strategy

    with TemporaryDirectory(prefix=f"p4-{case}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)

        # Replay slice-3-codex/interactions.py :: overrun_double_cascade on HEAD
        if case == "slice3_overrun_double_cascade":
            with offline.offline(offline.instant(DAY, 10)) as (w, _):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                calls = []

                def forecast():
                    calls.append(w.now.strftime("%H:%M"))
                    if w.now < offline.instant(DAY, 18, 25):
                        return None
                    if w.now == offline.instant(DAY, 18, 25):
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    w.advance(6 * 60)
                    return None

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "calls": calls,
                    "reused": st.fallback_refreshes[0]["reused_check_cascade"],
                    "dm_count": len(w.sends("dm")),
                    "refused_archives": len(list((w.picks / DAY).glob("refused_delivery*.json"))),
                    "scoreable": dec["scoreable"] if dec else False,
                }

        # Replay slice-3-codex/interactions.py :: cached_fallbacks (in_loop=True) verbatim on HEAD!
        if case == "slice3_cached_fallbacks_in_loop":
            with offline.offline(offline.instant(DAY, 18, 20)) as (w, mp):
                w.dated(0)
                w.schedule[DAY] = w.schedule[DAY][:1]
                original_save = picks.save_pick
                corrected = False

                def save_and_correct(daily_obj, directory):
                    nonlocal corrected
                    path = original_save(daily_obj, directory)
                    if not corrected:
                        corrected = True
                        w.dated(30)
                    return path

                mp.setattr(picks, "save_pick", save_and_correct)
                w.predictions = lambda: (None if corrected else w.frame((0.99,), (True,)))
                scheduler.run_day(DAY, w.config)
                pred_times = [t.strftime("%H:%M") for k, t, _ in w.events if k == "predict"]
                return {
                    "ok": True,
                    "corrected": corrected,
                    "dm_count": len(w.sends("dm")),
                    "health_dms": len(w.sends("health-dm")),
                    "predict_times": pred_times,
                    "pick_chosen_streak": picks.load_pick(DAY, w.picks).chosen_on["streak"],
                }

        # Replay contract-check-w3/timeline.py :: skip & lineup (accepted costs)
        if case in {"cc1_accepted_cost_skip", "cc1_accepted_cost_lineup"}:
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, _):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(30 if case.endswith("skip") else 0)
                calls = []

                def forecast():
                    calls.append(w.now.strftime("%H:%M"))
                    if len(calls) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    if case.endswith("skip"):
                        return w.frame((0.10,), (False,))
                    f = w.frame((0.99,), (False,))
                    f.loc[0, ["batter_name", "batter_id"]] = ["Confirmed Batter", 999]
                    return f

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "calls": calls,
                    "action": dec["action"] if dec else None,
                    "committed_batter": (dec.get("primary") or {}).get("batter_name") if dec else None,
                    "dm_count": len(w.sends("dm")),
                }

        # Replay contract-check-w3/r2/probe.py :: attempt_poll (attempt marker added during confirmation poll)
        if case in {"cc2_attempt_poll_deliver", "cc2_attempt_poll_defer", "cc2_delivered_during_poll", "cc2_scoreable_during_poll"}:
            at = lambda h, m: offline.instant(DAY, h, m)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 50), "game_pks": [102]},
                ])
                starts = []
                mutated = False

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(22 * 60)
                        return w.frame((0.99, 0.80), (True, True))
                    return None

                def poll_confirm(pks):
                    nonlocal mutated
                    if starts and pks == [102] and not mutated:
                        mutated = True
                        cur = picks.load_pick(DAY, w.picks)
                        if case == "cc2_delivered_during_poll":
                            cur.notification_sent = True
                            cur.notification_id = "ext-msg-1"
                            cur.delivered_at = w.now.isoformat()
                            picks.save_pick(cur, w.picks)
                        elif case == "cc2_scoreable_during_poll":
                            daily_decision.write_decision(
                                DAY, w.picks, action="single", source="mdp",
                                scoreable=True, delivery_status="private_locked",
                                primary={"batter_id": 777, "batter_name": "Ext Batter", "game_pk": 102},
                            )
                        else:
                            cur.delivery_attempted = True
                            picks.save_pick(cur, w.picks)
                        if case == "cc2_attempt_poll_deliver":
                            return {102: {"away", "home"}}
                    return {pk: {"away", "home"} for pk in pks if pk == 101}

                w.predictions = forecast
                mp.setattr(scheduler, "check_confirmed_lineups", poll_confirm)
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "dm_count": len(w.sends("dm")),
                    "unknown_alerts": sum("OUTCOME UNKNOWN" in a for a in w.sends("health-dm")),
                    "delivery_status": dec["delivery_status"] if dec else None,
                    "primary_id": (dec.get("primary") or {}).get("batter_id") if dec else None,
                }

        # Replay post-poll pick FIFO / decision FIFO / non-regular symlink during poll
        if case in {"cc2_pick_fifo_during_poll", "cc2_pick_corrupt_during_poll"}:
            at = lambda h, m: offline.instant(DAY, h, m)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 50), "game_pks": [102]},
                ])
                pick_path = w.picks / f"{DAY}.json"
                real_check = scheduler.run_single_check
                starts = []
                swapped = False

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(22 * 60)
                        return w.frame((0.99, 0.70), (True, True))
                    return w.frame((0.70, 0.99), (True, False)).iloc[[1, 0]]

                def check_and_repair(*a, **kw):
                    if swapped and pick_path.exists() or os.path.lexists(pick_path):
                        if len(starts) >= 1:
                            # Restore clean absence before second check so legacy load_pick doesn't fail/block
                            pick_path.unlink()
                    return real_check(*a, **kw)

                def poll_swap(pks):
                    nonlocal swapped
                    if starts and pks == [102] and not swapped:
                        swapped = True
                        pick_path.unlink()
                        if case == "cc2_pick_fifo_during_poll":
                            os.mkfifo(pick_path)
                        else:
                            pick_path.write_text("{corrupt")
                    return {pk: {"away", "home"} for pk in pks if pk == 101}

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", check_and_repair)
                mp.setattr(scheduler, "check_confirmed_lineups", poll_swap)

                def on_alarm(*_):
                    raise FifoDeadline("blocked_on_post_poll_pick_fifo")

                old = signal.signal(signal.SIGALRM, on_alarm)
                signal.alarm(2)
                try:
                    scheduler.run_day(DAY, w.config)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)

                dec = daily_decision.load_decision(DAY, w.picks)
                st = scheduler.load_state(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "refreshes": st.fallback_refreshes,
                    "deferred_archives": len(list((w.picks / DAY).glob("deferred_fallback*.json"))),
                    "committed_pk": dec["primary"]["game_pk"] if dec else None,
                }

        raise ValueError(case)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        print("PROBE4_JSON " + json.dumps(worker_main(sys.argv[2])), flush=True)
        return

    cases = [
        "slice3_overrun_double_cascade",
        "slice3_cached_fallbacks_in_loop",
        "cc1_accepted_cost_skip",
        "cc1_accepted_cost_lineup",
        "cc2_attempt_poll_deliver",
        "cc2_attempt_poll_defer",
        "cc2_delivered_during_poll",
        "cc2_scoreable_during_poll",
        "cc2_pick_fifo_during_poll",
        "cc2_pick_corrupt_during_poll",
    ]
    results = {}
    for c in cases:
        results[c] = run_case(c)
        print(f"[replay] {c}: ok={results[c].get('ok')}", flush=True)

    out_path = REVIEW_DIR / "probe_4_replays.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
