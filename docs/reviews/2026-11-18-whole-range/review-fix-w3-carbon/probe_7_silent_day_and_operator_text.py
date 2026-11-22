"""Probe 7: Race-free 2-game silent day (unconfirmed later game) + exact operator text capture for every new/changed line."""

from __future__ import annotations

from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_ceab9854" / "src"
HEAD_SRC = REPO / "src"
DAY = "2027-03-24"


def load_offline(tmp_root: Path):
    source = REPO / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("probe7_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_worker(tree: str, case: str) -> dict:
    src_dir = str(BASE_SRC if tree == "base" else HEAD_SRC)
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--worker", tree, case],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-1500:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE7_JSON ")]
    return json.loads(lines[-1][len("PROBE7_JSON "):])


def worker_main(tree: str, case: str) -> dict:
    from contextlib import redirect_stderr
    from bts import contest_state, daily_decision, picks, scheduler, strategy

    with TemporaryDirectory(prefix=f"p7-{tree}-{case}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)
        err_buf = io.StringIO()

        if case == "race_free_two_game_silent_day":
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp), redirect_stderr(err_buf):
                w.dated(30, best=30)
                mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {})
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99, 0.40), (True, True))
                    w.advance(6 * 60)
                    return None

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "pick_dms": len(w.sends("dm")),
                    "health_alerts": [a.splitlines()[-1] for a in w.sends("health-dm")],
                    "deferred_archives": len(list((w.picks / DAY).glob("deferred_fallback*.json"))),
                    "refused_archives": len(list((w.picks / DAY).glob("refused_delivery*.json"))),
                    "decision": dec,
                }

        if case == "operator_text_double_rollover":
            start = offline.instant("2027-06-18", 23, 20)
            with offline.offline(start) as (w, mp), redirect_stderr(err_buf):
                w.schedule["2027-06-18"] = [offline.game(101, start + timedelta(minutes=70))]
                w.config["scheduler"].update(fallback_deadline_min=40, fallback_deadline_min_morning=40)
                w.automatic(4)
                real_check = scheduler.run_single_check
                real_read = contest_state.load_decision_streak_state
                checked = False
                calls = 0

                def forecast():
                    if not checked:
                        w.advance(2399)
                        return w.frame((0.99,), (True,))
                    return None

                def wrap_check(*a, **kw):
                    nonlocal checked
                    r = real_check(*a, **kw)
                    checked = True
                    return r

                def rollover_each_read(*a, **kw):
                    nonlocal calls
                    out = real_read(*a, **kw)
                    if checked and calls < 2:
                        calls += 1
                        w.advance(2 if calls == 1 else 86400)
                    return out

                w.predictions = forecast
                mp.setattr(strategy, "_load_mdp", lambda: None)
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
                mp.setattr(scheduler, "run_single_check", wrap_check)
                mp.setattr(contest_state, "load_decision_streak_state", rollover_each_read)
                scheduler.run_day("2027-06-18", w.config)
                return {
                    "ok": True,
                    "alerts": w.sends("health-dm"),
                    "stderr_lines": [ln for ln in err_buf.getvalue().splitlines() if "CONTEST STATE ERROR" in ln or "FALLBACK" in ln],
                }

        if case == "operator_text_after_read_rollover":
            start = offline.instant("2027-06-18", 23, 20)
            with offline.offline(start) as (w, mp), redirect_stderr(err_buf):
                w.schedule["2027-06-18"] = [offline.game(101, start + timedelta(minutes=70))]
                w.config["scheduler"].update(fallback_deadline_min=40, fallback_deadline_min_morning=40)
                w.automatic(4)
                real_write = picks.atomic_write_text
                checked = False
                real_check = scheduler.run_single_check

                def forecast():
                    if not checked:
                        w.advance(2399)
                        return w.frame((0.99,), (True,))
                    return None

                def wrap_check(*a, **kw):
                    nonlocal checked
                    r = real_check(*a, **kw)
                    checked = True
                    return r

                def slow_marker(p, txt):
                    real_write(p, txt)
                    if p == w.picks / "2027-06-18.json" and json.loads(txt)["delivery_attempted"]:
                        w.advance(2)

                w.predictions = forecast
                mp.setattr(strategy, "_load_mdp", lambda: None)
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
                mp.setattr(scheduler, "run_single_check", wrap_check)
                mp.setattr(picks, "atomic_write_text", slow_marker)
                scheduler.run_day("2027-06-18", w.config)
                return {
                    "ok": True,
                    "pick_dms": w.sends("dm"),
                    "alerts": w.sends("health-dm"),
                    "stderr_lines": [ln for ln in err_buf.getvalue().splitlines() if "CONTEST STATE ERROR" in ln or "FALLBACK" in ln or "LOCKED" in ln],
                }

        raise ValueError(case)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        print("PROBE7_JSON " + json.dumps(worker_main(sys.argv[2], sys.argv[3])), flush=True)
        return

    out = {
        "race_free_base": run_worker("base", "race_free_two_game_silent_day"),
        "race_free_head": run_worker("head", "race_free_two_game_silent_day"),
        "op_double_rollover": run_worker("head", "operator_text_double_rollover"),
        "op_after_read_rollover": run_worker("head", "operator_text_after_read_rollover"),
    }
    out_path = REVIEW_DIR / "probe_7_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
