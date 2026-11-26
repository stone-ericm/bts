"""Probe 3: Byte-for-byte non-reuse caller parity (base vs head), result polling/scoring parity, and operator text capture."""

from __future__ import annotations

from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
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
    spec = spec_from_file_location("probe3_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_tree(tree: str, scenario: str) -> dict:
    src_dir = str(BASE_SRC if tree == "base" else HEAD_SRC)
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--worker", tree, scenario],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-1500:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE3_JSON ")]
    return json.loads(lines[-1][len("PROBE3_JSON "):])


def canonical_pick(d: dict | None) -> dict | None:
    if d is None:
        return None
    out = dict(d)
    # model_git_sha comes from cwd (which is tmp_path -> None on both)
    return out


def worker_main(tree: str, scenario: str) -> dict:
    import io
    from contextlib import redirect_stderr
    from bts import contest_state, daily_decision, picks, scheduler

    with TemporaryDirectory(prefix=f"p3-{tree}-{scenario}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)
        err_buf = io.StringIO()

        with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp), redirect_stderr(err_buf):
            w.schedule[DAY] = w.schedule[DAY][:1]
            if scenario.startswith("auto_stale_"):
                w.automatic(5, age_hours=18)
            else:
                w.dated(5, best=10, saver="active")

            if scenario in {"lineup_dm", "lineup_public", "lineup_private"}:
                mode = scenario.split("_")[1]
                w.config["scheduler"]["pick_delivery"] = mode
                if mode == "private":
                    w.config["scheduler"]["entry_intent"] = "research"
                w.predictions = lambda: w.frame((0.99,), (False,))
                scheduler.run_day(DAY, w.config)

            elif scenario in {"slept_fallback_dm", "slept_fallback_public", "slept_fallback_private"}:
                mode = scenario.split("_")[2]
                w.config["scheduler"]["pick_delivery"] = mode
                if mode == "private":
                    w.config["scheduler"]["entry_intent"] = "research"
                w.predictions = lambda: w.frame((0.99,), (True,))
                scheduler.run_day(DAY, w.config)

            elif scenario in {"post_loop_fallback_dm", "post_loop_missed_pick"}:
                calls = 0

                def staged():
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return w.frame((0.99,), (True,))
                    return None

                w.now = offline.instant(DAY, 15, 0)
                w.predictions = staged
                real_deliver = scheduler._deliver_and_lock_pick
                if scenario == "post_loop_missed_pick":
                    mp.setattr(
                        contest_state,
                        "load_decision_streak_state",
                        lambda *a, **kw: (
                            (_ for _ in ()).throw(contest_state.ContestStateError("simulated post-loop state error"))
                            if calls >= 2 and kw.get("require_contest_state")
                            else contest_state.DecisionStreakState(
                                streak=5, saver_available=True, allow_double=True,
                                source="contest_manual", status="fresh", model_streak=0,
                                model_saver_available=True, observation_kind="dated",
                                dated_identity={
                                    "effective_date": DAY, "active_streak": 5,
                                    "best_streak": 10, "saver_state": "active",
                                },
                            )
                        ),
                    )
                scheduler.run_day(DAY, w.config)

            elif scenario in {"caller_already_delivered", "caller_attempted_unconfirmed", "caller_late_refusal", "caller_void_refusal", "caller_contest_refusal", "auto_stale_dm"}:
                res = scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)
                daily_obj = res["pick_result"].daily
                st = scheduler.SchedulerState(DAY, w.now.isoformat(), [], [], [], False, None, None, None)
                if scenario == "caller_already_delivered":
                    daily_obj.notification_sent = True
                    daily_obj.notification_id = "msg-prev"
                    daily_obj.delivered_at = w.now.isoformat()
                    picks.save_pick(daily_obj, w.picks)
                    ret = scheduler._deliver_and_lock_pick(daily_obj, w.config, w.picks, st, DAY, "fallback")
                elif scenario == "caller_attempted_unconfirmed":
                    daily_obj.delivery_attempted = True
                    picks.save_pick(daily_obj, w.picks)
                    ret = scheduler._deliver_and_lock_pick(daily_obj, w.config, w.picks, st, DAY, "fallback")
                elif scenario == "caller_late_refusal":
                    w.advance(45 * 60)
                    ret = scheduler._deliver_and_lock_pick(daily_obj, w.config, w.picks, st, DAY, "fallback", selection=res["selection"])
                elif scenario == "caller_void_refusal":
                    w.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    ret = scheduler._deliver_and_lock_pick(daily_obj, w.config, w.picks, st, DAY, "fallback", selection=res["selection"])
                elif scenario == "caller_contest_refusal":
                    w.dated(5, best=11, saver="active")
                    ret = scheduler._deliver_and_lock_pick(daily_obj, w.config, w.picks, st, DAY, "fallback", selection=res["selection"])
                elif scenario == "auto_stale_dm":
                    ret = scheduler._deliver_and_lock_pick(daily_obj, w.config, w.picks, st, DAY, "lineup", selection=res["selection"])
                else:
                    ret = None
                _ = ret

            else:
                raise ValueError(scenario)

            pick_file = w.picks / f"{DAY}.json"
            pick_json = json.loads(pick_file.read_text()) if pick_file.exists() else None
            dec_json = daily_decision.load_decision(DAY, w.picks)
            if dec_json:
                dec_json = {k: v for k, v in dec_json.items() if k != "finalized_at"}
            st_obj = scheduler.load_state(DAY, w.picks)
            st_json = json.loads((w.picks / DAY / "scheduler_state.json").read_text()) if st_obj else None
            if st_json and st_json.get("runs_completed"):
                for r in st_json["runs_completed"]:
                    r.pop("duration_sec", None)
            if st_json and st_json.get("fallback_refreshes"):
                for r in st_json["fallback_refreshes"]:
                    r.pop("duration_sec", None)
            return {
                "ok": True,
                "events": [(k, t.isoformat(), b) for k, t, b in w.events],
                "pick": canonical_pick(pick_json),
                "decision": dec_json,
                "state": st_json,
                "stderr": err_buf.getvalue(),
            }


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        print("PROBE3_JSON " + json.dumps(worker_main(sys.argv[2], sys.argv[3])), flush=True)
        return

    scenarios = [
        "lineup_dm",
        "lineup_public",
        "lineup_private",
        "slept_fallback_dm",
        "slept_fallback_public",
        "slept_fallback_private",
        "post_loop_fallback_dm",
        "post_loop_missed_pick",
        "caller_already_delivered",
        "caller_attempted_unconfirmed",
        "caller_late_refusal",
        "caller_void_refusal",
        "caller_contest_refusal",
        "auto_stale_dm",
    ]
    summary = {}
    for sc in scenarios:
        b = run_tree("base", sc)
        h = run_tree("head", sc)
        diffs = {}
        for field in ("events", "pick", "decision", "state", "stderr"):
            if b.get(field) != h.get(field):
                diffs[field] = {"base": b.get(field), "head": h.get(field)}
        summary[sc] = {"ok": b.get("ok") and h.get("ok"), "byte_identical": len(diffs) == 0, "diffs": diffs}
        print(f"[parity] {sc}: ok={summary[sc]['ok']} identical={summary[sc]['byte_identical']}", flush=True)

    out_path = REVIEW_DIR / "probe_3_parity.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
