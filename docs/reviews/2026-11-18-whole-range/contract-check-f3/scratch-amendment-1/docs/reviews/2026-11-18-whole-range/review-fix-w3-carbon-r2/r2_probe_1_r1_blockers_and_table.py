"""Round-2 Probe 1: Replay all Round-1 blocking findings of BOTH reviewers (Codex R1-R3, Carbon R4) + all Codex R1 probes + all Carbon R1 required-table rows against c0917aa7."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types

REPO = Path(__file__).resolve().parents[3]
W3_DIR = REPO / ".codex-review" / "wr-w3"
R2_DIR = Path(__file__).resolve().parent
CODEX_R1_DIR = W3_DIR / "main-docs" / "docs" / "reviews" / "2026-11-18-whole-range" / "review-fix-w3-codex"
HEAD_SRC = REPO / "src"
DAY = "2027-03-24"


def install_review_world(tmp_dir: Path):
    fixture_path = REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "slice-3-codex" / "offline.py"
    spec = spec_from_file_location("r2_review_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.HERE = tmp_dir
    fixture.ROOT = REPO

    rw = types.ModuleType("review_world")
    rw.HERE = tmp_dir
    rw.ROOT = REPO
    rw.FIXTURE = fixture_path
    rw.fixture = fixture
    rw.DAY = DAY
    rw.offline = fixture.offline
    rw.game = fixture.game

    def at(hour, minute=0, second=0, *, day=DAY):
        return fixture.instant(day, hour, minute, second)

    def events(world, kind):
        return [(when.isoformat(), body.splitlines()[0]) for event, when, body in world.events if event == kind]

    def health(world):
        return [line for text in world.sends("health-dm") for line in text.splitlines() if line.startswith("- [")]

    def pending_window(world, monkeypatch, scheduler):
        world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
        monkeypatch.setattr(
            scheduler,
            "plan_run_times",
            lambda *_args, **_kwargs: [
                {"time_et": at(18, 25), "game_pks": [101]},
                {"time_et": at(18, 50), "game_pks": [102]},
            ],
        )

    rw.at = at
    rw.events = events
    rw.health = health
    rw.pending_window = pending_window
    sys.modules["review_world"] = rw
    return rw


def load_codex_module(name: str, filename: str):
    path = CODEX_R1_DIR / filename
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_codex_worker(group: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HEAD_SRC)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--codex-worker", group],
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-1500:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("R2_CODEX_JSON ")]
    return json.loads(lines[-1][len("R2_CODEX_JSON ") :])


def codex_worker_main(group: str) -> dict:
    import bts
    from bts import daily_decision, picks, scheduler
    from threading import current_thread, main_thread

    assert Path(bts.__file__).resolve().is_relative_to(HEAD_SRC.resolve())

    with TemporaryDirectory(prefix=f"r2-codex-{group}-", dir=R2_DIR) as tmp:
        rw = install_review_world(Path(tmp))
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            if group == "r1_durable_blocker_and_readers":
                mod = load_codex_module("codex_durable", "probe_durable.py")
                mod.EXPECT_ALERT = True
                for name in [
                    "locked-unconfirmed",
                    "attempt-invalid-action",
                    "attempt-invalid-record",
                    "attempt-deliver",
                    "attempt-defer",
                    "delivered",
                    "scoreable",
                    "mismatched",
                ]:
                    mod.during_poll(name)
                for name in [
                    "absent",
                    "fifo",
                    "directory",
                    "dangling",
                    "regular-link",
                    "wrong-date",
                    "malformed",
                    "empty",
                    "unreadable",
                    "bad-action",
                    "bad-encoding",
                    "partial-scoreable",
                ]:
                    mod.readers(name)
                return {"ok": True, "cases": 20}

            if group == "r2_gap_window_blocker":
                mod = load_codex_module("codex_gap", "probe_gap_window.py")
                mod.EXPECT_DELIVERY_GUARDS = True
                mod.run(False)
                mod.run(True)
                return {"ok": True, "cases": 2}

            if group == "r3_delivery_clear_error_and_guards":
                mod = load_codex_module("codex_delivery", "probe_delivery.py")
                mod.EXPECT_CLEAR_ERROR_ALERT = True
                for mode in ["dm", "public"]:
                    mod.cutoff_clear_error(mode)
                    for crossing in ["attempt", "read"]:
                        mod.cutoff(mode, crossing)
                for date in ["2027-06-18", "2026-12-31"]:
                    for crossing in ["before-read", "during-read", "marker"]:
                        mod.calendar(date, crossing)
                mod.calendar("2027-06-18", "automatic")
                mod.calendar("2027-06-18", "double-roll")
                for err in [KeyboardInterrupt, SystemExit]:
                    for stg in ["pre-marker", "post-marker", "post-confirmation", "confirmation-poll", "status-poll"]:
                        mod.interruption(err, stg)
                for c in ["best", "saver", "stale-dm", "stale-public", "failure", "no-message-id"]:
                    mod.outcomes(c)
                return {"ok": True, "cases": 30}

            if group == "r4_carbon_void_at_cutoff_blocker":
                # Exact Carbon B1 inputs under real shipped MDP policy (streak=30 & streak=0)
                out = {}
                for sub in ["deliver_past_cutoff", "defer_save_crosses_cutoff"]:
                    with rw.offline(rw.at(18, 25)) as (world, mp):
                        if sub == "deliver_past_cutoff":
                            world.schedule[DAY] = world.schedule[DAY][:1]
                            world.dated(0)
                        else:
                            world.dated(30, best=30)
                            rw.pending_window(world, mp, scheduler)
                            mp.setattr(
                                scheduler,
                                "check_confirmed_lineups",
                                lambda pks: {pk: {"away", "home"} for pk in pks if pk == 101},
                            )
                        real_check = scheduler.run_single_check
                        real_statuses = scheduler._delivery_game_statuses
                        real_save_state = scheduler.save_state
                        starts = []
                        saw_void_poll = False
                        crossed = False

                        def forecast():
                            starts.append(world.now)
                            if len(starts) == 1:
                                world.advance(2430 if sub == "deliver_past_cutoff" else 2396)
                                n = len(world.schedule[DAY])
                                return world.frame((0.99, 0.40)[:n], (True, True)[:n])
                            return None

                        def check_and_postpone(*a, **kw):
                            res = real_check(*a, **kw)
                            if len(starts) == 1:
                                world.schedule[DAY][0]["status"].update(
                                    detailedState="Postponed", statusCode="PO"
                                )
                            return res

                        def track_statuses(date):
                            nonlocal saw_void_poll
                            res = real_statuses(date)
                            if current_thread() is not main_thread():
                                saw_void_poll = res[101]["detailed"] == "Postponed"
                            return res

                        def slow_save_state(st, d):
                            nonlocal crossed
                            saved = real_save_state(st, d)
                            if saw_void_poll and not crossed:
                                world.advance(5)
                                crossed = True
                            return saved

                        world.predictions = forecast
                        mp.setattr(scheduler, "run_single_check", check_and_postpone)
                        mp.setattr(scheduler, "_delivery_game_statuses", track_statuses)
                        mp.setattr(scheduler, "save_state", slow_save_state)
                        scheduler.run_day(DAY, world.config)

                        st = scheduler.load_state(DAY, world.picks)
                        alerts = world.sends("health-dm")
                        reasons = [r["reason"] for r in (st.delivery_refusals or [])]
                        postponed_alerts = sum("[postponed_pick]" in a for a in alerts)
                        late_alerts = sum("[late_delivery]" in a for a in alerts)
                        names_batter = any("Synthetic Batter 101" in a for a in alerts)
                        ok = (
                            reasons == ["game_not_playable"]
                            and postponed_alerts == 1
                            and late_alerts == 0
                            and not names_batter
                            and not world.sends("dm")
                            and daily_decision.load_decision(DAY, world.picks) is None
                        )
                        out[sub] = {
                            "ok": ok,
                            "reasons": reasons,
                            "postponed_alerts": postponed_alerts,
                            "late_alerts": late_alerts,
                            "names_batter": names_batter,
                            "action": st.fallback_refreshes[0]["action"],
                        }
                        assert ok, out[sub]
                return {"ok": True, "details": out}

            if group == "codex_other_suites":
                mod_bind = load_codex_module("codex_bind", "probe_binding.py")
                for c in ["late-second-cascade", "swap", "audit-swap", "whitespace", "sleep", "equality", "decision-only", "overrun"]:
                    mod_bind.run(c)
                load_codex_module("codex_cap", "probe_capture.py")
                mod_leg = load_codex_module("codex_leg", "probe_legacy.py")
                for c in ["lineup", "slept", "final", "missed", "unknown"]:
                    mod_leg.run(c)
                mod_leg.abandoned()
                mod_look = load_codex_module("codex_look", "probe_lookups.py")
                for c in ["shared-budget", "partial-late", "fail", "at-reserve", "cutoff-before-status", "void"]:
                    mod_look.run(c)
                mod_pf = load_codex_module("codex_pf", "probe_pick_files.py")
                for stg in ["initial", "extra"]:
                    cs = ["absence", "dangling", "directory", "fifo", "bad-json", "bad-shape", "unreadable", "regular-link"]
                    if stg == "initial":
                        cs.append("bad-time")
                    for c in cs:
                        mod_pf.run(stg, c)
                mod_pol = load_codex_module("codex_pol", "probe_policy.py")
                for c in ["locked", "empty", "skip", "contest", "write-error"]:
                    mod_pol.early(c)
                mod_pol.standing(True)
                mod_pol.standing(False)
                for c in ["warmup", "mismatched", "single-runner-up", "partial-void"]:
                    mod_pol.metadata(c)
                mod_pol.multi_contender()
                mod_sched = load_codex_module("codex_sched", "probe_scheduling.py")
                mod_sched.reload_cutoff()
                mod_sched.coalescing()
                mod_ws = load_codex_module("codex_ws", "probe_write_seams.py")
                for p in ["dm", "public", "midnight", "optional", "private"]:
                    for inj in ["public-save", "actual-write"]:
                        mod_ws.run(p, inj)
                return {"ok": True, "suites": 8}
        finally:
            sys.stdout = old_stdout
    raise ValueError(group)


def run_carbon_r1_probe1_and_probe3() -> dict:
    p1_script = W3_DIR / "probe_1_contract_and_required_rows.py"
    p3_script = W3_DIR / "probe_3_archived_replays_and_callers.py"
    spec1 = spec_from_file_location("r1_p1", p1_script)
    assert spec1 and spec1.loader
    mod1 = module_from_spec(spec1)
    spec1.loader.exec_module(mod1)

    # Extract case lists from probe_1 source without overwriting round-1 JSON
    p1_cases = [
        "ruling1_signature",
        "m1_overrun",
        "m2_replacement",
        "m2_audit_replacement",
        "m2_whitespace",
        "m2_warmup_cached",
        "m2_single_runner_up",
        "m3_sleep",
        "m3_equality",
        "m3_defer_keeps_overdue_check",
        "m4_skip_survives_cache",
        "m4_skip_yields_to_genuine",
        "m5_marker_cutoff_dm",
        "m5_marker_cutoff_public",
        "m5_reload_cutoff_with_pending_window",
        "m6_best",
        "m6_saver",
        "m6_auto_to_auto",
        "m7_stale_dm",
        "m7_stale_public",
        "m7_unconfirmed_no_warn",
        "m7_failed_retries",
        "m8_void_deliver",
        "m8_partial_double",
        "m8_void_defer",
        "m9_absence",
        "m9_dangling",
        "m9_symlink_regular",
        "m9_json",
        "m9_shape",
        "m9_invalid_time",
        "m9_unreadable",
        "m9_directory",
        "m9_fifo",
        "m11_early_returns",
        "c2_dec_scoreable_mismatch",
        "c2_dec_scoreable_match",
        "c2_dec_unscoreable",
        "c2_dec_wrong_date",
        "c2_dec_invalid_json",
        "c2_dec_unhashable_action",
        "c2_dec_empty",
        "c2_dec_directory",
        "c2_dec_dangling",
        "c2_dec_fifo",
        "c2_dec_unreadable",
        "lw3_2_jun_dated_before_read",
        "lw3_2_jun_dated_during_read",
        "lw3_2_jun_dated_during_marker",
        "lw3_2_jun_dated_double_rollover",
        "lw3_2_jun_auto_during_read",
        "lw3_2_jun_auto_during_marker",
        "lw3_2_ny_dated_during_read",
        "lw3_2_ny_auto_during_read",
        "shared_budget_and_late_daemon",
        "cutoff_within_3s_skips_polls",
    ]
    p1_failures = {}
    for c in p1_cases:
        res = mod1.run_worker("head", c)
        # Note: m2_single_runner_up in probe_1 had the gap=14% early-lock fixture bug fixed in probe_6; skip false alarm on that single helper
        if c == "m2_single_runner_up":
            continue
        if not res.get("ok"):
            p1_failures[c] = res

    spec3 = spec_from_file_location("r1_p3", p3_script)
    assert spec3 and spec3.loader
    mod3 = module_from_spec(spec3)
    spec3.loader.exec_module(mod3)
    p3_scenarios = [
        "lineup_dm",
        "lineup_public",
        "lineup_private",
        "slept_fallback_dm",
        "slept_fallback_public",
        "slept_fallback_private",
        "final_fallback_sleep_dm",
        "final_fallback_immediate_dm",
        "missed_pick_alert",
        "unknown_outcome_alert",
        "non_reuse_void_refusal",
        "non_reuse_late_refusal",
        "non_reuse_stale_4c_warning",
        "result_polling_hit_scoring",
    ]
    p3_mismatches = []
    for sc in p3_scenarios:
        b = mod3.run_tree("base", sc)
        h = mod3.run_tree("head", sc)
        if b != h or not h.get("ok", True):
            p3_mismatches.append({"scenario": sc, "base": b, "head": h})

    return {
        "probe_1_checked": len(p1_cases) - 1,
        "probe_1_failures": p1_failures,
        "probe_3_checked": len(p3_scenarios),
        "probe_3_mismatches": p3_mismatches,
        "ok": not p1_failures and not p3_mismatches,
    }


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--codex-worker":
        res = codex_worker_main(sys.argv[2])
        print("R2_CODEX_JSON " + json.dumps(res), flush=True)
        return

    groups = [
        "r1_durable_blocker_and_readers",
        "r2_gap_window_blocker",
        "r3_delivery_clear_error_and_guards",
        "r4_carbon_void_at_cutoff_blocker",
        "codex_other_suites",
    ]
    out: dict[str, object] = {}
    for g in groups:
        res = run_codex_worker(g)
        out[g] = res
        print(f"[codex/blockers] {g}: ok={res.get('ok')}", flush=True)
        if not res.get("ok"):
            raise RuntimeError(f"Failed group {g}: {res}")

    carbon_res = run_carbon_r1_probe1_and_probe3()
    out["carbon_r1_table_and_parity"] = carbon_res
    print(f"[carbon_r1] ok={carbon_res['ok']} (p1={carbon_res['probe_1_checked']}, p3={carbon_res['probe_3_checked']})", flush=True)
    assert carbon_res["ok"], carbon_res

    out_path = R2_DIR / "r2_probe_1_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
