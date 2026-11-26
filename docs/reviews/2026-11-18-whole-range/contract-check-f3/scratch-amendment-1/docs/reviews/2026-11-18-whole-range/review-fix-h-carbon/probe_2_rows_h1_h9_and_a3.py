"""Adversarial probe 2: rows H1 to H9, full A3 artifact degradation matrix, interrupts, and test diff audit."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import hashlib
from itertools import product
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bts import daily_decision, dm, orchestrator, picks, scheduler, strategy
from bts.simulate import mdp, tail_policy
from scripts.audit.season_walk import walk


def _two_bin_tail(artifact: dict) -> tail_policy.TailPolicy:
    freq, hit, both = [0.5, 0.5], [0.75, 0.75], [0.5, 0.5]
    pol = tail_policy.solve_emax_season_best(freq, hit, both).policy
    manifest = tail_policy.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2])
    tp = replace(
        artifact["tail"],
        boundaries=[0.82],
        policy_table=pol,
        bin_freq=freq,
        bin_p_hit=hit,
        bin_p_both=both,
        manifest=manifest,
    )
    tail_policy.validate_tail_policy(tp, expected_base_sha=artifact["base_sha256"])
    return tp


def _rewrite_npz(path: Path, **updates) -> None:
    with np.load(path, allow_pickle=False) as arrays:
        data = dict(arrays)
    for k, v in updates.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    np.savez_compressed(path, **data)


def check_h1_h2_h3() -> dict:
    strategy._mdp_cache = None
    artifact = strategy._load_mdp()
    tail_sha = artifact["tail"].sha256
    end_2028 = date(2028, 9, 28)
    count = 0
    for raw in range(1, 272):
        day = (end_2028 - timedelta(days=raw)).isoformat()
        for streak in (57, 58, 60, 100):
            for saver in (False, True):
                for p in (0.01, 0.50, 0.80, 0.99):
                    for allow, diff in ((True, True), (False, True), (True, False), (False, False)):
                        ctx = strategy.DecisionContext(
                            p, 0.95 if diff else None, diff, day, allow, artifact, 57, "trusted"
                        )
                        dec = strategy.resolve_policy_decision(ctx, streak, saver)
                        want_pol = "double" if raw <= 3 else "single"
                        want_act = "double" if (want_pol == "double" and allow and diff) else "single"
                        assert dec.action == want_act
                        assert dec.policy_action == want_pol
                        assert dec.objective == "longest_streak"
                        assert dec.source == "mdp"
                        assert dec.tail_sha256 == tail_sha
                        assert dec.days_effective == min(raw, 180)
                        assert dec.degraded_reason is None and dec.guard is None
                        count += 1
    return {"h1_h2_h3_cases": count}


def check_h4_and_a3_degradations(tmp_dir: Path) -> dict:
    strategy._mdp_cache = None
    shipped = strategy._load_mdp()
    orig_base_default = mdp.DEFAULT_POLICY_PATH
    orig_tail_default = tail_policy.DEFAULT_TAIL_POLICY_PATH
    orig_rule = strategy.longest_streak_action

    modes = [
        "tail_missing",
        "tail_corrupt",
        "tail_truncated",
        "tail_directory",
        "tail_oversize",
        "tail_sha_mismatch",
        "tail_two_bins",
        "tail_malformed_manifest",
        "tail_missing_key",
        "base_missing",
        "base_corrupt",
        "base_directory",
        "base_oversize",
        "base_negative_horizon",
        "base_zero_horizon",
        "nothing_on_disk",
        "rule_raises_runtime_error",
    ]

    log_records: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record.getMessage())

    handler = ListHandler(level=logging.ERROR)
    strategy.log.addHandler(handler)

    tested = 0
    try:
        for mode in modes:
            case_dir = tmp_dir / f"h4_{mode}"
            case_dir.mkdir(parents=True, exist_ok=True)
            base_path = case_dir / "base.npz"
            tail_path = case_dir / "tail.npz"
            shutil.copyfile(REPO / "data/models/mdp_policy.npz", base_path)
            shutil.copyfile(REPO / "data/models/mdp_tail_policy.npz", tail_path)
            strategy.longest_streak_action = orig_rule

            if mode == "tail_missing":
                tail_path.unlink()
            elif mode == "tail_corrupt":
                tail_path.write_bytes(b"corrupt-not-zip")
            elif mode == "tail_truncated":
                tail_path.write_bytes(tail_path.read_bytes()[:64])
            elif mode == "tail_directory":
                tail_path.unlink()
                tail_path.mkdir()
            elif mode == "tail_oversize":
                tail_path.write_bytes(b"x" * (tail_policy.MAX_ARTIFACT_BYTES + 32))
            elif mode == "tail_sha_mismatch":
                with base_path.open("ab") as fh:
                    fh.write(b"extra-bytes")
            elif mode == "tail_two_bins":
                tail_policy.save_tail_policy(_two_bin_tail(shipped), tail_path)
            elif mode == "tail_malformed_manifest":
                bad_manifest = dict(shipped["tail"].manifest, bin_counts=[None])
                _rewrite_npz(tail_path, manifest=np.array(json.dumps(bad_manifest)))
                # Confirm the raw loader still raises TypeError per Limit 7 / A9
                try:
                    tail_policy.load_tail_policy(tail_path, expected_base_sha=shipped["base_sha256"])
                except TypeError:
                    pass
                else:
                    raise AssertionError("Expected TypeError from raw load_tail_policy on bin_counts=[None]")
            elif mode == "tail_missing_key":
                _rewrite_npz(tail_path, bin_p_both=None)
            elif mode == "base_missing":
                base_path.unlink()
            elif mode == "base_corrupt":
                base_path.write_bytes(b"not-a-valid-npz")
            elif mode == "base_directory":
                base_path.unlink()
                base_path.mkdir()
            elif mode == "base_oversize":
                base_path.write_bytes(b"y" * (tail_policy.MAX_ARTIFACT_BYTES + 32))
            elif mode in ("base_negative_horizon", "base_zero_horizon"):
                hz = -1 if mode == "base_negative_horizon" else 0
                _rewrite_npz(base_path, season_length=np.array(hz))
                sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
                tail_policy.save_tail_policy(replace(shipped["tail"], base_policy_sha256=sha), tail_path)
            elif mode == "nothing_on_disk":
                base_path.unlink()
                tail_path.unlink()
            elif mode == "rule_raises_runtime_error":
                def _boom(*_a, **_kw):
                    raise RuntimeError("rule exploded")
                strategy.longest_streak_action = _boom

            mdp.DEFAULT_POLICY_PATH = base_path
            tail_policy.DEFAULT_TAIL_POLICY_PATH = tail_path

            # Test on BOTH d=1 (2027-09-26, where healthy rule doubles!) and d=74 (2027-07-15),
            # and BOTH p=0.50 (below 0.80 heuristic!) and p=0.99: must ALWAYS force single!
            for day in (date(2027, 9, 26), date(2027, 7, 15)):
                for top_p in (0.50, 0.99):
                    strategy._mdp_cache = None
                    log_records.clear()
                    picks_dir = case_dir / f"picks_{day.isoformat()}_{top_p}"
                    picks_dir.mkdir(parents=True, exist_ok=True)
                    frame = walk._predictions(day, top_p)
                    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
                    with walk.offline(datetime.combine(day, time(12), walk.ET)):
                        sel = strategy.select_pick(
                            frame, day.isoformat(), picks_dir, streak=57, saver_available=False,
                            allow_double=True, game_statuses_detailed=statuses,
                        )
                        # Also verify attach_provenance + save_pick never raise even when path is a directory!
                        picks.attach_provenance(
                            sel.pick_result.daily,
                            blend_path=case_dir / "missing_blend.pkl",
                            policy_path=base_path,
                            tail_path=tail_path,
                        )
                        picks.save_pick(sel.pick_result.daily, picks_dir)
                    assert sel.action == "single", (mode, day, top_p, sel.action)
                    assert sel.source == "mdp", (mode, day, top_p, sel.source)
                    assert sel.pick_result.daily.double_down is None
                    dec = sel.decision
                    assert dec.objective == "longest_streak"
                    assert dec.policy_action == "single" and dec.action == "single"
                    assert dec.tail_sha256 is None
                    assert dec.degraded_reason is not None
                    assert dec.degraded_reason.startswith("longest-streak rule unavailable (")
                    assert dec.degraded_reason.endswith("); forced single")
                    assert any("longest-streak policy:" in m and dec.degraded_reason in m for m in log_records), (
                        mode, log_records
                    )
                    tested += 1

        # Verify KeyboardInterrupt and SystemExit propagate through select_pick, resolve_policy_decision, and effective_pick_bar
        mdp.DEFAULT_POLICY_PATH = REPO / "data/models/mdp_policy.npz"
        tail_policy.DEFAULT_TAIL_POLICY_PATH = REPO / "data/models/mdp_tail_policy.npz"
        strategy._mdp_cache = None
        valid_art = strategy._load_mdp()

        for sig_exc in (KeyboardInterrupt, SystemExit):
            def _raise_sig(*_a, **_kw):
                raise sig_exc("halt-now")

            strategy.longest_streak_action = _raise_sig
            ctx = strategy.DecisionContext(0.96, 0.95, True, "2027-09-26", True, valid_art, 57, "trusted")
            with pytest.raises(sig_exc):
                strategy.resolve_policy_decision(ctx, 57, False)

            strategy.longest_streak_action = orig_rule
            orig_loader = strategy._load_mdp
            strategy._load_mdp = _raise_sig
            try:
                frame = walk._predictions(date(2027, 9, 26), 0.96)
                statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
                with walk.offline(datetime(2027, 9, 26, 12, tzinfo=walk.ET)):
                    with pytest.raises(sig_exc):
                        strategy.select_pick(
                            frame, "2027-09-26", tmp_dir / "sig_picks", streak=57,
                            saver_available=False, game_statuses_detailed=statuses,
                        )
                with pytest.raises(sig_exc):
                    strategy.effective_pick_bar(56, "2027-07-15", False)
            finally:
                strategy._load_mdp = orig_loader
    finally:
        mdp.DEFAULT_POLICY_PATH = orig_base_default
        tail_policy.DEFAULT_TAIL_POLICY_PATH = orig_tail_default
        strategy.longest_streak_action = orig_rule
        strategy._mdp_cache = None
        strategy.log.removeHandler(handler)

    return {"h4_modes": len(modes), "h4_executions": tested}


def check_h5_h6_h7_h8(tmp_dir: Path) -> dict:
    strategy._mdp_cache = None
    artifact = strategy._load_mdp()
    tail_sha = artifact["tail"].sha256

    # --- H5: real run_and_pick AND scheduler save -> deliver -> commit -> restart ---
    h5_runs = 0
    for kind in ("dated", "automatic"):
        for d_str, want_action in (
            ("2027-07-15", "single"),
            ("2027-09-24", "double"),
            ("2027-09-25", "double"),
            ("2027-09-26", "double"),
        ):
            day = date.fromisoformat(d_str)
            # Part A: orchestrator.run_and_pick
            pdir_orch = tmp_dir / f"h5_orch_{kind}_{d_str}"
            pdir_orch.mkdir(parents=True, exist_ok=True)
            frame = walk._predictions(day, 0.96).head(2)
            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
            cfg_orch = walk._config(pdir_orch)
            with walk.offline(datetime.combine(day, time(12), walk.ET)), \
                    pytest.MonkeyPatch.context() as mp:
                mp.setattr(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe"))
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses)
                assert walk.install_state(pdir_orch, day, walk.State(kind, 57, 57, "used", 0.96)) == []
                _, sel, _ = orchestrator.run_and_pick(cfg_orch, d_str)
            assert sel.action == want_action
            daily = sel.pick_result.daily
            assert picks._valid_chosen_on(daily.chosen_on) is not None
            assert daily.chosen_on["kind"] == kind and daily.chosen_on["required"] is True
            assert daily.policy_decision["objective"] == "longest_streak"
            assert daily.policy_decision["tail_sha256"] == tail_sha
            assert (daily.double_down is not None) == (want_action == "double")

            # Part B: scheduler.run_single_check -> save -> _deliver_and_lock_pick -> commit -> restart check
            pdir_sched = tmp_dir / f"h5_sched_{kind}_{d_str}"
            pdir_sched.mkdir(parents=True, exist_ok=True)
            cfg_sched = walk._config(pdir_sched)
            cfg_sched["orchestrator"]["heartbeat_path"] = str(pdir_sched / ".heartbeat")
            cfg_sched["bluesky"] = {"dm_recipient": "operator-did"}
            sent_dms: list[tuple[str, str]] = []
            cascade_calls = 0

            def _cascade(*_a):
                nonlocal cascade_calls
                cascade_calls += 1
                return frame.copy(), "probe"

            now_noon = datetime.combine(day, time(12), walk.ET)
            with walk.offline(now_noon), pytest.MonkeyPatch.context() as mp:
                mp.setattr(orchestrator, "run_cascade", _cascade)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses)
                mp.setattr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
                mp.setattr(scheduler, "count_new_confirmations", lambda *a: 4)
                mp.setattr(scheduler, "_now_et", lambda: now_noon)
                mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
                mp.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
                mp.setattr(dm, "send_dm", lambda rec, txt: (sent_dms.append((rec, txt)) or f"dm-{d_str}"))
                assert walk.install_state(pdir_sched, day, walk.State(kind, 57, 57, "used", 0.96)) == []

                res = scheduler.run_single_check(d_str, list(statuses), set(), cfg_sched, 0.01)
                assert cascade_calls == 1
                assert res["should_post"] is True
                assert res["selection"].action == want_action
                saved_pick = picks.load_pick(d_str, pdir_sched)
                assert saved_pick is not None
                assert saved_pick.policy_decision["objective"] == "longest_streak"
                assert saved_pick.policy_decision["tail_sha256"] == tail_sha
                assert picks._valid_chosen_on(saved_pick.chosen_on)["kind"] == kind

                st = scheduler.SchedulerState(d_str, now_noon.isoformat(), [], [], [], False, None, None, None)
                delivered_ok = scheduler._deliver_and_lock_pick(
                    saved_pick, cfg_sched, pdir_sched, st, d_str, "lineup", selection=res["selection"]
                )
                assert delivered_ok is True
                assert len(sent_dms) == 1
                assert "Batter 0" in sent_dms[0][1]
                if want_action == "double":
                    assert "Batter 1" in sent_dms[0][1]

                committed = daily_decision.load_decision(d_str, pdir_sched)
                assert committed is not None
                assert committed["objective"] == "longest_streak"
                assert committed["tail_policy_sha256"] == tail_sha
                assert committed["action"] == want_action
                assert committed["delivery_status"] == "delivered"
                assert committed["scoreable"] is True

                # Simulated daemon restart: run_single_check sees committed decision.json and short-circuits!
                res_after_restart = scheduler.run_single_check(d_str, list(statuses), set(), cfg_sched, 0.01)
                assert cascade_calls == 1  # did NOT re-run cascade!
                assert res_after_restart["pick_result"].locked is True
            h5_runs += 1

    # --- H6: effective_pick_bar before artifacts, including FIFO non-open ---
    assert strategy.effective_pick_bar(57, "2027-07-15", False) == 0.0
    assert strategy.effective_pick_bar(57, "2027-09-26", True) == 0.0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(strategy, "_load_mdp", lambda: None)
        assert strategy.effective_pick_bar(57, "2027-07-15", False) == 0.0
    with pytest.MonkeyPatch.context() as mp:
        def _explode():
            raise RuntimeError("loader failed")
        mp.setattr(strategy, "_load_mdp", _explode)
        assert strategy.effective_pick_bar(57, "2027-07-15", False) == 0.0
        assert strategy.effective_pick_bar(56, "2027-07-15", False) is None
        assert strategy.effective_pick_bar(57, "2027-09-27", False) is None

    fifo_path = tmp_dir / "h6_fifo"
    os.mkfifo(fifo_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mdp, "DEFAULT_POLICY_PATH", fifo_path)
        mp.setattr(strategy, "_mdp_cache", None)
        old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("FIFO opened!")))
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.25)
            assert strategy.effective_pick_bar(57, "2027-07-15", False) == 0.0
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

    # --- H7: write_decision / decision_objective / is_reach57_mdp_skip ---
    pdir_h7 = tmp_dir / "h7"
    for act, sc in (("single", True), ("double", True), ("skip", False)):
        d_str = f"2027-07-1{5 + ('single', 'double', 'skip').index(act)}"
        rec = daily_decision.write_decision(
            d_str, pdir_h7, action=act, source="mdp", objective="longest_streak",
            delivery_status="delivered" if sc else "not_applicable", scoreable=sc,
            tail_policy_sha256=tail_sha,
        )
        assert rec is not None
        loaded = daily_decision.load_decision(d_str, pdir_h7)
        assert loaded == rec
        assert daily_decision.decision_objective(loaded) == "longest_streak"
        assert daily_decision.is_reach57_mdp_skip(loaded) is False

    # --- H8: cached fallback (empty cascade + warmup no_eligible) & classification restart ---
    h8_modes = ("cached_empty", "cached_warmup", "restart_attempted", "restart_delivered")
    for mode in h8_modes:
        day = date(2027, 9, 25)
        d_str = day.isoformat()
        pdir_h8 = tmp_dir / f"h8_{mode}"
        pdir_h8.mkdir(parents=True, exist_ok=True)
        frame = walk._predictions(day, 0.96).head(2)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        cfg = walk._config(pdir_h8)
        cfg["orchestrator"]["heartbeat_path"] = str(pdir_h8 / ".heartbeat")
        cfg["bluesky"] = {"dm_recipient": "operator-did"}
        now_noon = datetime.combine(day, time(12), walk.ET)

        with walk.offline(now_noon), pytest.MonkeyPatch.context() as mp:
            mp.setattr(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe"))
            mp.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses)
            mp.setattr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
            mp.setattr(scheduler, "_now_et", lambda: now_noon)
            mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
            mp.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
            mp.setattr(scheduler, "_alert_delivery_outcome_unknown", Mock())
            mp.setattr(dm, "send_dm", Mock(return_value="dm-h8"))
            assert walk.install_state(pdir_h8, day, walk.State("automatic", 57, 57, "used", 0.96)) == []

            _, sel0, _ = orchestrator.run_and_pick(cfg, d_str)
            daily0 = sel0.pick_result.daily
            assert daily0.double_down is not None  # 2027-09-25 is d=2 -> double!
            if mode == "restart_attempted":
                daily0.delivery_attempted = True
            elif mode == "restart_delivered":
                daily0.notification_sent = True
                daily0.notification_id = "dm-prior"
                daily0.delivered_at = now_noon.isoformat()
            picks.save_pick(daily0, pdir_h8)

            reloaded = picks.load_pick(d_str, pdir_h8)
            st = scheduler.SchedulerState(d_str, now_noon.isoformat(), [], [], [], False, None, None, None)

            if mode == "cached_empty":
                mp.setattr(orchestrator, "run_cascade", lambda *a: (frame.iloc[:0], "probe"))
                refresh = scheduler._refresh_pick_at_fallback_decision(cfg, d_str, reloaded, 0.01)
                assert refresh.selection is None and refresh.daily is reloaded
                assert scheduler._deliver_and_lock_pick(
                    refresh.daily, cfg, pdir_h8, st, d_str, "final fallback", selection=refresh.selection
                )
            elif mode == "cached_warmup":
                warmup_statuses = {int(pk): {"abstract": "P", "detailed": "Warmup"} for pk in frame["game_pk"]}
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a: warmup_statuses)
                refresh = scheduler._refresh_pick_at_fallback_decision(cfg, d_str, reloaded, 0.01)
                assert refresh.selection is not None and refresh.selection.decision is None
                assert scheduler._deliver_and_lock_pick(
                    refresh.daily, cfg, pdir_h8, st, d_str, "final fallback", selection=refresh.selection
                )
            else:
                # Restart path via run_single_check locked classification -> _write_classification_decision
                mp.setattr(scheduler, "count_new_confirmations", lambda *a: 0)
                chk = scheduler.run_single_check(d_str, list(statuses), set(), cfg, 0.01)
                assert chk["pick_result"].locked is True and chk["selection"] is None
                ld = chk["pick_result"].daily
                scheduler._write_classification_decision(
                    pdir_h8, d_str,
                    action=("double" if ld.double_down else "single"),
                    delivered=picks.pick_was_delivered(ld),
                    attempted=bool(ld.delivery_attempted),
                    primary=scheduler._row_from_daily(ld.pick),
                    double_down=scheduler._row_from_daily(ld.double_down),
                    state=st, daily=ld,
                )

            rec = daily_decision.load_decision(d_str, pdir_h8)
            assert rec is not None, mode
            assert rec["objective"] == "longest_streak"
            assert rec["tail_policy_sha256"] == tail_sha
            assert rec["action"] == "double"
            assert rec["primary"]["batter_id"] == 100
            assert rec["double_down"]["batter_id"] == 101
            want_status = "locked_unconfirmed" if mode == "restart_attempted" else "delivered"
            assert rec["delivery_status"] == want_status

    return {"h5_runs": h5_runs, "h8_modes": len(h8_modes)}


def check_h9_and_a10_test_diff() -> dict:
    changed_tests = subprocess.run(
        ["git", "diff", "--name-only", "0a2bec10", "813935bc", "--", "tests"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=15,
    ).stdout.splitlines()
    assert sorted(changed_tests) == [
        "tests/scripts/season_walk/test_season_walk.py",
        "tests/simulate/test_tail_policy.py",
        "tests/test_wr_g_guard.py",
        "tests/test_wr_h_longest.py",
    ], changed_tests

    # Verify test_season_walk.py has ZERO removed lines (only additions)
    sw_diff = subprocess.run(
        ["git", "diff", "-U0", "0a2bec10", "813935bc", "--", "tests/scripts/season_walk/test_season_walk.py"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    removed_sw = [ln for ln in sw_diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert removed_sw == [], removed_sw

    # Verify test_tail_policy.py removed ONLY the one ruled H9 assertion/comment
    tp_diff = subprocess.run(
        ["git", "diff", "-U0", "0a2bec10", "813935bc", "--", "tests/simulate/test_tail_policy.py"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    removed_tp = [ln[1:].strip() for ln in tp_diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert removed_tp == [
        "# lookup_action already returns skip for these; the base table owns them.",
        "assert mdp_objective(57, 10) == OBJECTIVE_REACH57",
    ], removed_tp

    # Verify test_wr_g_guard.py changed ONLY inside test_resolver_and_direct_bar_grid as A10 specifies
    g_diff = subprocess.run(
        ["git", "diff", "-U0", "0a2bec10", "813935bc", "--", "tests/test_wr_g_guard.py"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    removed_g = [ln[1:].strip() for ln in g_diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert removed_g == [
        "calls, changed = 0, 0",
        "for s, saver in product(range(58), (False, True)):",
        "assert calls == (stop - start) * 58 * 2 * 5 * 3",
    ], removed_g

    return {"changed_test_files": changed_tests}


def test_probe_2_all(tmp_path):
    r1 = check_h1_h2_h3()
    r2 = check_h4_and_a3_degradations(tmp_path)
    r3 = check_h5_h6_h7_h8(tmp_path)
    r4 = check_h9_and_a10_test_diff()
    assert r1["h1_h2_h3_cases"] == 271 * 4 * 2 * 4 * 4
    assert r2["h4_modes"] == 17
    assert r3["h5_runs"] == 8
    assert len(r4["changed_test_files"]) == 4


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp_dir = SCRATCH / "p2"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    summary = {
        "h1_h2_h3": check_h1_h2_h3(),
        "h4_a3": check_h4_and_a3_degradations(tmp_dir),
        "h5_h6_h7_h8": check_h5_h6_h7_h8(tmp_dir),
        "h9_a10_diff": check_h9_and_a10_test_diff(),
    }
    out_log = OUT_DIR / "probe_2_rows_h1_h9_and_a3.log"
    out_log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
