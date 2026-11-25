"""Adversarial probe for Item 2 (part A): base validation, H4/A3/R1, H6, interrupts, and B1 class hunt."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import date, datetime, time
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from bts import daily_decision, orchestrator, picks, scheduler, strategy
from bts.health import tail_policy as tail_health
from bts.simulate import mdp, tail_policy as tp
from scripts.audit.season_walk import walk


def _rebind_pair(folder: Path, base_arrays: dict) -> tuple[Path, Path, str, tp.TailPolicy]:
    base_path = folder / "base.npz"
    tail_path = folder / "tail.npz"
    np.savez_compressed(base_path, **base_arrays)
    base_sha = sha256(base_path.read_bytes()).hexdigest()
    shipped_tail = tp.load_tail_policy(tp.DEFAULT_TAIL_POLICY_PATH)
    tp.save_tail_policy(replace(shipped_tail, base_policy_sha256=base_sha), tail_path)
    paired_tail = tp.load_tail_policy(tail_path, expected_base_sha=base_sha)
    return base_path, tail_path, base_sha, paired_tail


def _run_scheduler_delivery(folder: Path, base_path: Path, tail_path: Path, day: date) -> dict:
    picks_dir = folder / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    frame = walk._predictions(day, 0.84).head(2)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
    cfg = walk._config(picks_dir)
    cfg["orchestrator"]["heartbeat_path"] = str(picks_dir / ".heartbeat")
    cfg["bluesky"] = {"dm_recipient": "synthetic"}
    text = day.isoformat()
    send = Mock(return_value="msg-1")
    with (
        walk.offline(datetime.combine(day, time(12), walk.ET)),
        patch.object(mdp, "DEFAULT_POLICY_PATH", base_path),
        patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", tail_path),
        patch.object(strategy, "_mdp_cache", None),
        patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")),
        patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
        patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
        patch.object(scheduler, "count_new_confirmations", lambda *a: 6),
        patch.object(
            scheduler,
            "_lock_decision_from_predictions",
            lambda *a, **kw: scheduler.LockDecision(True, None, True),
        ),
        patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
        patch.object(scheduler, "_void_committed_games", lambda *a: {}),
        patch("bts.dm.send_dm", send),
    ):
        problems = walk.install_state(picks_dir, day, walk.State("dated", 57, 57, "used", 0.84))
        assert problems == [], problems
        res = scheduler.run_single_check(text, list(statuses), set(), cfg, 0.01)
        saved = picks.load_pick(text, picks_dir)
        assert saved is not None
        scheduler_state = scheduler.SchedulerState(text, None, [], [], [], False, None, None, None)
        delivered = scheduler._deliver_and_lock_pick(
            saved, cfg, picks_dir, scheduler_state, text, "final fallback", selection=res["selection"]
        )
        assert delivered is True
        rec = daily_decision.load_decision(text, picks_dir)
        msg = send.call_args.args[1]
        return {
            "action": rec["action"],
            "objective": rec["objective"],
            "tail_sha": rec["tail_policy_sha256"],
            "degraded_reason": rec["degraded_reason"],
            "double_down": saved.double_down.batter_name if saved.double_down else None,
            "days_effective": saved.policy_decision["days_effective"],
            "dm": msg,
        }


def _select_at_57(folder: Path, base_path: Path, tail_path: Path, day: date = date(2027, 9, 26)):
    picks_dir = folder / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    frame = walk._predictions(day, 0.84).head(2)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
    with (
        walk.offline(datetime.combine(day, time(12), walk.ET)),
        patch.object(mdp, "DEFAULT_POLICY_PATH", base_path),
        patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", tail_path),
        patch.object(strategy, "_mdp_cache", None),
    ):
        return strategy.select_pick(
            frame, day.isoformat(), picks_dir, streak=57, saver_available=False,
            allow_double=True, game_statuses_detailed=statuses,
        )


def _assert_forced_single(sel, label: str) -> None:
    dec = sel.decision
    assert sel.action == "single", f"{label}: got action={sel.action}"
    assert dec.action == "single" and dec.policy_action == "single", f"{label}: {dec}"
    assert dec.objective == "longest_streak" and dec.source == "mdp", f"{label}: {dec}"
    assert dec.tail_sha256 is None, f"{label}: non-null tail_sha256={dec.tail_sha256}"
    assert dec.degraded_reason and dec.degraded_reason.startswith("longest-streak rule unavailable (")
    assert dec.degraded_reason.endswith("); forced single"), f"{label}: {dec.degraded_reason}"


def main() -> None:
    with np.load(mdp.DEFAULT_POLICY_PATH) as f:
        shipped_base = dict(f)
    shipped_table = shipped_base["policy_table"]
    shipped_bounds = shipped_base["boundaries"]

    with tempfile.TemporaryDirectory(prefix="wr-h2-item2-") as tmp_str:
        root = Path(tmp_str)

        # 1. Confirm Round 1 B1 (scalar array) is closed on its exact inputs
        b1_dir = root / "b1_scalar"
        b1_dir.mkdir()
        b_path, t_path, _, _ = _rebind_pair(
            b1_dir, shipped_base | {"policy_table": np.array(0, dtype=np.int8), "season_length": np.array(180)}
        )
        assert tail_health.check(base_path=b_path, tail_path=t_path, today=date(2027, 9, 26)) == []
        b1_out = _run_scheduler_delivery(b1_dir, b_path, t_path, date(2027, 9, 26))
        assert b1_out["action"] == "single" and b1_out["double_down"] is None
        assert b1_out["tail_sha"] is None and b1_out["degraded_reason"].endswith("); forced single")
        assert "Batter 0" in b1_out["dm"] and "Batter 1" not in b1_out["dm"]
        print("OK: Round 1 B1 scalar base closed ->", b1_out["action"], b1_out["degraded_reason"])

        # 2. Healthy controls (short table on 09-23 and 09-26, s=57 exact, uint8, >i4, 0 boundaries, readonly, memmap)
        short_dir = root / "healthy_short"
        short_dir.mkdir()
        sb_path, st_path, _, paired_short = _rebind_pair(
            short_dir, shipped_base | {"policy_table": shipped_table[:, :2], "season_length": np.array(1)}
        )
        out_923 = _run_scheduler_delivery(root / "short_923", sb_path, st_path, date(2027, 9, 23))
        out_926 = _run_scheduler_delivery(root / "short_926", sb_path, st_path, date(2027, 9, 26))
        assert out_923["action"] == "single" and out_923["days_effective"] == 4 and out_923["tail_sha"] == paired_short.sha256
        assert out_926["action"] == "double" and out_926["days_effective"] == 1 and out_926["tail_sha"] == paired_short.sha256

        for cname, cdict in [
            ("s57_exact", shipped_base | {"policy_table": shipped_table[:57]}),
            ("uint8", shipped_base | {"policy_table": shipped_table.astype(np.uint8)}),
            ("big_endian_i4", shipped_base | {"policy_table": shipped_table.astype(">i4")}),
            ("one_bin_base", shipped_base | {"policy_table": shipped_table[:, :, :, :1], "boundaries": np.array([], dtype=float)}),
        ]:
            cdir = root / f"healthy_{cname}"
            cdir.mkdir()
            cp_b, cp_t, _, cp_tail = _rebind_pair(cdir, cdict)
            sel = _select_at_57(cdir, cp_b, cp_t, date(2027, 9, 26))
            assert sel.action == "double" and sel.decision.tail_sha256 == cp_tail.sha256 and sel.decision.degraded_reason is None
            with patch.object(mdp, "DEFAULT_POLICY_PATH", cp_b), patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", cp_t), patch.object(strategy, "_mdp_cache", None):
                loaded = strategy._load_mdp()
            act, _ = mdp.lookup_action_guarded(loaded["policy_table"], loaded["boundaries"], 56, 1, False, 0.84, loaded["season_length"])
            assert act in ("skip", "single", "double")

        # Read-only and memmap arrays in context (healthy and 1-bad-cell)
        shipped_ctx = strategy._load_mdp()
        ro_good = shipped_table.copy()
        ro_good.flags.writeable = False
        mmap_file = root / "table.mmap"
        mm_good = np.memmap(mmap_file, dtype=np.int8, mode="w+", shape=shipped_table.shape)
        mm_good[:] = shipped_table[:]
        mm_ro = np.memmap(mmap_file, dtype=np.int8, mode="r", shape=shipped_table.shape)
        for label, arr in [("readonly_good", ro_good), ("memmap_ro_good", mm_ro)]:
            ctx = strategy.DecisionContext(0.84, 0.83, True, "2027-09-26", True, shipped_ctx | {"policy_table": arr}, 57, "trusted")
            d = strategy.resolve_policy_decision(ctx, 57, False)
            assert d.action == "double" and d.tail_sha256 == shipped_ctx["tail"].sha256, label

        mm_bad_file = root / "table_bad.mmap"
        mm_bad_w = np.memmap(mm_bad_file, dtype=np.int8, mode="w+", shape=shipped_table.shape)
        mm_bad_w[:] = shipped_table[:]
        mm_bad_w[28, 90, 1, 2] = 3
        mm_bad_ro = np.memmap(mm_bad_file, dtype=np.int8, mode="r", shape=shipped_table.shape)
        ro_bad = shipped_table.copy()
        ro_bad[56, 180, 1, 4] = -1
        ro_bad.flags.writeable = False
        for label, arr in [("readonly_bad_cell", ro_bad), ("memmap_ro_bad_cell", mm_bad_ro)]:
            ctx = strategy.DecisionContext(0.84, 0.83, True, "2027-09-26", True, shipped_ctx | {"policy_table": arr}, 57, "trusted")
            d = strategy.resolve_policy_decision(ctx, 57, False)
            assert d.action == "single" and d.tail_sha256 is None and d.degraded_reason.endswith("); forced single"), label
        print("OK: Healthy controls + readonly/memmap passed")

        # 3. Malformed on-disk base files (each with valid tail re-bound to its sha)
        one_cell_row57 = shipped_table.copy(); one_cell_row57[57, 0, 0, 0] = 3
        one_cell_mid = shipped_table.copy(); one_cell_mid[31, 44, 0, 2] = 2**62 if False else 99
        huge_i64 = shipped_table.astype(np.int64); huge_i64[10, 10, 1, 3] = 2**62
        huge_u64 = shipped_table.astype(np.uint64); huge_u64[10, 10, 1, 3] = 2**63 + 7
        neg_i64 = shipped_table.astype(np.int64); neg_i64[0, 1, 0, 0] = -(2**62)

        malformed_bases = {
            "2d": shipped_base | {"policy_table": shipped_table[0, :, 0, :]},
            "5d": shipped_base | {"policy_table": shipped_table[..., None]},
            "zero_s": shipped_base | {"policy_table": np.zeros((0, 181, 2, 5), dtype=np.int8)},
            "zero_d": shipped_base | {"policy_table": np.zeros((58, 0, 2, 5), dtype=np.int8)},
            "zero_v": shipped_base | {"policy_table": np.zeros((58, 181, 0, 5), dtype=np.int8)},
            "zero_q": shipped_base | {"policy_table": np.zeros((58, 181, 2, 0), dtype=np.int8)},
            "s56": shipped_base | {"policy_table": shipped_table[:56]},
            "saver_1": shipped_base | {"policy_table": shipped_table[:, :, :1, :]},
            "saver_3": shipped_base | {"policy_table": np.concatenate([shipped_table, shipped_table[:, :, :1, :]], axis=2)},
            "day_plus1": shipped_base | {"policy_table": np.concatenate([shipped_table, shipped_table[:, :1, :, :]], axis=1)},
            "bin_plus1": shipped_base | {"policy_table": np.concatenate([shipped_table, shipped_table[:, :, :, :1]], axis=3)},
            "dtype_bool": shipped_base | {"policy_table": np.zeros(shipped_table.shape, dtype=bool)},
            "dtype_float32": shipped_base | {"policy_table": shipped_table.astype(np.float32)},
            "dtype_complex": shipped_base | {"policy_table": shipped_table.astype(np.complex128)},
            "dtype_str": shipped_base | {"policy_table": shipped_table.astype("<U1")},
            "dtype_struct": shipped_base | {"policy_table": np.zeros(shipped_table.shape, dtype=[("a", "i1")])},
            "one_cell_row57_val3": shipped_base | {"policy_table": one_cell_row57},
            "one_cell_mid_val99": shipped_base | {"policy_table": one_cell_mid},
            "huge_i64": shipped_base | {"policy_table": huge_i64},
            "huge_u64": shipped_base | {"policy_table": huge_u64},
            "neg_i64": shipped_base | {"policy_table": neg_i64},
            "horizon_zero": shipped_base | {"season_length": np.array(0)},
            "horizon_neg": shipped_base | {"season_length": np.array(-5)},
            "horizon_huge_mismatch": shipped_base | {"season_length": np.array(10**12)},
            "horizon_nan": shipped_base | {"season_length": np.array(np.nan)},
            "horizon_str": shipped_base | {"season_length": np.array("not_an_int")},
            "horizon_1d": shipped_base | {"season_length": np.array([180, 180])},
            "bounds_scalar": shipped_base | {"boundaries": np.array(0.8)},
            "missing_table_key": {k: v for k, v in shipped_base.items() if k != "policy_table"},
            "missing_bounds_key": {k: v for k, v in shipped_base.items() if k != "boundaries"},
            "missing_horizon_key": {k: v for k, v in shipped_base.items() if k != "season_length"},
        }
        for mname, mdict in malformed_bases.items():
            mdir = root / f"mal_{mname}"
            mdir.mkdir()
            mb, mt, _, _ = _rebind_pair(mdir, mdict)
            sel = _select_at_57(mdir, mb, mt, date(2027, 9, 26))
            _assert_forced_single(sel, mname)

        # Additional non-npz / oversize / dir / tail-broken H4 cases
        extra_cases = ["base_dir", "base_oversize", "tail_oversize", "both_corrupt"]
        for ename in extra_cases:
            edir = root / f"extra_{ename}"
            edir.mkdir()
            eb = edir / "base.npz"
            et = edir / "tail.npz"
            shutil.copyfile(mdp.DEFAULT_POLICY_PATH, eb)
            shutil.copyfile(tp.DEFAULT_TAIL_POLICY_PATH, et)
            if ename == "base_dir":
                eb.unlink(); eb.mkdir()
            elif ename == "base_oversize":
                eb.write_bytes(b"x" * (tp.MAX_ARTIFACT_BYTES + 10))
            elif ename == "tail_oversize":
                et.write_bytes(b"x" * (tp.MAX_ARTIFACT_BYTES + 10))
            elif ename == "both_corrupt":
                eb.write_bytes(b"bad1"); et.write_bytes(b"bad2")
            sel = _select_at_57(edir, eb, et, date(2027, 9, 26))
            _assert_forced_single(sel, ename)
        print(f"OK: {len(malformed_bases) + len(extra_cases)} malformed artifact cases forced single")

        # 4. H6 + Interrupt propagation (both rule and loader, both select_pick and effective_pick_bar)
        for intr in (KeyboardInterrupt, SystemExit):
            def raise_intr(*a, **kw):
                raise intr("stop-now")

            with patch.object(strategy, "_load_mdp", raise_intr):
                assert strategy.effective_pick_bar(57, "2027-07-15", False) == 0.0
                try:
                    strategy.effective_pick_bar(56, "2027-07-15", False)
                except intr:
                    pass
                else:
                    raise AssertionError(f"effective_pick_bar swallowed {intr}")

                try:
                    _select_at_57(root / "intr_sel", mdp.DEFAULT_POLICY_PATH, tp.DEFAULT_TAIL_POLICY_PATH)
                except intr:
                    pass
                else:
                    raise AssertionError(f"select_pick swallowed {intr}")
        print("OK: H6 and interrupt propagation verified")

        # 5. Hunt for OTHER members of B1's class (decoded base that cannot answer a reach-57 lookup!)
        print("\n--- Hunting for unclosed members of B1's class ---")
        b1_class_candidates = {
            "bounds_str_1d": shipped_base | {"boundaries": np.array(["0.76", "0.78", "0.80", "0.82"])},
            "bounds_2d_col": shipped_base | {"boundaries": shipped_bounds.reshape(4, 1)},
            "bounds_2d_empty_cols": shipped_base | {"boundaries": np.zeros((4, 0), dtype=float)},
            "bounds_complex_1d": shipped_base | {"boundaries": np.array([0.76 + 0j, 0.78 + 0j, 0.80 + 0j, 0.82 + 0j])},
            "bounds_bytes_1d": shipped_base | {"boundaries": np.array([b"a", b"b", b"c", b"d"])},
            "bounds_struct_1d": shipped_base | {"boundaries": np.zeros(4, dtype=[("cut", "f8")])},
            "bounds_nan": shipped_base | {"boundaries": np.array([np.nan, 0.78, 0.80, 0.82])},
            "bounds_inf": shipped_base | {"boundaries": np.array([np.inf, np.inf, np.inf, np.inf])},
            "bounds_unsorted": shipped_base | {"boundaries": np.array([0.85, 0.75, 0.80, 0.82])},
            "bounds_bool_1d": shipped_base | {"boundaries": np.array([False, True, False, True])},
        }
        for kname, kdict in b1_class_candidates.items():
            kdir = root / f"hunt_{kname}"
            kdir.mkdir()
            kb, kt, _, _ = _rebind_pair(kdir, kdict)
            alerts = tail_health.check(base_path=kb, tail_path=kt, today=date(2027, 9, 26))
            with patch.object(mdp, "DEFAULT_POLICY_PATH", kb), patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", kt), patch.object(strategy, "_mdp_cache", None):
                decoded = strategy._load_mdp()
            try:
                r57_act, _ = mdp.lookup_action_guarded(
                    decoded["policy_table"], decoded["boundaries"], 56, 1, False, 0.84, decoded["season_length"]
                )
                r57_status = f"returned {r57_act!r} (bar={strategy.effective_pick_bar(56, '2027-09-26', False, mdp=decoded)!r})"
            except Exception as exc:
                r57_status = f"RAISED {type(exc).__name__}: {exc}"
            sched_out = _run_scheduler_delivery(kdir, kb, kt, date(2027, 9, 26))
            print(
                f"  {kname}: health_alerts={len(alerts)} | reach57_lookup={r57_status} | "
                f"scheduler_at_57={sched_out['action']} (dd={sched_out['double_down']}, "
                f"tail_sha={bool(sched_out['tail_sha'])}, degraded={sched_out['degraded_reason']})"
            )

        # Also check MaskedArray in-memory with a masked out-of-range cell
        raw_bad = shipped_table.copy()
        raw_bad[56, 1, 0, 4] = 99
        mask = np.zeros(shipped_table.shape, dtype=bool)
        mask[56, 1, 0, 4] = True
        masked_arr = np.ma.masked_array(raw_bad, mask=mask)
        try:
            mdp.lookup_action_guarded(masked_arr, shipped_ctx["boundaries"], 56, 1, False, 0.99, 180)
            m_r57 = "returned"
        except Exception as exc:
            m_r57 = f"RAISED {type(exc).__name__}: {exc}"
        m_dec = strategy.resolve_policy_decision(
            strategy.DecisionContext(0.99, 0.98, True, "2027-09-26", True, shipped_ctx | {"policy_table": masked_arr}, 57, "trusted"),
            57, False,
        )
        print(f"  masked_array_in_ctx: reach57_lookup={m_r57} | resolver_at_57={m_dec.action} (degraded={m_dec.degraded_reason})")


if __name__ == "__main__":
    main()
