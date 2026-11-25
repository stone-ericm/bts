"""Item 4 regression probe: decision records, cross-revision readers, consumers, and all 5 _write_commit_decision early-return scenarios (S1-S5)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
from click.testing import CliRunner

from bts import cli, daily_decision, day_outcome, dm, orchestrator, picks, posting, scheduler, skip_policy_shadow, strategy
from scripts.audit.boundary_shadow_census import CensusHalt, run_census
from scripts.audit.season_walk import walk

REVIEW_DIR = REPO / ".codex-review" / "wr-g"
BASE_SRC = REVIEW_DIR / "base_f83e0d0b" / "src"
ROLLBACK_DIR = REVIEW_DIR / "rollback_619abda7"
HEAD_SRC = REPO / "src"

GUARD_SINGLE = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}


def ensure_rollback_extracted() -> Path:
    src_path = ROLLBACK_DIR / "src"
    if not (src_path / "bts" / "__init__.py").exists():
        ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
        tar_proc = subprocess.run(
            ["git", "archive", "619abda7", "src"],
            cwd=REPO, check=True, capture_output=True, timeout=15,
        )
        subprocess.run(
            ["tar", "-x", "-C", str(ROLLBACK_DIR)],
            input=tar_proc.stdout, check=True, timeout=15,
        )
    return src_path


def test_record_paths_and_consumers() -> dict:
    rollback_src = ensure_rollback_extracted()
    day_iso = "2027-03-24"
    day_obj = date(2027, 3, 24)
    now_et = walk.REAL_DATETIME(2027, 3, 24, 18, 25, tzinfo=walk.ET)
    frame = walk._predictions(day_obj, 0.96)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in frame["game_pk"]}

    verified = {}

    for mode, expected_status in (("dm", "delivered"), ("public", "delivered"), ("private", "private_locked")):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix=f"tmp_p04r_{mode}_") as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            cfg = walk._config(pdir)
            cfg["scheduler"]["pick_delivery"] = mode
            cfg["scheduler"]["live_forward_capture_on_lock"] = False
            cfg["bluesky"] = {"dm_recipient": "eric.test"}
            with walk.offline(now_et), \
                    patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                    patch.object(scheduler, "_now_et", lambda: now_et), \
                    patch.object(scheduler, "_void_committed_games", lambda *a: {}), \
                    patch.object(dm, "send_dm", lambda r, t: "dm-ok-1"), \
                    patch.object(posting, "post_to_bluesky", lambda t: "at://post/1"):
                assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
                _, sel, _ = orchestrator.run_and_pick(cfg, day_iso)
                daily = sel.pick_result.daily
                picks.save_pick(daily, pdir)
                st = scheduler.SchedulerState(
                    date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                    runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                assert scheduler._deliver_and_lock_pick(daily, cfg, pdir, st, day_iso, "lineup", selection=sel) is True
                rec = daily_decision.load_decision(day_iso, pdir)
                assert rec["delivery_status"] == expected_status
                assert rec["scoreable"] is True
                assert rec["policy_guard"] == GUARD_SINGLE
                saved_pick = picks.load_pick(day_iso, pdir)
                assert saved_pick.policy_decision["guard"] == GUARD_SINGLE
                verified[f"fresh_{mode}"] = rec["delivery_status"]

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p04r_clslock_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        cfg = walk._config(pdir)
        with walk.offline(now_et), \
                patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda d: statuses):
            assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
            _, sel, _ = orchestrator.run_and_pick(cfg, day_iso)
            daily = sel.pick_result.daily
            daily.notification_sent = True
            daily.notification_id = "dm-prior-1"
            picks.save_pick(daily, pdir)
            chk = scheduler.run_single_check(day_iso, [70000, 70001], set(), cfg, 0.03)
            assert chk["pick_result"].locked is True and chk["selection"] is None
            st = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=True, pick_locked_at=now_et.isoformat(), result_status=None, next_wakeup=None,
            )
            scheduler._write_classification_decision(
                pdir, day_iso, action="single", delivered=True, attempted=True,
                primary=scheduler._row_from_daily(daily.pick), double_down=None, state=st, daily=daily,
            )
            rec_cls = daily_decision.load_decision(day_iso, pdir)
            assert rec_cls["delivery_status"] == "delivered" and rec_cls["policy_guard"] == GUARD_SINGLE
            raw_cls_bytes = daily_decision.decision_path(day_iso, pdir).read_bytes()
            scheduler._write_classification_decision(
                pdir, day_iso, action="single", delivered=True, attempted=True,
                primary=scheduler._row_from_daily(daily.pick), double_down=None, state=st, daily=daily,
            )
            assert daily_decision.decision_path(day_iso, pdir).read_bytes() == raw_cls_bytes
            st_eod = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                final_skip_candidate={"source": "mdp", "objective": "reach57"},
            )
            scheduler._write_endofday_skip(pdir, day_iso, st_eod)
            assert daily_decision.decision_path(day_iso, pdir).read_bytes() == raw_cls_bytes
            verified["classification_lock_and_byte_preserve"] = "PASS"

    for fb_case in ("exception_none_sel", "nan_preds_empty_sel"):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix=f"tmp_p04r_fb_{fb_case}_") as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            cfg = walk._config(pdir)
            cfg["scheduler"]["pick_delivery"] = "dm"
            cfg["scheduler"]["live_forward_capture_on_lock"] = False
            cfg["bluesky"] = {"dm_recipient": "eric.test"}
            nan_frame = frame.copy()
            nan_frame["p_game_hit"] = float("nan")

            def fake_cascade(tiers, d, _case=fb_case):
                if _case == "exception_none_sel":
                    raise RuntimeError("cascade boom")
                return nan_frame.copy(), "local"

            with walk.offline(now_et), \
                    patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                    patch.object(scheduler, "_now_et", lambda: now_et), \
                    patch.object(scheduler, "_void_committed_games", lambda *a: {}), \
                    patch.object(dm, "send_dm", lambda r, t: "dm-cached-1"):
                assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
                with patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")):
                    _, sel0, _ = orchestrator.run_and_pick(cfg, day_iso)
                cached = sel0.pick_result.daily
                picks.save_pick(cached, pdir)
                with patch.object(orchestrator, "run_cascade", fake_cascade):
                    refresh = scheduler._refresh_pick_at_fallback_decision(cfg, day_iso, cached, 0.03)
                st = scheduler.SchedulerState(
                    date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                    runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                assert scheduler._deliver_and_lock_pick(
                    refresh.daily, cfg, pdir, st, day_iso, "fallback", selection=refresh.selection
                ) is True
                rec = daily_decision.load_decision(day_iso, pdir)
                assert rec["policy_guard"] == GUARD_SINGLE and rec["scoreable"] is True
                verified[f"cached_fallback_{fb_case}"] = rec["policy_guard"]

    for rec_mode in ("direct", "classification"):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix=f"tmp_p04r_att_{rec_mode}_") as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            cfg = walk._config(pdir)
            cfg["scheduler"]["pick_delivery"] = "dm"
            cfg["scheduler"]["live_forward_capture_on_lock"] = False
            cfg["bluesky"] = {"dm_recipient": "eric.test"}

            class CrashMidSend(BaseException):
                pass

            send_calls = {"n": 0}

            def crash_send(r, t, _c=send_calls):
                _c["n"] += 1
                raise CrashMidSend("killed mid-send")

            with walk.offline(now_et), \
                    patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                    patch.object(scheduler, "_now_et", lambda: now_et), \
                    patch.object(scheduler, "_void_committed_games", lambda *a: {}), \
                    patch.object(scheduler, "_alert_delivery_outcome_unknown", lambda *a, **k: None):
                assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
                _, sel, _ = orchestrator.run_and_pick(cfg, day_iso)
                daily = sel.pick_result.daily
                picks.save_pick(daily, pdir)
                st = scheduler.SchedulerState(
                    date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                    runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                with patch.object(dm, "send_dm", crash_send):
                    try:
                        scheduler._deliver_and_lock_pick(daily, cfg, pdir, st, day_iso, "lineup", selection=sel)
                    except CrashMidSend:
                        pass
                assert send_calls["n"] == 1
                persisted = picks.load_pick(day_iso, pdir)
                assert persisted.delivery_attempted is True
                assert daily_decision.load_decision(day_iso, pdir) is None

                def forbid_second_send(r, t):
                    raise AssertionError("second send must not happen")

                with patch.object(dm, "send_dm", forbid_second_send):
                    st2 = scheduler.SchedulerState(
                        date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                        runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                    )
                    if rec_mode == "direct":
                        ok = scheduler._deliver_and_lock_pick(persisted, cfg, pdir, st2, day_iso, "lineup", selection=None)
                        assert ok is False
                    else:
                        scheduler._write_classification_decision(
                            pdir, day_iso, action="single", delivered=False, attempted=True,
                            primary=scheduler._row_from_daily(persisted.pick), double_down=None,
                            state=st2, daily=persisted,
                        )
                rec = daily_decision.load_decision(day_iso, pdir)
                assert rec["delivery_status"] == "locked_unconfirmed"
                assert rec["scoreable"] is True
                assert rec["policy_guard"] == GUARD_SINGLE
                verified[f"attempt_restart_{rec_mode}"] = rec["delivery_status"]

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p04r_skip_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        st = scheduler.SchedulerState(
            date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
            runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            final_skip_candidate={
                "source": "mdp", "objective": "reach57", "streak": 56, "saver_available": False,
                "primary": {"batter_id": 101, "batter_name": "Batter 0", "team": "T0", "game_pk": 70000, "p_game_hit": 0.96},
                "policy_guard": GUARD_SINGLE,
            },
        )
        scheduler._write_endofday_skip(pdir, day_iso, st)
        skip_rec = daily_decision.load_decision(day_iso, pdir)
        assert "policy_guard" in skip_rec and skip_rec["policy_guard"] is None
        verified["eod_skip_policy_guard"] = skip_rec["policy_guard"]

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p04r_compat_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        with np.load(REPO / "data" / "models" / "mdp_policy.npz") as npz:
            tbl = npz["policy_table"]
            b_edges = npz["boundaries"].tolist()
        u_streak = next(
            s for s in range(56, -1, -1)
            if int(tbl[s, 180, 0, 2]) == int(tbl[s, 180, 0, 3]) != 0
        )
        u_p = float(b_edges[2] + b_edges[3]) / 2.0

        g_day, u_day, s_day = "2026-03-25", "2026-03-26", "2026-03-27"
        now_2026 = walk.REAL_DATETIME(2026, 3, 25, 18, 0, tzinfo=walk.ET)
        with walk.offline(now_2026), patch.object(picks, "get_game_statuses_detailed", lambda d: statuses):
            f_g = walk._predictions(date(2026, 3, 25), 0.96)
            st_g = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in f_g["game_pk"]}
            sel_g = strategy.select_pick(f_g, g_day, pdir, streak=50, saver_available=False,
                                         game_statuses_detailed=st_g, require_detailed_statuses=True)
            assert sel_g.decision.guard == GUARD_SINGLE
            sel_g.pick_result.daily.notification_sent = True
            sel_g.pick_result.daily.notification_id = "dm-g"
            picks.save_pick(sel_g.pick_result.daily, pdir)
            daily_decision.write_decision(
                g_day, pdir, action="single", source="mdp", objective="reach57",
                primary=sel_g.primary_candidate, streak=50, saver_available=False,
                delivery_status="delivered", scoreable=True, policy_guard=sel_g.decision.guard,
                now=datetime(2026, 3, 25, 22, 0, tzinfo=timezone.utc),
            )

            f_u = walk._predictions(date(2026, 3, 26), u_p)
            st_u = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in f_u["game_pk"]}
            sel_u = strategy.select_pick(f_u, u_day, pdir, streak=u_streak, saver_available=False,
                                         allow_double=False,
                                         game_statuses_detailed=st_u, require_detailed_statuses=True)
            assert sel_u.decision.guard is None and sel_u.action == "single"
            sel_u.pick_result.daily.notification_sent = True
            sel_u.pick_result.daily.notification_id = "dm-u"
            picks.save_pick(sel_u.pick_result.daily, pdir)
            daily_decision.write_decision(
                u_day, pdir, action="single", source="mdp", objective="reach57",
                primary=sel_u.primary_candidate, streak=u_streak, saver_available=False,
                allow_double=False, delivery_status="delivered", scoreable=True, policy_guard=None,
                now=datetime(2026, 3, 26, 22, 0, tzinfo=timezone.utc),
            )

            f_s = walk._predictions(date(2026, 3, 27), 0.99)
            st_s = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in f_s["game_pk"]}
            sel_s = strategy.select_pick(f_s, s_day, pdir, streak=56, saver_available=False,
                                         game_statuses_detailed=st_s, require_detailed_statuses=True)
            assert sel_s.action == "skip" and sel_s.decision.guard is None
            daily_decision.write_decision(
                s_day, pdir, action="skip", source="mdp", objective="reach57",
                primary=sel_s.primary_candidate, streak=56, saver_available=False,
                delivery_status="not_applicable", scoreable=False, policy_guard=None,
                now=datetime(2026, 3, 27, 22, 0, tzinfo=timezone.utc),
            )
            for extra_d, extra_p in (("2026-03-28", 0.60), ("2026-03-29", 0.70)):
                daily_decision.write_decision(
                    extra_d, pdir, action="skip", source="mdp", objective="reach57",
                    primary={"batter_id": 199, "batter_name": "Extra", "team": "T", "game_pk": 1, "p_game_hit": extra_p},
                    streak=56, saver_available=False, delivery_status="not_applicable", scoreable=False,
                    now=datetime.fromisoformat(f"{extra_d}T22:00:00+00:00"),
                )

        parsed_g = daily_decision.parse_decision(daily_decision.decision_path(g_day, pdir).read_text())
        parsed_u = daily_decision.parse_decision(daily_decision.decision_path(u_day, pdir).read_text())
        parsed_s = daily_decision.parse_decision(daily_decision.decision_path(s_day, pdir).read_text())
        assert parsed_g["policy_guard"] == GUARD_SINGLE and parsed_g["scoreable"] is True
        assert parsed_u["policy_guard"] is None and parsed_u["scoreable"] is True
        assert parsed_s["policy_guard"] is None and parsed_s["scoreable"] is False

        reader_code = """
import sys
from pathlib import Path
from bts import picks, daily_decision
pdir = Path(sys.argv[1])
p_g = picks.load_pick("2026-03-25", pdir)
d_g = daily_decision.load_decision("2026-03-25", pdir)
d_u = daily_decision.load_decision("2026-03-26", pdir)
d_s = daily_decision.load_decision("2026-03-27", pdir)
assert p_g.pick.batter_id == 100
assert d_g["scoreable"] is True and d_g["policy_guard"]["rule"] == "monotone_skip_v1"
assert d_u["scoreable"] is True and d_u["policy_guard"] is None
assert d_s["scoreable"] is False and d_s["policy_guard"] is None
p_g.date = "2026-03-31"
if isinstance(getattr(p_g, "policy_decision", None), dict):
    p_g.policy_decision = {k: v for k, v in p_g.policy_decision.items() if k != "guard"}
picks.save_pick(p_g, pdir)
daily_decision.write_decision("2026-03-31", pdir, action="single", source="mdp",
                              primary={"batter_id": 100, "p_game_hit": 0.83},
                              delivery_status="delivered", scoreable=True)
"""
        for label, rev_src in (("f83e0d0b", BASE_SRC), ("619abda7", rollback_src)):
            sub_dir = Path(tmp) / f"from_{label}"
            subprocess.run(["cp", "-R", str(pdir), str(sub_dir)], check=True, timeout=10)
            env = dict(os.environ, PYTHONPATH=str(rev_src))
            subprocess.run([sys.executable, "-c", reader_code, str(sub_dir)], cwd=REPO, env=env, check=True, timeout=15)
            lp = picks.load_pick("2026-03-31", sub_dir)
            ld = daily_decision.load_decision("2026-03-31", sub_dir)
            assert lp is not None and (lp.policy_decision or {}).get("guard") is None
            assert ld is not None and ld.get("policy_guard") is None
            assert daily_decision.is_scoreable_commit("2026-03-31", sub_dir, lp) is True

        ledger = Path(tmp) / "ledger.jsonl"
        ledger_rows = [
            ("2026-03-25", 50), ("2026-03-26", u_streak), ("2026-03-27", 56),
            ("2026-03-28", 56), ("2026-03-29", 56),
        ]
        ledger.write_text("\n".join(
            json.dumps({"recorded_at": f"{d}T00:00:00Z", "active_streak": st}) for d, st in ledger_rows
        ))
        census_out = run_census(
            picks_dir=pdir, ledger_path=ledger, policy_path=REPO / "data" / "models" / "mdp_policy.npz",
            saver_active=False, start="2026-03-25", end="2026-03-29", as_of="2026-03-29",
            output=Path(tmp) / "census.json",
        )
        by_date = {r["date"]: r for r in census_out["rows"]}
        assert by_date[g_day]["parity"] == "guarded_play_excluded"
        assert by_date[g_day]["raw_deployed"] == "skip"
        assert by_date[g_day]["diffs"]["primary"] == {"from": "skip", "to": "single"}
        assert census_out["follow_up"]["primary_n_diff"] == 0
        assert by_date[u_day]["parity"] in ("parity", "clamped_double_downgrade")
        assert "primary" not in by_date[u_day]["diffs"]
        assert by_date[s_day]["parity"] == "parity"

        bad_dir = Path(tmp) / "bad_census"
        subprocess.run(["cp", "-R", str(pdir), str(bad_dir)], check=True, timeout=10)
        bad_dec_path = daily_decision.decision_path(g_day, bad_dir)
        bad_dec = json.loads(bad_dec_path.read_text())
        bad_dec["policy_guard"] = None
        bad_dec_path.write_text(json.dumps(bad_dec))
        halted = False
        try:
            run_census(
                picks_dir=bad_dir, ledger_path=ledger, policy_path=REPO / "data" / "models" / "mdp_policy.npz",
                saver_active=False, start="2026-03-25", end="2026-03-29", as_of="2026-03-29",
                output=Path(tmp) / "bad_out.json",
            )
        except CensusHalt:
            halted = True
        assert halted

        assert daily_decision.is_reach57_mdp_skip(daily_decision.load_decision(g_day, pdir)) is False
        assert daily_decision.is_reach57_mdp_skip(daily_decision.load_decision(u_day, pdir)) is False
        assert daily_decision.is_reach57_mdp_skip(daily_decision.load_decision(s_day, pdir)) is True
        assert skip_policy_shadow.record_skip_from_decision(g_day, pdir) is None
        assert skip_policy_shadow.record_skip_from_decision(u_day, pdir) is None
        assert skip_policy_shadow.record_skip_from_decision(s_day, pdir) is not None

        cfg_research = {"scheduler": {"pick_delivery": "private", "entry_intent": "research"}}
        toml_cfg = Path(tmp) / "day_out.toml"
        toml_cfg.write_text(
            f'[orchestrator]\npicks_dir = "{pdir}"\n'
            f'[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
        )
        runner = CliRunner()
        for check_d, exp_state, exp_rc in (
            (g_day, "delivered_under_research", 1),
            (u_day, "delivered_under_research", 1),
            (s_day, "deliberate_skip", 0),
        ):
            out_d = day_outcome.gather(
                check_d, pdir, cfg_research, expected_username="eric",
                fetch_schedule=lambda d: [{"status": {"detailedState": "Final"}}],
            )
            assert out_d.state == exp_state, (check_d, out_d)
            with patch.object(scheduler, "fetch_schedule", lambda d: [{"status": {"detailedState": "Final"}}]):
                res_do = runner.invoke(
                    cli.cli, ["day-outcome", "--date", check_d, "--config", str(toml_cfg), "--picks-dir", str(pdir)]
                )
                assert res_do.exit_code == exp_rc and exp_state in res_do.output, (res_do.exit_code, res_do.output)

        with walk.offline(walk.REAL_DATETIME(2026, 3, 28, 2, 0, tzinfo=walk.ET)), \
                patch("bts.cli._now_et", lambda: walk.REAL_DATETIME(2026, 3, 26, 2, 0, tzinfo=walk.ET)), \
                patch("bts.picks.resolve_daily_slot_results", lambda daily, d, **k: {"pick": "hit"}):
            res_cr_g = runner.invoke(cli.cli, ["check-results", "--date", g_day, "--picks-dir", str(pdir), "--allow-stale-scoring"])
            res_cr_u = runner.invoke(cli.cli, ["check-results", "--date", u_day, "--picks-dir", str(pdir), "--allow-stale-scoring"])
            res_cr_s = runner.invoke(cli.cli, ["check-results", "--date", s_day, "--picks-dir", str(pdir), "--allow-stale-scoring"])
        assert res_cr_g.exit_code == 0 and "HIT!" in res_cr_g.output, res_cr_g.output
        assert res_cr_u.exit_code == 0 and "HIT!" in res_cr_u.output, res_cr_u.output
        assert res_cr_s.exit_code == 0 and "No pick found" in res_cr_s.output, res_cr_s.output
        verified["consumers_and_cross_rev"] = "PASS"

    return verified


EARLY_RETURN_WORKER = """
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import bts
from bts import cli, contest_fetch, daily_decision, day_outcome, dm, orchestrator, picks, scheduler
from bts.leaderboard import auth
from click.testing import CliRunner
from scripts.audit.season_walk import walk

out_path = Path(sys.argv[1])
day_iso = "2027-03-24"
day_obj = date(2027, 3, 24)
now_et = walk.REAL_DATETIME(2027, 3, 24, 18, 25, tzinfo=walk.ET)

statuses = {
    70000: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    70001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    70002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
}

def make_sched(pks):
    times = {70000: "22:25:00Z", 70001: "23:05:00Z", 70002: "23:35:00Z"}
    return [{
        "gamePk": pk, "gameType": "R", "officialDate": day_iso,
        "gameDate": f"{day_iso}T{times[pk]}",
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled", "statusCode": "S"},
        "teams": {"away": {"team": {"name": f"A{pk}"}}, "home": {"team": {"name": f"H{pk}"}}},
    } for pk in pks]

results = {"bts_file": str(Path(bts.__file__).resolve())}

# S1: locked_unconfirmed written by restart while slow send_dm completes (unguarded day)
with tempfile.TemporaryDirectory() as tmp1:
    pdir = Path(tmp1) / "picks"
    pdir.mkdir()
    cfg = walk._config(pdir)
    cfg["bluesky"] = {"dm_recipient": "eric.test"}
    cfg["scheduler"]["pick_delivery"] = "dm"
    cfg["scheduler"]["live_forward_capture_on_lock"] = False
    frame = walk._predictions(day_obj, 0.83)

    with walk.offline(now_et), \
            patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
            patch.object(picks, "get_game_statuses_detailed", lambda d: {int(pk): statuses[70001] for pk in frame["game_pk"]}), \
            patch.object(scheduler, "_now_et", lambda: now_et), \
            patch.object(scheduler, "_void_committed_games", lambda *a: {}), \
            patch.object(scheduler, "_alert_delivery_outcome_unknown", lambda *a, **k: None):
        assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.83)) == []
        _, sel, _ = orchestrator.run_and_pick(cfg, day_iso)
        daily = sel.pick_result.daily
        picks.save_pick(daily, pdir)
        st1 = scheduler.SchedulerState(
            date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
            runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
        )

        def slow_send_dm(rec, txt):
            persisted_mid = picks.load_pick(day_iso, pdir)
            st2 = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            scheduler._write_classification_decision(
                pdir, day_iso, action="single", delivered=False, attempted=True,
                primary=scheduler._row_from_daily(persisted_mid.pick), double_down=None,
                state=st2, daily=persisted_mid,
            )
            return "dm-confirmed-123"

        with patch.object(dm, "send_dm", slow_send_dm):
            ok = scheduler._deliver_and_lock_pick(daily, cfg, pdir, st1, day_iso, "lineup", selection=sel)
        assert ok is True
        final_pick = picks.load_pick(day_iso, pdir)
        final_dec = daily_decision.load_decision(day_iso, pdir)

        entry_dms = []
        with patch.object(auth, "load_session_cookies", lambda: {}), \
                patch.object(auth, "extract_uid", lambda c: 1), \
                patch.object(auth, "fetch_login_session", lambda **k: SimpleNamespace(user_id=1, username="eric", xsid="x")), \
                patch.object(contest_fetch, "fetch_profile", lambda *a: {"predictions": []}), \
                patch.object(contest_fetch, "fetch_pending_predictions", lambda *a: []), \
                patch.object(cli, "_fetch_rounds", lambda: {1: day_obj}), \
                patch.object(cli, "_fetch_bts_to_mlb", lambda: {}), \
                patch.object(dm, "send_dm", lambda r, m: entry_dms.append(m) or "nag-1"):
            CliRunner().invoke(cli.cli, [
                "check-pick-entered", "--picks-dir", str(pdir), "--expected-username", "eric",
                "--dm-recipient", "eric.test", "--now-et", "2027-03-24T18:30:00",
            ])

        results["S1_locked_unconfirmed_then_confirmed"] = {
            "pick_was_delivered": picks.pick_was_delivered(final_pick),
            "pick_notification_id": final_pick.notification_id,
            "decision_delivery_status": final_dec["delivery_status"],
            "decision_streak": final_dec["streak"],
            "entry_watchdog_dm": entry_dms[0] if entry_dms else None,
        }

# S2: Private lock followed by DM commit after delivery-mode change (via fallback refresh without deleting <date>.json!)
with tempfile.TemporaryDirectory() as tmp2:
    pdir = Path(tmp2) / "picks"
    pdir.mkdir()
    cfg_priv = walk._config(pdir)
    cfg_priv["scheduler"]["pick_delivery"] = "private"
    cfg_priv["scheduler"]["entry_intent"] = "research"
    cfg_priv["scheduler"]["live_forward_capture_on_lock"] = False

    early_frame = pd.DataFrame([{
        "batter_name": "Old Private Batter", "batter_id": 999, "team": "BOS", "lineup": 1,
        "pitcher_name": "P0", "pitcher_id": 800, "p_game_hit": 0.83, "flags": "",
        "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "NYY",
    }])
    later_double_frame = pd.DataFrame([
        {
            "batter_name": "New Primary", "batter_id": 101, "team": "NYY", "lineup": 1,
            "pitcher_name": "P1", "pitcher_id": 801, "p_game_hit": 0.90, "flags": "",
            "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "BOS",
        },
        {
            "batter_name": "New DoubleLeg", "batter_id": 202, "team": "LAD", "lineup": 1,
            "pitcher_name": "P2", "pitcher_id": 802, "p_game_hit": 0.88, "flags": "",
            "game_pk": 70002, "game_time": f"{day_iso}T23:35:00Z", "pitcher_team": "SF",
        },
    ])

    with walk.offline(now_et), \
            patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
            patch.object(scheduler, "_now_et", lambda: now_et), \
            patch.object(scheduler, "_void_committed_games", lambda *a: {}):
        assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.83)) == []
        with patch.object(orchestrator, "run_cascade", lambda tiers, d: (early_frame.copy(), "local")):
            _, sel_priv, _ = orchestrator.run_and_pick(cfg_priv, day_iso)
        priv_daily = sel_priv.pick_result.daily
        picks.save_pick(priv_daily, pdir)
        st_priv = scheduler.SchedulerState(
            date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
            runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        assert scheduler._deliver_and_lock_pick(priv_daily, cfg_priv, pdir, st_priv, day_iso, "lineup", selection=sel_priv) is True
        assert daily_decision.load_decision(day_iso, pdir)["delivery_status"] == "private_locked"

        assert walk.install_state(pdir, day_obj, walk.State("dated", 5, 5, "not_earned", 0.90)) == []
        cfg_dm = walk._config(pdir)
        cfg_dm["bluesky"] = {"dm_recipient": "eric.test"}
        cfg_dm["scheduler"]["pick_delivery"] = "dm"
        cfg_dm["scheduler"]["entry_intent"] = "enter"
        cfg_dm["scheduler"]["live_forward_capture_on_lock"] = False
        sent_dms_2 = []
        loaded_for_fb = picks.load_pick(day_iso, pdir)
        assert not picks.pick_was_delivered(loaded_for_fb)
        with patch.object(orchestrator, "run_cascade", lambda tiers, d: (later_double_frame.copy(), "local")), \
                patch.object(dm, "send_dm", lambda r, t: sent_dms_2.append(t) or "dm-real-2"):
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg_dm, day_iso, loaded_for_fb, 0.03)
            st_dm = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            assert scheduler._deliver_and_lock_pick(refresh.daily, cfg_dm, pdir, st_dm, day_iso, "fallback", selection=refresh.selection) is True

        final_pick_2 = picks.load_pick(day_iso, pdir)
        final_dec_2 = daily_decision.load_decision(day_iso, pdir)
        out_day_2 = day_outcome.gather(
            day_iso, pdir, cfg_dm, expected_username="eric",
            fetch_schedule=lambda d: [{"status": {"detailedState": "Final"}}],
        )
        results["S2_private_then_dm_fallback"] = {
            "dm_sent_count": len(sent_dms_2),
            "pick_file_primary": final_pick_2.pick.batter_name,
            "pick_file_double": final_pick_2.double_down.batter_name if final_pick_2.double_down else None,
            "decision_action": final_dec_2["action"],
            "decision_primary": final_dec_2["primary"]["batter_name"],
            "decision_double": final_dec_2["double_down"]["batter_name"] if final_dec_2["double_down"] else None,
            "decision_delivery_status": final_dec_2["delivery_status"],
            "day_outcome_pick": out_day_2.pick,
        }

# S3: Refused void-game pick (archiving & unlinking <date>.json after an earlier private_locked record) followed by replacement DM commit on next check
with tempfile.TemporaryDirectory() as tmp3:
    pdir = Path(tmp3) / "picks"
    pdir.mkdir()
    cfg_priv = walk._config(pdir)
    cfg_priv["scheduler"]["pick_delivery"] = "private"
    cfg_priv["scheduler"]["live_forward_capture_on_lock"] = False
    void_daily = picks.DailyPick(
        date=day_iso, run_time="2027-03-24T21:00:00+00:00",
        pick=picks.Pick(
            batter_name="Postponed Game Batter", batter_id=550, team="CHC", lineup_position=1,
            pitcher_name="P0", pitcher_id=800, p_game_hit=0.83, flags=[], projected_lineup=False,
            game_pk=70000, game_time=f"{day_iso}T23:05:00Z",
        ),
        double_down=None, runner_up=None,
    )
    later_frame = pd.DataFrame([
        {
            "batter_name": "AfterRefusal Primary", "batter_id": 101, "team": "NYY", "lineup": 1,
            "pitcher_name": "P1", "pitcher_id": 801, "p_game_hit": 0.90, "flags": "",
            "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "BOS",
        },
        {
            "batter_name": "AfterRefusal Double", "batter_id": 202, "team": "LAD", "lineup": 1,
            "pitcher_name": "P2", "pitcher_id": 802, "p_game_hit": 0.88, "flags": "",
            "game_pk": 70002, "game_time": f"{day_iso}T23:35:00Z", "pitcher_team": "SF",
        },
    ])
    picks.save_pick(void_daily, pdir)
    st0 = scheduler.SchedulerState(
        date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=True, pick_locked_at=now_et.isoformat(), result_status=None, next_wakeup=None,
    )
    scheduler._commit_decision_for_pick(pdir, day_iso, void_daily, selection=None, delivery_status="private_locked", state=st0)

    cfg_dm = walk._config(pdir)
    cfg_dm["bluesky"] = {"dm_recipient": "eric.test"}
    cfg_dm["scheduler"]["pick_delivery"] = "dm"
    cfg_dm["scheduler"]["entry_intent"] = "enter"
    cfg_dm["scheduler"]["live_forward_capture_on_lock"] = False
    cfg_dm["health_checks"]["enabled"] = False
    sent_dms_3 = []
    with walk.offline(now_et), \
            patch.object(scheduler, "_now_et", lambda: now_et), \
            patch("bts.health.alert.dispatch_dm_for_health_alerts", lambda *a, **k: None):
        st_ref = scheduler.SchedulerState(
            date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
            runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        with patch.object(scheduler, "_void_committed_games", lambda *a: {70000: "Postponed"}):
            assert scheduler._deliver_and_lock_pick(void_daily, cfg_dm, pdir, st_ref, day_iso, "fallback") is False
        assert not (pdir / f"{day_iso}.json").exists()
        assert walk.install_state(pdir, day_obj, walk.State("dated", 5, 5, "not_earned", 0.90)) == []
        with patch.object(orchestrator, "run_cascade", lambda tiers, d: (later_frame.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                patch.object(scheduler, "fetch_schedule", lambda d: make_sched((70001, 70002)) if d == day_iso else []), \
                patch.object(scheduler, "check_confirmed_lineups", lambda pks: {pk: {"away", "home"} for pk in pks}), \
                patch.object(scheduler, "_void_committed_games", lambda *a: {}), \
                patch.object(scheduler, "_watchdog_ping_sleep", lambda s, *a, **k: None), \
                patch.object(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None), \
                patch.object(scheduler, "run_result_polling", lambda *a, **k: "final"), \
                patch.object(dm, "send_dm", lambda r, t: sent_dms_3.append(t) or "dm-after-refusal"), \
                redirect_stderr(io.StringIO()):
            scheduler.run_day(day_iso, cfg_dm)

    final_pick_3 = picks.load_pick(day_iso, pdir)
    final_dec_3 = daily_decision.load_decision(day_iso, pdir)
    results["S3_refused_void_then_replaced"] = {
        "refused_archives": sorted(p.name for p in (pdir / day_iso).glob("refused_delivery_*.json")),
        "dm_sent_count": len(sent_dms_3),
        "pick_file_primary": final_pick_3.pick.batter_name,
        "decision_action": final_dec_3["action"],
        "decision_primary": final_dec_3["primary"]["batter_name"],
        "decision_delivery_status": final_dec_3["delivery_status"],
    }

# S4: Earlier process writes private_locked right after Process 2 passes run_single_check:1869 (call 1), then Process 2 archives & replaces
with tempfile.TemporaryDirectory() as tmp4:
    pdir = Path(tmp4) / "picks"
    pdir.mkdir()
    cfg_dm = walk._config(pdir)
    cfg_dm["bluesky"] = {"dm_recipient": "eric.test"}
    cfg_dm["scheduler"]["pick_delivery"] = "dm"
    cfg_dm["scheduler"]["entry_intent"] = "enter"
    cfg_dm["scheduler"]["live_forward_capture_on_lock"] = False
    cfg_dm["health_checks"]["enabled"] = False

    race_now = walk.REAL_DATETIME(2027, 3, 24, 18, 22, tzinfo=walk.ET)
    early_daily = picks.DailyPick(
        date=day_iso, run_time="2027-03-24T21:00:00+00:00",
        pick=picks.Pick(
            batter_name="Early Cutoff Batter", batter_id=555, team="CHC", lineup_position=1,
            pitcher_name="P0", pitcher_id=800, p_game_hit=0.83, flags=[], projected_lineup=False,
            game_pk=70000, game_time=f"{day_iso}T22:25:00Z",
        ),
        double_down=None, runner_up=None,
    )
    later_frame = pd.DataFrame([
        {
            "batter_name": "Replacement Primary", "batter_id": 101, "team": "NYY", "lineup": 1,
            "pitcher_name": "P1", "pitcher_id": 801, "p_game_hit": 0.90, "flags": "",
            "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "BOS",
        },
        {
            "batter_name": "Replacement Double", "batter_id": 202, "team": "LAD", "lineup": 1,
            "pitcher_name": "P2", "pitcher_id": 802, "p_game_hit": 0.88, "flags": "",
            "game_pk": 70002, "game_time": f"{day_iso}T23:35:00Z", "pitcher_team": "SF",
        },
    ])

    picks.save_pick(early_daily, pdir)
    real_load_decision = daily_decision.load_decision
    load_calls = {"n": 0}

    def interleaved_load_decision(d_arg, p_arg):
        load_calls["n"] += 1
        res = real_load_decision(d_arg, p_arg)
        if load_calls["n"] == 1:
            st_p1 = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=race_now.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=True, pick_locked_at=race_now.isoformat(),
                result_status=None, next_wakeup=None,
            )
            scheduler._commit_decision_for_pick(
                pdir, day_iso, early_daily, selection=None, delivery_status="private_locked", state=st_p1,
            )
        return res

    sent_dms_4 = []
    with walk.offline(race_now), \
            patch.object(daily_decision, "load_decision", interleaved_load_decision), \
            patch.object(orchestrator, "run_cascade", lambda tiers, d: (later_frame.copy(), "local")), \
            patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
            patch.object(scheduler, "fetch_schedule", lambda d: make_sched((70000, 70001, 70002)) if d == day_iso else []), \
            patch.object(scheduler, "check_confirmed_lineups", lambda pks: {pk: {"away", "home"} for pk in pks}), \
            patch.object(scheduler, "_now_et", lambda: race_now), \
            patch.object(scheduler, "_watchdog_ping_sleep", lambda s, *a, **k: None), \
            patch.object(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None), \
            patch.object(scheduler, "run_result_polling", lambda *a, **k: "final"), \
            patch.object(dm, "send_dm", lambda r, t: sent_dms_4.append(t) or "dm-rep-4"):
        assert walk.install_state(pdir, day_obj, walk.State("dated", 5, 5, "not_earned", 0.90)) == []
        with redirect_stderr(io.StringIO()):
            scheduler.run_day(day_iso, cfg_dm)

    final_pick_4 = picks.load_pick(day_iso, pdir)
    final_dec_4 = real_load_decision(day_iso, pdir)
    archived = sorted(p.name for p in (pdir / day_iso).glob("stale_pick_*.json"))
    results["S4_interleaved_earlier_process_replaced"] = {
        "archived_files": archived,
        "dm_sent_count": len(sent_dms_4),
        "pick_file_primary": final_pick_4.pick.batter_name,
        "pick_file_double": final_pick_4.double_down.batter_name if final_pick_4.double_down else None,
        "decision_action": final_dec_4["action"],
        "decision_primary": final_dec_4["primary"]["batter_name"],
        "decision_delivery_status": final_dec_4["delivery_status"],
    }

# S5: Stale decision.json from a previous run of the same date (with <date>.json absent) followed by a fresh run_day DM commit
with tempfile.TemporaryDirectory() as tmp5:
    pdir = Path(tmp5) / "picks"
    pdir.mkdir()
    daily_decision.write_decision(
        day_iso, pdir, action="single", source="mdp", objective="reach57",
        primary={"batter_id": 999, "batter_name": "Stale Prior Run Batter", "team": "BOS", "game_pk": 70001, "p_game_hit": 0.83},
        streak=50, saver_available=False, delivery_status="private_locked", scoreable=True,
    )
    later_double_frame = pd.DataFrame([
        {
            "batter_name": "Fresh Run Primary", "batter_id": 101, "team": "NYY", "lineup": 1,
            "pitcher_name": "P1", "pitcher_id": 801, "p_game_hit": 0.90, "flags": "",
            "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "BOS",
        },
        {
            "batter_name": "Fresh Run Double", "batter_id": 202, "team": "LAD", "lineup": 1,
            "pitcher_name": "P2", "pitcher_id": 802, "p_game_hit": 0.88, "flags": "",
            "game_pk": 70002, "game_time": f"{day_iso}T23:35:00Z", "pitcher_team": "SF",
        },
    ])
    cfg_dm = walk._config(pdir)
    cfg_dm["bluesky"] = {"dm_recipient": "eric.test"}
    cfg_dm["scheduler"]["pick_delivery"] = "dm"
    cfg_dm["scheduler"]["entry_intent"] = "enter"
    cfg_dm["scheduler"]["live_forward_capture_on_lock"] = False
    cfg_dm["health_checks"]["enabled"] = False
    sent_dms_5 = []
    with walk.offline(now_et), \
            patch.object(orchestrator, "run_cascade", lambda tiers, d: (later_double_frame.copy(), "local")), \
            patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
            patch.object(scheduler, "fetch_schedule", lambda d: make_sched((70001, 70002)) if d == day_iso else []), \
            patch.object(scheduler, "check_confirmed_lineups", lambda pks: {pk: {"away", "home"} for pk in pks}), \
            patch.object(scheduler, "_now_et", lambda: now_et), \
            patch.object(scheduler, "_watchdog_ping_sleep", lambda s, *a, **k: None), \
            patch.object(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None), \
            patch.object(scheduler, "run_result_polling", lambda *a, **k: "final"), \
            patch.object(dm, "send_dm", lambda r, t: sent_dms_5.append(t) or "dm-fresh-5"):
        assert walk.install_state(pdir, day_obj, walk.State("dated", 5, 5, "not_earned", 0.90)) == []
        with redirect_stderr(io.StringIO()):
            scheduler.run_day(day_iso, cfg_dm)

    final_pick_5 = picks.load_pick(day_iso, pdir)
    final_dec_5 = daily_decision.load_decision(day_iso, pdir)
    results["S5_stale_prior_run_record"] = {
        "dm_sent_count": len(sent_dms_5),
        "pick_file_primary": final_pick_5.pick.batter_name,
        "pick_file_double": final_pick_5.double_down.batter_name if final_pick_5.double_down else None,
        "decision_action": final_dec_5["action"],
        "decision_primary": final_dec_5["primary"]["batter_name"],
        "decision_delivery_status": final_dec_5["delivery_status"],
    }

out_path.write_text(json.dumps(results, indent=2))
"""


def compare_early_return_base_vs_head() -> dict:
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p04r_er_") as tmp:
        tmp_dir = Path(tmp)
        worker = tmp_dir / "er_worker.py"
        worker.write_text(EARLY_RETURN_WORKER)
        base_json = tmp_dir / "base.json"
        head_json = tmp_dir / "head.json"

        for src_dir, out_f in ((BASE_SRC, base_json), (HEAD_SRC, head_json)):
            env = dict(os.environ, PYTHONPATH=f"{src_dir}:{REPO}")
            subprocess.run([sys.executable, str(worker), str(out_f)], cwd=REPO, env=env, check=True, timeout=60)

        b = json.loads(base_json.read_text())
        h = json.loads(head_json.read_text())

        assert b["S1_locked_unconfirmed_then_confirmed"]["decision_delivery_status"] == "delivered"
        assert h["S1_locked_unconfirmed_then_confirmed"]["decision_delivery_status"] == "locked_unconfirmed"
        assert b["S2_private_then_dm_fallback"]["decision_primary"] == "New Primary"
        assert h["S2_private_then_dm_fallback"]["decision_primary"] == "Old Private Batter"
        assert b["S3_refused_void_then_replaced"]["decision_primary"] == "AfterRefusal Primary"
        assert h["S3_refused_void_then_replaced"]["decision_primary"] == "Postponed Game Batter"
        assert b["S4_interleaved_earlier_process_replaced"]["decision_primary"] == "Replacement Primary"
        assert h["S4_interleaved_earlier_process_replaced"]["decision_primary"] == "Early Cutoff Batter"
        assert b["S5_stale_prior_run_record"]["decision_primary"] == "Fresh Run Primary"
        assert h["S5_stale_prior_run_record"]["decision_primary"] == "Stale Prior Run Batter"

        return {"base_f83e0d0b": b, "head_d6141728": h}


def main() -> None:
    rec_results = test_record_paths_and_consumers()
    er_comparison = compare_early_return_base_vs_head()
    out = {
        "status": "PASS",
        "record_paths_and_consumers": rec_results,
        "early_return_base_vs_head": er_comparison,
    }
    (REVIEW_DIR / "probe_04_regression.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
