"""Adversarial probe for Items 6, 7, 8, and 9 (every live caller, base vs branch).

Installs sys.addaudithook denying all DNS/socket connect events and recording
filesystem operations; captures all fetch, DM, posting, and schedule transports.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

AUDIT = {
    "active": False,
    "denied_net": [],
    "events": [],
    "write_opens": [],
    "mutations": [],
}

_NET_EVENTS = frozenset({
    "socket.getaddrinfo",
    "socket.getnameinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.connect",
    "socket.connect_ex",
    "socket.sendto",
})


def _hook(event: str, args: tuple) -> None:
    if event in _NET_EVENTS:
        AUDIT["denied_net"].append((event, repr(args)))
        raise RuntimeError(f"network forbidden by audit hook: {event}")
    if not AUDIT["active"]:
        return
    if event == "open" and args:
        path_s = str(args[0])
        mode = str(args[1]) if len(args) > 1 else "r"
        AUDIT["events"].append(("open", path_s, mode))
        if any(ch in mode for ch in "wax+"):
            AUDIT["write_opens"].append((path_s, mode))
    elif event == "os.stat" and args:
        AUDIT["events"].append(("stat", str(args[0])))
    elif event in ("os.mkdir", "os.unlink", "os.rmdir"):
        AUDIT["events"].append((event, str(args[0])))
        AUDIT["mutations"].append((event, str(args[0])))
    elif event in ("os.rename", "os.replace"):
        AUDIT["events"].append((event, str(args[0]), str(args[1])))
        AUDIT["mutations"].append((event, str(args[0]), str(args[1])))


sys.addaudithook(_hook)

import argparse
from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import httpx
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32
DEEP = b"[" * 200_000 + b"]" * 200_000


def _reset_audit() -> None:
    AUDIT["events"].clear()
    AUDIT["write_opens"].clear()
    AUDIT["mutations"].clear()


@contextmanager
def audited():
    _reset_audit()
    AUDIT["active"] = True
    try:
        yield AUDIT
    finally:
        AUDIT["active"] = False


def _tree(root: Path) -> dict[str, bytes | str]:
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = f"SYMLINK->{os.readlink(p)}"
        elif p.is_file():
            out[rel] = p.read_bytes()
        elif p.is_dir():
            out[rel] = "DIR"
    return out


def _put_json(picks: Path, name: str, obj) -> Path:
    p = picks / "account_state" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, bytes):
        p.write_bytes(obj)
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _sample_predictions(date_str: str = "2027-07-15") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "batter_id": 680757,
            "batter_name": "Steven Kwan",
            "team": "CLE",
            "lineup": 1,
            "pitcher_id": 543037,
            "pitcher_name": "Gerrit Cole",
            "opp": "NYY",
            "p_hit_pa": 0.24,
            "p_game_hit": 0.82,
            "flags": "",
            "projected_lineup": False,
            "game_pk": 101,
            "game_time": f"{date_str}T23:10:00Z",
        },
        {
            "batter_id": 665742,
            "batter_name": "Juan Soto",
            "team": "NYM",
            "lineup": 2,
            "pitcher_id": 605483,
            "pitcher_name": "Blake Snell",
            "opp": "LAD",
            "p_hit_pa": 0.23,
            "p_game_hit": 0.81,
            "flags": "",
            "projected_lineup": False,
            "game_pk": 102,
            "game_time": f"{date_str}T23:40:00Z",
        },
    ])


def run_items_6_7_8(repo_root: Path) -> dict:
    sys.path.insert(0, str(repo_root / "src"))
    import bts.cli as cli
    import bts.contest_fetch as cf
    import bts.contest_state as cs
    import bts.dm as dm_mod
    import bts.health.contest_state as hc
    import bts.leaderboard.auth as auth
    import bts.posting as posting_mod
    import bts.saver_state as ss
    import bts.season_contract as sc

    def sw(**kw):
        return {
            "schema_version": sc.SCHEMA,
            "season": 2027,
            "status": "automatic",
            "activation_id": ID,
            "switched_at": "2027-07-15T16:00:00Z",
            "revoked_at": None,
            "note": None,
        } | kw

    def auto(**kw):
        return {
            "schema_version": sc.AUTO_SCHEMA,
            "season": 2027,
            "activation_id": ID,
            "active_streak": 12,
            "best_streak": 15,
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T15:00:00Z",
            "user_id": 50311,
            "username": "stonehengee",
        } | kw

    def dated(day: str = "2027-07-15", **kw):
        d = dt.date.fromisoformat(day)
        return {
            "schema_version": cs.DATED_MANUAL_SCHEMA,
            "effective_date": day,
            "season": d.year,
            "expires_at": REAL_DT.combine(d, dt.time.max, ET).isoformat(),
            "active_streak": 12,
            "best_streak": 15,
            "saver_state": "used",
            "recorded_at": "2027-07-15T15:00:00Z",
            "source": "manual_cli",
        } | kw

    @contextmanager
    def env(now_box):
        dms, posts = [], []

        class FakeDT(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                cur = now_box[0]
                return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

        with (
            patch.object(dt, "datetime", FakeDT),
            patch.object(httpx, "get", side_effect=AssertionError("uncaptured httpx")),
            patch.object(auth, "load_session_cookies", return_value={}),
            patch.object(auth, "extract_uid", return_value="u1"),
            patch.object(auth, "fetch_login_session", side_effect=lambda **_: SimpleNamespace(user_id=50311, username="stonehengee", xsid="x")),
            patch.object(cf, "fetch_profile", side_effect=lambda *_: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]}),
            patch.object(cli, "_fetch_rounds", side_effect=lambda: {1: now_box[0].astimezone(ET).date() - dt.timedelta(days=1)}),
            patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-1"),
            patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-1"),
        ):
            yield {"dms": dms, "posts": posts}

    runner = CliRunner()
    now_box = [NOW]

    # --- ITEM 6 ---
    checked_status = []
    with env(now_box) as tr:
        with tempfile.TemporaryDirectory() as tmp:
            cold = Path(tmp) / "nonexistent_picks"
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(cold)])
            assert res.exit_code == 0
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            assert "live decision now: refused; Live decision refused: season 2027 switch is absent." in res.output
            checked_status.append("cold_absent")

        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "season_contract.json", sw())
            _put_json(picks, "contest_streak.json", auto())
            _put_json(picks, "saver_state.json", {"season": 2027, "state": "active"})
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks), "--season", "2026"])
            assert res.exit_code == 0 and "switch: other_season" in res.output
            real = cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
            assert (
                f"live decision now: accepted; source={real.source}; streak={real.streak}; "
                f"status={real.status}; saver={real.saver_flag_state}"
            ) in res.output
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            checked_status.append("accepted_covered_with_season_2026")

        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "season_contract.json", sw(season=2026))
            _put_json(picks, "contest_streak.json", auto(season=2026))
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks), "--season", "2026"])
            assert res.exit_code == 0 and "switch: automatic" in res.output
            assert "live decision now: refused; Live decision refused: season 2027 switch is other_season (file season 2026)." in res.output
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            checked_status.append("other_season_file_with_season_2026")

        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            _put_json(picks, "season_contract.json", [])
            _put_json(picks, "contest_streak.manual.json", dated())
            before = _tree(Path(tmp))
            with audited() as aud:
                res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
            assert res.exit_code == 1 and "switch: corrupt" in res.output
            assert "live decision now: accepted; source=contest; streak=12; status=fresh; saver=used" in res.output
            assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
            checked_status.append("accepted_dated_beside_corrupt_switch_exits_1")

        for kind, payload in [
            ("overlong_int", b'{"n":' + (b"9" * 5000) + b"}"),
            ("deep_recursion_200k", DEEP),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                picks.mkdir()
                _put_json(picks, "season_contract.json", sw())
                _put_json(picks, "contest_streak.json", auto())
                _put_json(picks, "saver_state.json", payload)
                before = _tree(Path(tmp))
                with audited() as aud:
                    res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
                assert res.exit_code == 0, (kind, res.output)
                assert "live decision now: unavailable;" in res.output
                assert res.output.index("saver:") < res.output.index("live decision now:") < res.output.index("ledger:")
                assert aud["write_opens"] == [] and aud["mutations"] == [] and _tree(Path(tmp)) == before
                checked_status.append(f"unavailable_{kind}")

        assert tr["dms"] == [] and tr["posts"] == []

    # --- ITEM 7 ---
    from probe_cli_sequences_v2 import probe_item7_revoke, probe_item8_sequences
    item7 = probe_item7_revoke()
    item8 = probe_item8_sequences()
    return {"item6": checked_status, "item7": item7, "item8": item8}


def run_caller_worker(src_dir: Path, out_path: Path) -> None:
    sys.path.insert(0, str(src_dir.resolve()))
    import bts.cli as cli
    import bts.contest_state as cs
    import bts.daily_decision as dd
    import bts.day_outcome as day_out
    import bts.dm as dm_mod
    import bts.health.alert as halert
    import bts.orchestrator as orch
    import bts.picks as picks_mod
    import bts.posting as posting_mod
    import bts.scheduler as sch

    runner = CliRunner()
    date_s = "2027-07-15"
    df = _sample_predictions(date_s)
    statuses_ok = {
        101: {"abstractGameState": "Preview", "detailedState": "Scheduled"},
        102: {"abstractGameState": "Preview", "detailedState": "Scheduled"},
    }
    out = {}

    def seed_uncovered(picks: Path) -> None:
        # Uncovered auto_v1 (streak=0, best=0 -> policy picks double!) and absent switch
        _put_json(picks, "contest_streak.json", {
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": 0,
            "best_streak": 0,
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T10:00:00Z",
        })
        _put_json(picks, "saver_state.json", {"season": 2027, "state": "not_earned"})

    def seed_covered(picks: Path, streak: int = 0) -> None:
        _put_json(picks, "season_contract.json", {
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": ID,
            "switched_at": "2027-07-15T10:00:00Z",
            "revoked_at": None,
            "note": None,
        })
        _put_json(picks, "contest_streak.json", {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ID,
            "active_streak": streak,
            "best_streak": max(streak, 9),
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T10:00:00Z",
        })
        _put_json(picks, "saver_state.json", {"season": 2027, "state": "not_earned"})

    class FakeDT(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    with (
        patch.object(dt, "datetime", FakeDT),
        patch.object(sch, "_now_et", return_value=NOW),
        patch.object(sch, "_void_committed_games", return_value={}),
        patch.object(sch, "_trigger_live_forward_capture_on_lock", return_value=None),
        patch.object(picks_mod, "get_game_statuses_detailed", return_value=statuses_ok),
        patch.object(picks_mod, "attach_provenance", return_value=None),
    ):
        # 1. orchestrator.run_and_pick (with predictions, and with empty predictions)
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg = {
                "orchestrator": {"picks_dir": str(picks)},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            with patch.object(orch, "run_cascade", return_value=(pd.DataFrame(), "local")):
                _, sel_empty, _ = orch.run_and_pick(cfg, date_s)
                out["run_and_pick_empty_predictions"] = {"sel_is_none": sel_empty is None}
            with patch.object(orch, "run_cascade", return_value=(df, "local")):
                try:
                    _, sel, _ = orch.run_and_pick(cfg, date_s)
                    out["run_and_pick_uncovered"] = {"accepted": True, "action": sel.action}
                except cs.ContestStateError as exc:
                    out["run_and_pick_uncovered"] = {"accepted": False, "err": str(exc).splitlines()[0]}

        # 2. orchestrator.orchestrate initial uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg_path = Path(tmp) / "orch.toml"
            cfg_obj = {
                "orchestrator": {"picks_dir": str(picks)},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True},
                "tiers": [],
            }
            dms, posts = [], []
            with (
                patch.object(orch, "load_config", return_value=cfg_obj),
                patch.object(orch, "run_cascade", return_value=(df, "local")),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-1"),
                patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-1"),
                patch.object(posting_mod, "should_post_now", return_value=True),
            ):
                ok = orch.orchestrate(cfg_path, date_s)
            out["orchestrate_initial_uncovered"] = {
                "returned": ok,
                "pick_saved": (picks / f"{date_s}.json").exists(),
                "dms_count": len(dms),
                "posts_count": len(posts),
            }

        # 3. orchestrator.orchestrate later reads (post-save read after revoke between run_and_pick and post)
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_covered(picks, streak=0)
            cfg_obj = {
                "orchestrator": {"picks_dir": str(picks)},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True},
                "tiers": [],
            }
            dms, posts = [], []
            real_rap = orch.run_and_pick

            def rap_then_revoke(*a, **kw):
                res = real_rap(*a, **kw)
                # Revoke switch between selection and post!
                _put_json(picks, "season_contract.json", {
                    "schema_version": "bts_season_contract_v1",
                    "season": 2027,
                    "status": "revoked",
                    "activation_id": ID,
                    "switched_at": "2027-07-15T10:00:00Z",
                    "revoked_at": "2027-07-15T12:00:00Z",
                    "note": "mid-orchestrate",
                })
                return res

            with (
                patch.object(orch, "load_config", return_value=cfg_obj),
                patch.object(orch, "run_cascade", return_value=(df, "local")),
                patch.object(orch, "run_and_pick", side_effect=rap_then_revoke),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-1"),
                patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-1"),
                patch.object(posting_mod, "should_post_now", return_value=True),
            ):
                try:
                    ok = orch.orchestrate(Path(tmp) / "orch.toml", date_s)
                    out["orchestrate_mid_revoke_post_save"] = {"raised": False, "returned": ok, "posts": len(posts)}
                except cs.ContestStateError as exc:
                    out["orchestrate_mid_revoke_post_save"] = {
                        "raised": True,
                        "err": str(exc).splitlines()[0],
                        "pick_saved_undelivered": (picks / f"{date_s}.json").exists(),
                        "posts": len(posts),
                    }

        # 4. scheduler.run_single_check on uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg = {
                "orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(Path(tmp) / ".hb")},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
            }
            dms = []
            with (
                patch.object(sch, "count_new_confirmations", return_value=2),
                patch.object(orch, "run_cascade", return_value=(df, "local")),
                patch.object(halert, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-h"),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-d"),
            ):
                chk = sch.run_single_check(
                    date=date_s,
                    all_game_pks=[101, 102],
                    confirmed_sides={(101, "away"), (101, "home")},
                    config=cfg,
                    early_lock_gap=0.03,
                )
            out["scheduler_run_single_check_uncovered"] = {
                "should_post": chk["should_post"],
                "pick_result_none": chk["pick_result"] is None,
                "selection_none": chk["selection"] is None,
                "pick_file_exists": (picks / f"{date_s}.json").exists(),
                "decision_exists": (picks / date_s / "decision.json").exists(),
                "alert_dms": len(dms),
            }

        # 5. scheduler._deliver_and_lock_pick across dm, public, private, delivered, crash on uncovered state
        for mode in ("dm", "public", "private", "delivered", "crash"):
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                picks.mkdir()
                seed_uncovered(picks)
                cfg = {
                    "orchestrator": {"picks_dir": str(picks)},
                    "bluesky": {"dm_recipient": "eric.test"},
                    "scheduler": {
                        "contest_state_required": True,
                        "pick_delivery": "private" if mode == "private" else ("public" if mode == "public" else "dm"),
                    },
                }
                dp = picks_mod.DailyPick(
                    date=date_s,
                    run_time=NOW.isoformat(),
                    pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                    double_down=None,
                    runner_up=None,
                )
                if mode == "delivered":
                    dp.notification_sent = True
                    dp.notification_channel = "bluesky_dm"
                    dp.notification_id = "already"
                    dp.delivered_at = NOW.isoformat()
                elif mode == "crash":
                    dp.delivery_attempted = True
                picks_mod.save_pick(dp, picks)
                st = sch.SchedulerState(
                    date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                    runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                pick_dms, alert_dms, posts = [], [], []
                with (
                    patch.object(dm_mod, "send_dm", side_effect=lambda r, t: pick_dms.append((r, t)) or "dm-p"),
                    patch.object(halert, "send_dm", side_effect=lambda r, t: alert_dms.append((r, t)) or "dm-a"),
                    patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-p"),
                ):
                    ok = sch._deliver_and_lock_pick(dp, cfg, picks, st, date_s, mode)
                out[f"deliver_and_lock_{mode}"] = {
                    "returned": ok,
                    "pick_locked": st.pick_locked,
                    "delivery_attempted": dp.delivery_attempted,
                    "pick_dms": len(pick_dms),
                    "alert_dms": len(alert_dms),
                    "posts": len(posts),
                    "decision_exists": (picks / date_s / "decision.json").exists(),
                }

        # 6. scheduler._refresh_pick_at_fallback_decision on uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg = {
                "orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(Path(tmp) / ".hb")},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            cached = picks_mod.DailyPick(
                date=date_s,
                run_time=NOW.isoformat(),
                pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                double_down=None,
                runner_up=None,
            )
            with patch.object(orch, "run_cascade", return_value=(df, "local")):
                try:
                    ref = sch._refresh_pick_at_fallback_decision(cfg, date_s, cached, 0.03)
                    out["fallback_refresh_uncovered"] = {"raised": False, "batter": ref.daily.pick.batter_name}
                except cs.ContestStateError as exc:
                    out["fallback_refresh_uncovered"] = {"raised": True, "err": str(exc).splitlines()[0]}

        # 7. Subtle cross-path probe: bts preview (optional loader) writes <date>.json on uncovered streak=0 (double!),
        #    then operator enters dated streak=56 (where policy would pick single, NOT double!), and fallback cascade fails!
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            with patch("bts.model.predict.run_pipeline", return_value=df):
                r_prev = runner.invoke(
                    cli.cli,
                    ["preview", "--date", date_s, "--picks-dir", str(picks), "--models-dir", str(models_dir)],
                )
            assert r_prev.exit_code == 0, r_prev.output
            preview_daily = picks_mod.load_pick(date_s, picks)
            preview_was_double = preview_daily.double_down is not None

            # Case 7a: state STILL uncovered when fallback cascade returns None -> dm delivery STILL blocked!
            cfg = {
                "orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(Path(tmp) / ".hb")},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            pick_dms, alert_dms = [], []
            st_a = sch.SchedulerState(
                date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            with (
                patch.object(orch, "run_cascade", return_value=(None, None)),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: pick_dms.append((r, t)) or "dm-p"),
                patch.object(halert, "send_dm", side_effect=lambda r, t: alert_dms.append((r, t)) or "dm-a"),
            ):
                ref_a = sch._refresh_pick_at_fallback_decision(cfg, date_s, preview_daily, 0.03)
                ok_a = sch._deliver_and_lock_pick(ref_a.daily, cfg, picks, st_a, date_s, "final fallback", selection=ref_a.selection)

            # Case 7b: operator supplies dated streak=56 (where policy only plays SINGLE, never double!),
            # then fallback cascade returns None -> _deliver_and_lock_pick accepts & delivers cached DOUBLE!
            d56 = {
                "schema_version": "bts_contest_streak_manual_v3",
                "effective_date": date_s,
                "season": 2027,
                "expires_at": REAL_DT.combine(dt.date(2027, 7, 15), dt.time.max, ET).isoformat(),
                "active_streak": 56,
                "best_streak": 56,
                "saver_state": "used",
                "recorded_at": NOW.isoformat(),
                "source": "manual_cli",
            }
            _put_json(picks, "contest_streak.manual.json", d56)
            pick_dms_b = []
            st_b = sch.SchedulerState(
                date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            preview_daily_b = picks_mod.load_pick(date_s, picks)
            with (
                patch.object(orch, "run_cascade", return_value=(None, None)),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: pick_dms_b.append((r, t)) or "dm-p"),
            ):
                ref_b = sch._refresh_pick_at_fallback_decision(cfg, date_s, preview_daily_b, 0.03)
                ok_b = sch._deliver_and_lock_pick(ref_b.daily, cfg, picks, st_b, date_s, "final fallback", selection=ref_b.selection)

            out["preview_plus_fallback_cascade_failure"] = {
                "preview_created_double_from_uncovered_streak_0": preview_was_double,
                "still_uncovered_dm_delivered": ok_a,
                "still_uncovered_pick_dms": len(pick_dms),
                "after_dated_56_cascade_fail_delivered_cached_double": ok_b and (preview_daily_b.double_down is not None),
                "after_dated_56_dm_text": pick_dms_b[0][1] if pick_dms_b else None,
            }

        # 8. Optional-loader live posting path: bts run (non-dry-run) on uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            posts = []
            with (
                patch("bts.model.predict.run_pipeline", return_value=df),
                patch.object(posting_mod, "should_post_now", return_value=True),
                patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-run"),
            ):
                r_run = runner.invoke(
                    cli.cli,
                    ["run", "--date", date_s, "--picks-dir", str(picks), "--models-dir", str(models_dir)],
                )
            out["bts_run_optional_loader_uncovered"] = {
                "exit_code": r_run.exit_code,
                "pick_saved": (picks / f"{date_s}.json").exists(),
                "posts_sent": len(posts),
            }

        # 9. Non-blocked paths on refused state: check-results, run_result_polling, EOD skip, day-outcome
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            dp = picks_mod.DailyPick(
                date=date_s,
                run_time=NOW.isoformat(),
                pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                double_down=None,
                runner_up=None,
                notification_sent=True,
                notification_channel="bluesky_dm",
                notification_id="dm-ok",
                delivered_at=NOW.isoformat(),
            )
            picks_mod.save_pick(dp, picks)
            dd.write_decision(
                date_s, picks, action="single", source="mdp", streak=0, saver_available=True,
                state_source="contest", state_status="fresh", delivery_status="delivered", scoreable=True,
            )
            with (
                patch.object(cli, "_now_et", return_value=NOW),
                patch.object(picks_mod, "resolve_daily_slot_results", return_value={"primary": "hit"}),
            ):
                r_chk = runner.invoke(cli.cli, ["check-results", "--date", date_s, "--picks-dir", str(picks)])
            streak_after_chk = json.loads((picks / "streak.json").read_text(encoding="utf-8"))["streak"]

            # EOD skip + day-outcome on a second refused directory
            picks_skip = Path(tmp) / "picks_skip"
            picks_skip.mkdir()
            seed_uncovered(picks_skip)
            st_skip = sch.SchedulerState(
                date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            st_skip.final_skip_candidate = {
                "source": "mdp",
                "primary": {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 101, "p_game_hit": 0.72},
                "double": None,
                "streak": 18,
                "saver_available": False,
                "state_source": "contest",
                "state_status": "fresh",
                "allow_double": True,
                "contest_source_date": "2027-07-14",
            }
            sch._write_endofday_skip(picks_skip, date_s, st_skip)
            cfg_toml = Path(tmp) / "do.toml"
            cfg_toml.write_text(
                f'[orchestrator]\npicks_dir = "{picks_skip}"\n[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n',
                encoding="utf-8",
            )
            dms_do = []
            with (
                patch.object(sch, "fetch_schedule", return_value=[{"gamePk": 101, "status": {"detailedState": "Final"}}]),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms_do.append((r, t)) or "dm"),
            ):
                r_do = runner.invoke(
                    cli.cli,
                    [
                        "day-outcome", "--date", date_s, "--config", str(cfg_toml),
                        "--picks-dir", str(picks_skip), "--expected-username", "stonehengee",
                        "--dm-recipient", "eric.test",
                    ],
                )
            out["non_blocked_grading_skip_day_outcome"] = {
                "check_results_exit": r_chk.exit_code,
                "streak_after_check_results": streak_after_chk,
                "eod_skip_written": (picks_skip / date_s / "decision.json").exists(),
                "day_outcome_exit": r_do.exit_code,
                "day_outcome_output": r_do.output.strip(),
                "day_outcome_dms": len(dms_do),
            }

    out["denied_network"] = AUDIT["denied_net"]
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caller-worker", action="store_true")
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    work = Path(__file__).resolve().parent

    if args.caller_worker:
        run_caller_worker(args.src, args.out)
        return

    res_678 = run_items_6_7_8(repo)
    base_callers_path = work / "callers_base.json"
    branch_callers_path = work / "callers_branch.json"
    script = Path(__file__).resolve()

    for src, out_p in [(work / "base/src", base_callers_path), (repo / "src", branch_callers_path)]:
        proc = subprocess.run(
            [sys.executable, str(script), "--caller-worker", "--src", str(src), "--out", str(out_p)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"caller worker failed ({src}):\n{proc.stdout}\n{proc.stderr}")

    base_c = json.loads(base_callers_path.read_text(encoding="utf-8"))
    branch_c = json.loads(branch_callers_path.read_text(encoding="utf-8"))
    assert base_c["denied_network"] == []
    assert branch_c["denied_network"] == []
    assert AUDIT["denied_net"] == []

    # Key caller assertions comparing base vs branch
    assert base_c["run_and_pick_uncovered"]["accepted"] is True
    assert branch_c["run_and_pick_uncovered"]["accepted"] is False
    assert base_c["orchestrate_initial_uncovered"] == {"returned": True, "pick_saved": True, "dms_count": 0, "posts_count": 1}
    assert branch_c["orchestrate_initial_uncovered"] == {"returned": False, "pick_saved": False, "dms_count": 1, "posts_count": 0}
    assert branch_c["orchestrate_mid_revoke_post_save"]["raised"] is True
    assert branch_c["orchestrate_mid_revoke_post_save"]["posts"] == 0
    assert base_c["scheduler_run_single_check_uncovered"]["pick_file_exists"] is True
    assert branch_c["scheduler_run_single_check_uncovered"] == {
        "should_post": False,
        "pick_result_none": True,
        "selection_none": True,
        "pick_file_exists": False,
        "decision_exists": False,
        "alert_dms": 1,
    }
    assert base_c["deliver_and_lock_dm"]["returned"] is True and base_c["deliver_and_lock_dm"]["pick_dms"] == 1
    assert branch_c["deliver_and_lock_dm"]["returned"] is False and branch_c["deliver_and_lock_dm"]["pick_dms"] == 0
    assert base_c["deliver_and_lock_public"]["returned"] is True and base_c["deliver_and_lock_public"]["posts"] == 1
    assert branch_c["deliver_and_lock_public"]["returned"] is False and branch_c["deliver_and_lock_public"]["posts"] == 0
    assert branch_c["deliver_and_lock_private"]["returned"] is True and branch_c["deliver_and_lock_private"]["pick_locked"] is True
    assert branch_c["deliver_and_lock_delivered"]["returned"] is True and branch_c["deliver_and_lock_delivered"]["pick_locked"] is True
    assert branch_c["deliver_and_lock_crash"]["returned"] is False and branch_c["deliver_and_lock_crash"]["pick_locked"] is True
    assert base_c["fallback_refresh_uncovered"]["raised"] is False
    assert branch_c["fallback_refresh_uncovered"]["raised"] is True
    assert branch_c["non_blocked_grading_skip_day_outcome"]["check_results_exit"] == 0
    assert branch_c["non_blocked_grading_skip_day_outcome"]["streak_after_check_results"] == 1
    assert branch_c["non_blocked_grading_skip_day_outcome"]["eod_skip_written"] is True
    assert branch_c["non_blocked_grading_skip_day_outcome"]["day_outcome_exit"] == 0

    full = {
        "items_6_7_8": res_678,
        "callers_base": base_c,
        "callers_branch": branch_c,
    }
    (work / "report_6_7_8_9.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(json.dumps({
        "item6": res_678["item6"],
        "item7_summary": {k: v for k, v in res_678["item7"].items() if k != "audit_step_order"},
        "item8_keys": list(res_678["item8"].keys()),
        "branch_callers": branch_c,
    }, indent=2))


if __name__ == "__main__":
    main()
