"""Complete, host-independent regression suite for whole-range review slice 2
(`BASESHA = 619abda7` .. `HEAD = bc143131`).

Exercises:
1. Mid-season (`2027-07-15`) and late-season (`2027-09-20`) automatic-path days across
   single hit/miss (established streak, saver earned), stale reading & multi-day
   operator saver trap, double-down (F2 gate + `day-outcome`), skip days (`reach57`
   and `emax_season_best` stop rule), postponed games (pre-selection, mid-day
   delivery refusal + duplicate EOD `[late_delivery]` CRITICAL DM, partial-void DD),
   doubleheaders (`is_doubleheader_game2`, unlocked recheck loop, `entered_game_unverified`),
   and void grading (post-lock rainout, L01 walk-only 0-AB, L01 suspended 0-pre-hit).
2. Deploy & startup on restored 2026 snapshot state under old config (`private`,
   `research`: proves 2026 `contest_streak.json` leak under `require_contest_state=False`
   and false-positive EOD `post_failure` CRITICAL DM on `private_locked`), half-edited
   configs (`dm` without contest flag -> exit 2 + 0 DMs across systemd restart loop;
   `contest_state_required=true` without `season_contract.json` -> `READY=1` canary pass +
   bounded DMs + clean overnight idle), and rollback to `BASESHA` after `HEAD` wrote files.
3. Cron (`scripts/cron-setup-hetzner.sh` under `research` and `enter`), CLI `--help`
   & Click option validation + stubbed execution for every cron line, restart/heartbeat
   and fetch staleness timing windows, and the `enter` DM matrix.
4. `deploy/systemd/bts-leaderboard.service` recipient drift blindspot across
   `unit_drift.check` and `scripts/install-systemd-hetzner.sh`.

Adoptable directly by pytest: no module-level import of `bts.model.predict`,
no process-wide state left changed, no network, no writes outside
`.carbon-review/whole-range/`.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from harness_lib import (  # noqa: E402
    BASE_SRC,
    HEAD_SRC,
    REPO_ROOT,
    REVIEW_DIR,
    GameSpec,
    RealDatetime,
    SyntheticMLBWorld,
    VirtualClock,
    base_live_config,
    harness_env,
    isolated_tmp_root,
    make_pinned_datetime_cls,
    run_worker_subprocess,
    stage_models,
    write_covered_auto_state,
    write_toml_config,
)

if not (len(sys.argv) >= 3 and sys.argv[1] == "--worker" and sys.argv[2] == "base"):
    if str(HEAD_SRC) not in sys.path:
        sys.path.insert(0, str(HEAD_SRC))


@contextlib.contextmanager
def _host_clean_env(
    workdir: Path,
    clock: VirtualClock,
    world: SyntheticMLBWorld,
    predict_fn,
    blend_dates: tuple[str, ...],
):
    models_dir = stage_models(workdir)
    for d in blend_dates:
        p = models_dir / f"blend_{d}.pkl"
        if not p.exists():
            p.write_bytes(b"synthetic_blend_placeholder")
    fake_usage = shutil._ntuple_diskusage(100_000_000_000, 30_000_000_000, 70_000_000_000)
    with patch.object(shutil, "disk_usage", lambda *a, **k: fake_usage), \
         harness_env(workdir, clock, world, predict_fn) as cap:
        yield cap


def _pred_df(world: SyntheticMLBWorld, rows: list[dict]) -> pd.DataFrame:
    now_et = world.clock.now_et()
    records = []
    for r in rows:
        g_found = None
        for glist in world.games_by_date.values():
            for g in glist:
                if g.game_pk == r["game_pk"]:
                    g_found = g
                    break
        side = r.get("side", "away")
        confirmed = True
        if g_found is not None:
            spec = g_found.away_confirm_at_et if side == "away" else g_found.home_confirm_at_et
            confirmed = world._side_confirmed(spec, now_et)
        flags = "" if confirmed else "PROJECTED"
        records.append({
            "batter_name": r["batter_name"],
            "batter_id": r["batter_id"],
            "team": r["team"],
            "lineup": r.get("lineup", 1),
            "pitcher_name": r.get("pitcher_name", "Synth Pitcher"),
            "pitcher_id": r.get("pitcher_id", 770001),
            "pitcher_team": r.get("pitcher_team", "OPP"),
            "game_pk": r["game_pk"],
            "game_time": r["game_time"],
            "p_hit_pa": round(r["p_game_hit"] * 0.35, 4),
            "p_game_hit": r["p_game_hit"],
            "flags": flags,
        })
    return pd.DataFrame(records)


def _tomorrow_game(date_str: str) -> tuple[str, list[GameSpec]]:
    d = RealDatetime.strptime(date_str, "%Y-%m-%d").date()
    nxt = d.fromordinal(d.toordinal() + 1).isoformat()
    return nxt, [
        GameSpec(
            game_pk=99001,
            official_date=nxt,
            game_date_utc=f"{nxt}T23:05:00Z",
            batters=[{"id": 600001, "name": "Next Day Batter", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
        )
    ]


def _collect_run_artifacts(workdir: Path, date_str: str, captured, stderr_buf: io.StringIO) -> dict:
    picks_dir = workdir / "data" / "picks"
    health_dir = workdir / "data" / "health_state"
    pick_file = picks_dir / f"{date_str}.json"
    dec_file = picks_dir / date_str / "decision.json"
    state_file = picks_dir / date_str / "scheduler_state.json"
    streak_file = picks_dir / "streak.json"
    saver_file = picks_dir / "account_state" / "saver_state.json"
    hb_file = workdir / "data" / ".heartbeat"
    health_files = sorted(p.name for p in health_dir.glob("*.json")) if health_dir.exists() else []
    health_dm_status = None
    if (health_dir / "health_dm_delivery_status.json").exists():
        health_dm_status = json.loads((health_dir / "health_dm_delivery_status.json").read_text())
    return {
        "dms": captured.dms,
        "posts": captured.posts,
        "replies": captured.replies,
        "pick_json": json.loads(pick_file.read_text()) if pick_file.exists() else None,
        "decision_json": json.loads(dec_file.read_text()) if dec_file.exists() else None,
        "scheduler_state": json.loads(state_file.read_text()) if state_file.exists() else None,
        "streak_json": json.loads(streak_file.read_text()) if streak_file.exists() else None,
        "saver_json": json.loads(saver_file.read_text()) if saver_file.exists() else None,
        "heartbeat": json.loads(hb_file.read_text()) if hb_file.exists() else None,
        "health_files": health_files,
        "health_dm_status": health_dm_status,
        "stderr": stderr_buf.getvalue(),
    }


@contextlib.contextmanager
def _maybe_horizon_patch(mode: str):
    if mode == "horizon":
        import bts.strategy as strat_mod
        with patch.object(strat_mod, "SEASON_END_DATE", "2027-09-27"):
            strat_mod._mdp_cache = None
            yield
    else:
        yield


def _single_game_world(clock: VirtualClock, date_str: str, nxt_str: str, p_hit: float = 0.822) -> tuple[SyntheticMLBWorld, list[dict]]:
    games = {
        date_str: [
            GameSpec(
                game_pk=7001, official_date=date_str, game_date_utc=f"{date_str}T23:35:00Z",
                batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
            ),
        ],
        nxt_str: [
            GameSpec(
                game_pk=7002, official_date=nxt_str, game_date_utc=f"{nxt_str}T23:05:00Z",
                batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
            ),
        ],
    }
    rows = [{
        "batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "lineup": 1,
        "pitcher_name": "Synth Pitcher", "pitcher_id": 770001, "pitcher_team": "BOS",
        "game_pk": 7001, "game_time": f"{date_str}T23:35:00Z", "p_hit_pa": 0.29,
        "p_game_hit": p_hit, "flags": "",
    }]
    return SyntheticMLBWorld(clock, games), rows


# ---------------------------------------------------------------------------
# Subprocess worker dispatcher (isolates HEAD_SRC vs BASE_SRC cleanly)
# ---------------------------------------------------------------------------

def _worker_entry(code_tree: str, scenario: str, extra: list[str]) -> dict:
    import bts
    import bts.scheduler as sch_mod

    expected_src = (HEAD_SRC if code_tree == "head" else BASE_SRC).resolve()
    actual_bts = Path(bts.__file__).resolve()
    assert actual_bts.is_relative_to(expected_src), f"Expected {expected_src}, imported {actual_bts}"

    if scenario.startswith("rb_"):
        import bts.contest_state as cs_mod
        import bts.health.contest_state as h_cs_mod
        import bts.picks as picks_mod
        import bts.web as web_mod

        workdir = Path(extra[0])
        picks_dir = workdir / "data" / "picks"
        clock = VirtualClock("2027-07-15T18:10:00-04:00")
        world, rows = _single_game_world(clock, "2027-07-15", "2027-07-16")

        if scenario == "rb_inspect_with_auto":
            buf = io.StringIO()
            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-07-15", "2027-07-16")) as cap, \
                 contextlib.redirect_stderr(buf):
                loaded = picks_mod.load_pick("2027-07-15", picks_dir)
                loaded_sh = picks_mod.load_shadow_pick("2027-07-15", picks_dir)
                assert loaded is not None and loaded_sh is not None
                with contextlib.ExitStack() as web_stack:
                    if hasattr(web_mod, "fetch_bluesky_posts"):
                        web_stack.enter_context(patch.object(web_mod, "fetch_bluesky_posts", lambda *a, **k: []))
                    if hasattr(web_mod, "datetime"):
                        web_stack.enter_context(patch.object(web_mod, "datetime", make_pinned_datetime_cls(clock)))
                    html = web_mod.render_page()
                    assert "Luis Arraez" in html
                ds = cs_mod.load_decision_streak_state(picks_dir, require_contest_state=True, now=clock.now_utc())
                h_alerts = [
                    {"level": a.level, "source": a.source, "message": a.message}
                    for a in h_cs_mod.check(picks_dir, expected=True, now=clock.now_utc())
                ]
                sch_mod.run_day(date="2027-07-15", config=base_live_config(workdir))
                pick_after = json.loads((picks_dir / "2027-07-15.json").read_text())
                streak_after = json.loads((picks_dir / "streak.json").read_text())
                return {
                    "decision_streak_state": {
                        "streak": ds.streak, "saver_available": ds.saver_available,
                        "source": ds.source, "best_streak": ds.best_streak, "best_status": ds.best_status,
                    },
                    "health_contest_alerts": h_alerts,
                    "dms_sent_by_basesha": [d["text"] for d in cap.dms],
                    "pick_keys_after_basesha_save": sorted(pick_after.keys()),
                    "pick_result_after_basesha": pick_after["result"],
                    "streak_after_basesha_save": streak_after,
                }

        if scenario == "rb_v3_without_auto":
            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")):
                ds_day1 = cs_mod.load_decision_streak_state(picks_dir, require_contest_state=True, now=clock.now_utc())
                alerts_day1 = [
                    {"level": a.level, "message": a.message}
                    for a in h_cs_mod.check(picks_dir, expected=True, now=clock.now_utc())
                ]
                clock.set_et("2027-07-20T12:00:00-04:00")
                ds_day6 = cs_mod.load_decision_streak_state(picks_dir, require_contest_state=True, now=clock.now_utc())
                return {
                    "day1": {
                        "streak": ds_day1.streak, "saver_available": ds_day1.saver_available,
                        "best_status": ds_day1.best_status, "alerts": alerts_day1,
                    },
                    "day6_still_active": {"streak": ds_day6.streak, "saver_available": ds_day6.saver_available},
                }

        if scenario == "rb_unlocked_day":
            buf = io.StringIO()
            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-07-15", "2027-07-16")) as cap, \
                 contextlib.redirect_stderr(buf):
                sch_mod.run_day(date="2027-07-15", config=base_live_config(workdir))
            return {
                "dms": [d["text"] for d in cap.dms],
                "decision": json.loads((picks_dir / "2027-07-15" / "decision.json").read_text()),
            }

    mode = extra[0] if extra else "default"
    auto_schema = extra[1] if len(extra) > 1 else ("bts_contest_streak_auto_v2" if code_tree == "head" else "bts_contest_streak_auto_v1")

    with isolated_tmp_root(prefix=f"w_{code_tree}_{scenario}_") as workdir:
        with _maybe_horizon_patch(mode):
            if scenario in ("single_mid_hit", "single_mid_miss", "single_late_hit"):
                is_late = scenario == "single_late_hit"
                is_miss = scenario == "single_mid_miss"
                date_str = "2027-09-20" if is_late else "2027-07-15"
                prev_date = "2027-09-19" if is_late else "2027-07-14"
                streak = 16 if is_late else 12
                best = 18 if is_late else 15
                p1 = 0.835 if is_late else 0.818
                p2 = 0.830 if is_late else 0.812
                hits = 0 if is_miss else 1

                # Start at 11:00 ET with 10:30 ET fetch (`14:30Z`) so reading stays <14h fresh through evening lock
                clock = VirtualClock(f"{date_str}T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)
                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7001, official_date=date_str, game_date_utc=f"{date_str}T23:05:00Z",
                            batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": hits}}],
                        ),
                        GameSpec(
                            game_pk=7002, official_date=date_str, game_date_utc=f"{date_str}T23:40:00Z",
                            away_name="Houston Astros", away_abbrev="HOU", home_name="Texas Rangers", home_abbrev="TEX",
                            batters=[{"id": 500002, "name": "Yordan Alvarez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=streak, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T14:30:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": 7001, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p1},
                    {"batter_name": "Yordan Alvarez", "batter_id": 500002, "team": "HOU", "game_pk": 7002, "game_time": f"{date_str}T23:40:00Z", "p_game_hit": p2},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario == "stale_mid":
                date_str = "2027-07-15"
                clock = VirtualClock("2027-07-15T10:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)
                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7001, official_date=date_str, game_date_utc="2027-07-15T23:05:00Z",
                            batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=12, best_streak=15,
                    source_date="2027-07-14", recorded_at_utc="2027-07-14T18:00:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": 7001, "game_time": "2027-07-15T23:05:00Z", "p_game_hit": 0.818},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("double_mid", "double_late"):
                is_late = scenario == "double_late"
                date_str = "2027-09-20" if is_late else "2027-07-15"
                prev_date = "2027-09-19" if is_late else "2027-07-14"
                clock = VirtualClock(f"{date_str}T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)
                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7101, official_date=date_str, game_date_utc=f"{date_str}T23:05:00Z",
                            away_confirm_at_et=f"{date_str}T17:30:00-04:00",
                            home_confirm_at_et=f"{date_str}T17:30:00-04:00",
                            batters=[{"id": 500001, "name": "Bobby Witt Jr.", "side": "away", "batting": {"atBats": 4, "hits": 2}}],
                        ),
                        GameSpec(
                            game_pk=7102, official_date=date_str, game_date_utc=f"{date_str}T23:25:00Z",
                            away_name="Los Angeles Dodgers", away_abbrev="LAD", away_id=119,
                            home_name="San Diego Padres", home_abbrev="SD", home_id=135,
                            away_confirm_at_et=f"{date_str}T18:20:00-04:00",
                            home_confirm_at_et=f"{date_str}T18:20:00-04:00",
                            batters=[{"id": 500002, "name": "Freddie Freeman", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=12, best_streak=15,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T14:30:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Bobby Witt Jr.", "batter_id": 500001, "team": "NYY", "game_pk": 7101, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": 0.835},
                    {"batter_name": "Freddie Freeman", "batter_id": 500002, "team": "LAD", "game_pk": 7102, "game_time": f"{date_str}T23:25:00Z", "p_game_hit": 0.831},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("skip_mid", "skip_late_stop"):
                is_late = scenario == "skip_late_stop"
                date_str = "2027-09-20" if is_late else "2027-07-15"
                prev_date = "2027-09-19" if is_late else "2027-07-14"
                best = 30 if is_late else 15
                p1 = 0.860 if is_late else 0.780
                clock = VirtualClock(f"{date_str}T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)
                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7201, official_date=date_str, game_date_utc=f"{date_str}T23:05:00Z",
                            batters=[{"id": 500001, "name": "Steven Kwan", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                        GameSpec(
                            game_pk=7202, official_date=date_str, game_date_utc=f"{date_str}T23:35:00Z",
                            away_name="Atlanta Braves", away_abbrev="ATL", away_id=144,
                            batters=[{"id": 500002, "name": "Ronald Acuna Jr.", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                picks_dir = workdir / "data" / "picks"
                write_covered_auto_state(
                    picks_dir, season=2027, streak=12, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T14:30:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                (picks_dir / f"{date_str}.json").write_text(json.dumps({
                    "date": date_str, "run_time": f"{date_str}T07:05:00Z",
                    "pick": {
                        "batter_name": "Steven Kwan", "batter_id": 500001, "team": "NYY",
                        "lineup_position": 1, "pitcher_name": "Synth Pitcher", "pitcher_id": 770001,
                        "p_game_hit": 0.815, "flags": ["PROJECTED"], "projected_lineup": True,
                        "game_pk": 7201, "game_time": f"{date_str}T23:05:00Z",
                    },
                    "double_down": None, "runner_up": None,
                }, indent=2))
                rows = [
                    {"batter_name": "Steven Kwan", "batter_id": 500001, "team": "NYY", "game_pk": 7201, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p1},
                    {"batter_name": "Ronald Acuna Jr.", "batter_id": 500002, "team": "ATL", "game_pk": 7202, "game_time": f"{date_str}T23:35:00Z", "p_game_hit": p1 - 0.01},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("postponed_pre", "postponed_midday_mid", "postponed_midday_late"):
                is_late = scenario == "postponed_midday_late"
                is_pre = scenario == "postponed_pre"
                date_str = extra[2] if (is_pre and len(extra) > 2) else ("2027-09-20" if is_late else "2027-07-15")
                prev_date = "2027-09-19" if date_str == "2027-09-20" else "2027-07-14"
                streak = 16 if date_str == "2027-09-20" else 12
                best = 18 if date_str == "2027-09-20" else 15
                p_g1 = 0.840 if date_str == "2027-09-20" else 0.822
                p_g2 = 0.825 if date_str == "2027-09-20" else 0.818

                clock = VirtualClock(f"{date_str}T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)
                state_path = workdir / "data" / "picks" / date_str / "scheduler_state.json"

                def _g1_status(_now_et):
                    if is_pre:
                        return ("F", "Postponed", "DR")
                    if state_path.exists():
                        try:
                            st_doc = json.loads(state_path.read_text())
                            if len(st_doc.get("runs_completed") or []) >= 1:
                                return ("F", "Postponed", "DR")
                        except Exception:
                            pass
                    return ("P", "Scheduled", "S")

                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7301, official_date=date_str, game_date_utc=f"{date_str}T23:05:00Z",
                            status_override=_g1_status,
                            batters=[{"id": 500001, "name": "Corey Seager", "side": "away", "batting": {"atBats": 0, "hits": 0}}],
                        ),
                        GameSpec(
                            game_pk=7302, official_date=date_str, game_date_utc=f"{date_str}T23:50:00Z",
                            away_name="Seattle Mariners", away_abbrev="SEA", away_id=136,
                            batters=[{"id": 500002, "name": "Julio Rodriguez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=streak, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T14:30:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Corey Seager", "batter_id": 500001, "team": "NYY", "game_pk": 7301, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p_g1},
                    {"batter_name": "Julio Rodriguez", "batter_id": 500002, "team": "SEA", "game_pk": 7302, "game_time": f"{date_str}T23:50:00Z", "p_game_hit": p_g2},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario == "partial_void_double":
                date_str = "2027-07-15"
                clock = VirtualClock("2027-07-15T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)

                def _g2_post_lock_rainout(now_et):
                    if now_et >= RealDatetime.fromisoformat("2027-07-15T19:00:00-04:00"):
                        return ("F", "Postponed", "DR")
                    return ("P", "Scheduled", "S")

                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7401, official_date=date_str, game_date_utc="2027-07-15T23:05:00Z",
                            batters=[{"id": 500001, "name": "Bobby Witt Jr.", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                        GameSpec(
                            game_pk=7402, official_date=date_str, game_date_utc="2027-07-15T23:35:00Z",
                            away_name="Chicago Cubs", away_abbrev="CHC", away_id=112,
                            status_override=_g2_post_lock_rainout,
                            batters=[{"id": 500002, "name": "Ian Happ", "side": "away", "batting": {"atBats": 0, "hits": 0}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=12, best_streak=15,
                    source_date="2027-07-14", recorded_at_utc="2027-07-15T14:30:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Bobby Witt Jr.", "batter_id": 500001, "team": "NYY", "game_pk": 7401, "game_time": "2027-07-15T23:05:00Z", "p_game_hit": 0.836},
                    {"batter_name": "Ian Happ", "batter_id": 500002, "team": "CHC", "game_pk": 7402, "game_time": "2027-07-15T23:35:00Z", "p_game_hit": 0.831},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                    import bts.picks as picks_mod
                    clock.set_et("2027-07-16T02:00:00-04:00")
                    picks_mod.reconcile_results(workdir / "data" / "picks", lookback_days=3, clock=clock.now_et)
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("dh_locked", "dh_unlocked_recheck"):
                date_str = extra[2] if len(extra) > 2 else "2027-07-15"
                prev_date = "2027-09-19" if date_str == "2027-09-20" else "2027-07-14"
                is_skip = scenario == "dh_unlocked_recheck"
                streak = 16 if date_str == "2027-09-20" else 12
                best = 30 if (is_skip and date_str == "2027-09-20") else (18 if date_str == "2027-09-20" else 15)
                p1 = 0.775 if is_skip else (0.835 if date_str == "2027-09-20" else 0.818)

                clock = VirtualClock(f"{date_str}T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)
                g2_confirm = f"{date_str}T18:40:00-04:00" if is_skip else "always"
                games = {
                    date_str: [
                        GameSpec(
                            game_pk=8001, official_date=date_str, game_date_utc=f"{date_str}T17:10:00Z",
                            double_header="S",
                            batters=[{"id": 500001, "name": "Rafael Devers", "side": "home", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                        GameSpec(
                            game_pk=8002, official_date=date_str, game_date_utc=f"{date_str}T23:10:00Z",
                            double_header="S", away_confirm_at_et=g2_confirm, home_confirm_at_et=g2_confirm,
                            batters=[{"id": 500002, "name": "Aaron Judge", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=streak, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T14:30:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Rafael Devers", "batter_id": 500001, "team": "BOS", "side": "home", "game_pk": 8001, "game_time": f"{date_str}T17:10:00Z", "p_game_hit": p1},
                    {"batter_name": "Aaron Judge", "batter_id": 500002, "team": "NYY", "side": "away", "game_pk": 8002, "game_time": f"{date_str}T23:10:00Z", "p_game_hit": p1 - 0.005},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("void_rainout", "void_walk_only", "void_suspended_pre0"):
                date_str = extra[2] if len(extra) > 2 else "2027-07-15"
                prev_date = "2027-09-19" if date_str == "2027-09-20" else "2027-07-14"
                clock = VirtualClock(f"{date_str}T11:00:00-04:00")
                nxt, nxt_games = _tomorrow_game(date_str)

                status_override = None
                resume_dt = None
                plays = None
                batting = {"atBats": 4, "hits": 1}
                if scenario == "void_rainout":
                    def _rain(now_et):
                        if now_et >= RealDatetime.fromisoformat(f"{date_str}T19:00:00-04:00"):
                            return ("F", "Postponed", "DR")
                        return ("P", "Scheduled", "S")
                    status_override = _rain
                    batting = {"atBats": 0, "hits": 0}
                elif scenario == "void_walk_only":
                    batting = {"atBats": 0, "hits": 0, "baseOnBalls": 4, "sacFlies": 0}
                elif scenario == "void_suspended_pre0":
                    resume_dt = f"{nxt}T18:00:00Z"
                    batting = {"atBats": 3, "hits": 1}
                    plays = [
                        {
                            "result": {"eventType": "field_out"},
                            "matchup": {"batter": {"id": 500001, "fullName": "Mookie Betts"}},
                            "about": {"startTime": f"{date_str}T23:20:00Z"},
                        },
                        {
                            "result": {"eventType": "field_out"},
                            "matchup": {"batter": {"id": 500001, "fullName": "Mookie Betts"}},
                            "about": {"startTime": f"{date_str}T23:55:00Z"},
                        },
                        {
                            "result": {"eventType": "single"},
                            "matchup": {"batter": {"id": 500001, "fullName": "Mookie Betts"}},
                            "about": {"startTime": f"{nxt}T18:30:00Z"},
                        },
                    ]

                games = {
                    date_str: [
                        GameSpec(
                            game_pk=7501, official_date=date_str, game_date_utc=f"{date_str}T23:05:00Z",
                            status_override=status_override, resume_datetime_utc=resume_dt, plays=plays,
                            batters=[{"id": 500001, "name": "Mookie Betts", "side": "away", "batting": batting}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks", season=2027, streak=20, best_streak=22,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T14:30:00Z",
                    saver_state="used", model_streak=20, model_saver=False, auto_schema=auto_schema,
                )
                p1 = 0.835 if date_str == "2027-09-20" else 0.818
                rows = [
                    {"batter_name": "Mookie Betts", "batter_id": 500001, "team": "NYY", "game_pk": 7501, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p1},
                ]
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=base_live_config(workdir))
                return _collect_run_artifacts(workdir, date_str, cap, buf)

    raise ValueError(f"Unknown scenario: {scenario}")


# ===========================================================================
# ITEM 1 TESTS
# ===========================================================================

def test_item1_single_hit_and_miss() -> dict:
    script = Path(__file__).resolve()
    out = {}
    for scen in ("single_mid_hit", "single_mid_miss", "single_late_hit"):
        h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2"])
        b_lit = run_worker_subprocess(script, "base", scen, ["default", "bts_contest_streak_auto_v2"])
        b_hor = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1"])

        assert b_lit["decision_json"]["action"] == "skip"
        assert len(b_lit["dms"]) == 1 and "No pick yet" in b_lit["dms"][0]["text"]

        assert len(h["dms"]) == 1 and len(b_hor["dms"]) == 1
        assert h["dms"][0]["text"] == b_hor["dms"][0]["text"]
        assert h["decision_json"]["action"] == "single" == b_hor["decision_json"]["action"]

        assert set(h["pick_json"]) - set(b_hor["pick_json"]) == {"chosen_on", "automatic_reading"}
        assert set(h["streak_json"]) - set(b_hor["streak_json"]) == {"season", "applied_dates"}

        if scen == "single_mid_miss":
            assert h["streak_json"]["streak"] == 12 and h["streak_json"]["saver_available"] is False
            assert h["saver_json"]["state"] == "active"
        elif scen == "single_mid_hit":
            assert h["streak_json"]["streak"] == 13 == b_hor["streak_json"]["streak"]
            assert h["decision_json"]["objective"] == "reach57"
        elif scen == "single_late_hit":
            assert h["streak_json"]["streak"] == 17 == b_hor["streak_json"]["streak"]
            assert h["decision_json"]["objective"] == "emax_season_best"

        out[scen] = {"head_dm": h["dms"][0]["text"], "head_streak_after": h["streak_json"]}
    return out


def test_item1_stale_and_multiday_saver_trap() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    import bts.saver_state as sv_mod
    import bts.scheduler as sch_mod

    script = Path(__file__).resolve()
    h_stale = run_worker_subprocess(script, "head", "stale_mid", ["default", "bts_contest_streak_auto_v2"])
    b_stale = run_worker_subprocess(script, "base", "stale_mid", ["horizon", "bts_contest_streak_auto_v1"])
    assert len(h_stale["dms"]) == 2 and len(b_stale["dms"]) == 1
    assert "BTS health WARN attention:\n- [contest_state] Morning check for 2027-07-15" in h_stale["dms"][0]["text"]
    assert "saver-state" not in h_stale["dms"][0]["text"]
    assert h_stale["pick_json"]["automatic_reading"]["stale"] is True

    with isolated_tmp_root(prefix="multiday_saver_trap_") as workdir:
        clock = VirtualClock("2027-07-15T10:00:00-04:00")
        games = {
            d: [GameSpec(game_pk=7600 + i, official_date=d, game_date_utc=f"{d}T23:05:00Z",
                         batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}])]
            for i, d in enumerate(("2027-07-15", "2027-07-16", "2027-07-17"), start=1)
        }
        world = SyntheticMLBWorld(clock, games)
        picks_dir = workdir / "data" / "picks"
        write_covered_auto_state(
            picks_dir, season=2027, streak=14, best_streak=15,
            source_date="2027-07-13", recorded_at_utc="2027-07-14T18:00:00Z", saver_state="active",
        )
        cfg = base_live_config(workdir)
        buf15, buf16 = io.StringIO(), io.StringIO()

        def _preds_day(*a, **k):
            cur = clock.now_et().date().isoformat()
            pk = 7601 if cur == "2027-07-15" else 7602
            return _pred_df(world, [
                {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": pk, "game_time": f"{cur}T23:05:00Z", "p_game_hit": 0.805},
            ])

        with _host_clean_env(workdir, clock, world, _preds_day, ("2027-07-15", "2027-07-16", "2027-07-17")) as cap:
            clock.set_et("2027-07-15T10:15:00-04:00")
            res_cli = CliRunner().invoke(cli_mod.cli, [
                "set-contest-streak", "--date", "2027-07-15",
                "--streak", "14", "--best-streak", "15", "--saver", "used",
                "--picks-dir", str(picks_dir),
            ])
            assert res_cli.exit_code == 0, res_cli.output

            with contextlib.redirect_stderr(buf15):
                sch_mod.run_day(date="2027-07-15", config=cfg)
            dec15 = json.loads((picks_dir / "2027-07-15" / "decision.json").read_text())
            assert dec15["action"] == "skip" and dec15["saver_available"] is False

            # Refresh contest_streak.json at 2027-07-16 10:30 ET (`14:30Z`) + maybe_auto_earn_saver
            clock.set_et("2027-07-16T10:30:00-04:00")
            obs_path = picks_dir / "account_state" / "contest_streak.json"
            obs = json.loads(obs_path.read_text())
            obs.update(active_streak=14, best_streak=15, source_date="2027-07-15", recorded_at="2027-07-16T14:30:00Z")
            obs_path.write_text(json.dumps(obs, indent=2) + "\n")
            sv_mod.maybe_auto_earn_saver(picks_dir, best_streak=15, season=2027)

            dms_before_16 = len(cap.dms)
            clock.set_et("2027-07-16T11:00:00-04:00")
            with contextlib.redirect_stderr(buf16):
                sch_mod.run_day(date="2027-07-16", config=cfg)
            dec16 = json.loads((picks_dir / "2027-07-16" / "decision.json").read_text())
            pick16 = json.loads((picks_dir / "2027-07-16.json").read_text())
            dms_16 = cap.dms[dms_before_16:]

            assert "Morning state check: automatic contest state accepted (streak 14)." in buf16.getvalue()
            assert dec16["action"] == "single" and dec16["saver_available"] is True
            assert pick16["chosen_on"]["saver_available"] is True
            assert any("Today's pick: Luis Arraez" in d["text"] for d in dms_16), dms_16

    return {
        "day1_dated_saver_used_action": dec15["action"],
        "day2_unwarned_stale_saver_flag_action": dec16["action"],
        "day2_chosen_on": pick16["chosen_on"],
    }


def test_item1_double_down_and_day_outcome() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    from bts import entry_receipt, receipt_io

    script = Path(__file__).resolve()
    out = {}
    for scen in ("double_mid", "double_late"):
        h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2"])
        b_hor = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1"])
        # F2 gate blocked lock at 18:05 ET check while Game 2 was PROJECTED, then locked at 18:25 ET check!
        assert "should_lock=False" in h["stderr"] and "should_lock=True" in h["stderr"]
        assert len(h["dms"]) == 1 and "DOUBLE DOWN" in h["dms"][0]["text"]
        assert h["dms"][0]["text"] == b_hor["dms"][0]["text"]
        assert h["streak_json"]["streak"] == 14 == b_hor["streak_json"]["streak"]
        out[scen] = {"dm": h["dms"][0]["text"], "objective": h["decision_json"]["objective"]}

    with isolated_tmp_root(prefix="day_outcome_dd_") as workdir:
        date_str = "2027-07-15"
        clock = VirtualClock("2027-07-16T08:30:00-04:00")
        world = SyntheticMLBWorld(clock, {
            date_str: [
                GameSpec(game_pk=7101, official_date=date_str, game_date_utc="2027-07-15T23:05:00Z"),
                GameSpec(game_pk=7102, official_date=date_str, game_date_utc="2027-07-15T23:25:00Z"),
            ]
        })
        picks_dir = workdir / "data" / "picks"
        (picks_dir / date_str).mkdir(parents=True, exist_ok=True)
        dec_bytes = (json.dumps({
            "schema_version": "bts_daily_decision_v1", "date": date_str, "action": "double",
            "source": "mdp", "scoreable": True, "delivery_status": "delivered",
            "primary": {"batter_name": "Bobby Witt Jr.", "batter_id": 500001, "game_pk": 7101},
            "double_down": {"batter_name": "Freddie Freeman", "batter_id": 500002, "game_pk": 7102},
        }, indent=2) + "\n").encode("utf-8")
        (picks_dir / date_str / "decision.json").write_bytes(dec_bytes)
        dec_sha = hashlib.sha256(dec_bytes).hexdigest()
        cfg_path = write_toml_config(workdir / "orch.toml", base_live_config(workdir))

        def _pub(attempt_id: str, all_conf: bool, reason: str, slots: list[dict]) -> None:
            body = {
                "schema": "bts_pick_entry_receipt_v1", "attempt_id": attempt_id,
                "et_date": date_str, "season": 2027, "outcome": "observed", "degraded": [],
                "account": {"user_id": 12345, "username": "stonehengee"},
                "selection": {"commit": {"decision_sha256": dec_sha}},
                "observation": {"before_cutoff": True, "response_completed_at": "2027-07-15T22:30:00Z"},
                "verifier": {"ok": all_conf, "reason": reason},
                "qualification": {"all_confirmed": all_conf, "slots": slots},
            }
            receipt_io.publish(
                entry_receipt.receipts_root(picks_dir) / date_str / f"20270715T223000-{attempt_id}.json",
                (json.dumps(body, indent=1, sort_keys=True) + "\n").encode("utf-8"),
            )

        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")) as cap:
            runner = CliRunner()
            base_args = ["day-outcome", "--date", date_str, "--config", str(cfg_path),
                         "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
                         "--dm-recipient", "stonehengee.bsky.social"]
            r0 = runner.invoke(cli_mod.cli, base_args)
            assert r0.exit_code == 1 and json.loads(r0.output)["state"] == "entry_unverifiable"
            _pub("001", False, "mismatch", [{"role": "pick", "state": "confirmed"}, {"role": "double_down", "state": "missing"}])
            r1 = runner.invoke(cli_mod.cli, base_args)
            assert r1.exit_code == 1 and json.loads(r1.output)["state"] == "not_entered"
            _pub("002", True, "match", [{"role": "pick", "state": "confirmed"}, {"role": "double_down", "state": "confirmed"}])
            r2 = runner.invoke(cli_mod.cli, base_args)
            assert r2.exit_code == 0 and json.loads(r2.output)["state"] == "entered"
            assert len(cap.dms) == 2
        out["day_outcome_states"] = [json.loads(r0.output)["state"], json.loads(r1.output)["state"], json.loads(r2.output)["state"]]
    return out


def test_item1_skip_days() -> dict:
    script = Path(__file__).resolve()
    h_mid = run_worker_subprocess(script, "head", "skip_mid", ["default", "bts_contest_streak_auto_v2"])
    b_mid = run_worker_subprocess(script, "base", "skip_mid", ["horizon", "bts_contest_streak_auto_v1"])
    assert h_mid["decision_json"]["action"] == "skip" == b_mid["decision_json"]["action"]
    assert h_mid["dms"][0]["text"] == b_mid["dms"][0]["text"]

    h_late = run_worker_subprocess(script, "head", "skip_late_stop", ["default", "bts_contest_streak_auto_v2"])
    b_late_v1 = run_worker_subprocess(script, "base", "skip_late_stop", ["horizon", "bts_contest_streak_auto_v1"])
    b_late_v2 = run_worker_subprocess(script, "base", "skip_late_stop", ["horizon", "bts_contest_streak_auto_v2"])
    assert h_late["decision_json"]["action"] == "skip" and h_late["decision_json"]["best_status"] == "trusted"
    assert b_late_v1["decision_json"]["action"] == "skip"
    # On HEAD's auto_v2 file, BASESHA marks best_status='untrusted' (`effective_best=12`), playing 'single' instead of skipping!
    assert b_late_v2["decision_json"]["best_status"] == "untrusted" and b_late_v2["decision_json"]["action"] == "single"
    return {
        "late_head_action": h_late["decision_json"]["action"],
        "late_base_on_auto_v2_action": b_late_v2["decision_json"]["action"],
    }


def test_item1_postponed_and_duplicate_eod_alert() -> dict:
    import bts.scheduler as sch_mod
    from bts.health.runner import run_all_checks
    from bts.picks import DailyPick, Pick

    script = Path(__file__).resolve()
    out = {}
    for date_str in ("2027-07-15", "2027-09-20"):
        hp = run_worker_subprocess(script, "head", "postponed_pre", ["default", "bts_contest_streak_auto_v2", date_str])
        bp = run_worker_subprocess(script, "base", "postponed_pre", ["horizon", "bts_contest_streak_auto_v1", date_str])
        assert len(hp["dms"]) == 1 and hp["dms"][0]["text"] == bp["dms"][0]["text"]

    for scen in ("postponed_midday_mid", "postponed_midday_late"):
        h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2"])
        b_hor = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1"])
        assert len(h["dms"]) == 3, [d["text"] for d in h["dms"]]
        assert "- [postponed_pick] DELIVERY REFUSED" in h["dms"][0]["text"]
        assert "Today's pick: Julio Rodriguez" in h["dms"][1]["text"]
        assert "- [late_delivery] delivery REFUSED" in h["dms"][2]["text"]
        assert sorted(h["health_dm_status"]["sent_sources"]) == ["CRITICAL:late_delivery", "CRITICAL:postponed_pick"]
        assert len(b_hor["dms"]) == 1 and "Corey Seager" in b_hor["dms"][0]["text"] and b_hor["pick_json"]["result"] == "void"
        out[scen] = {"head_dms": [d["text"] for d in h["dms"]], "base_result": b_hor["pick_json"]["result"]}

    with isolated_tmp_root(prefix="late_refusal_dedup_ctrl_") as workdir:
        clock = VirtualClock("2027-07-15T19:02:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-07-15": []})
        picks_dir = workdir / "data" / "picks"
        write_covered_auto_state(picks_dir, season=2027, streak=12, best_streak=15, source_date="2027-07-14", recorded_at_utc="2027-07-15T14:30:00Z")
        cfg = base_live_config(workdir)
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")) as cap:
            st = sch_mod.SchedulerState(
                date="2027-07-15", schedule_fetched_at=clock.now_et().isoformat(),
                games=[], confirmed_game_pks=[], runs_completed=[],
                pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            dp = DailyPick("2027-07-15", clock.now_et().isoformat(),
                           Pick("Late Batter", 500099, "NYY", 1, "P", 770001, 0.82, [], False, 7999, "2027-07-15T23:05:00Z"),
                           None, None)
            sch_mod._refuse_late_delivery(dp, cfg, picks_dir, st, "2027-07-15", "lineup", RealDatetime.fromisoformat("2027-07-15T19:00:00-04:00"))
            assert len(cap.dms) == 1
            clock.set_et("2027-07-15T23:00:00-04:00")
            run_all_checks(picks_dir=picks_dir, models_dir=workdir / "data" / "models",
                           today=RealDatetime(2027, 7, 15).date(), data_dir=workdir / "data" / "processed",
                           dm_recipient="stonehengee.bsky.social", live_forward_capture_enabled=False, contest_state_expected=True)
            assert len(cap.dms) == 1

    h_pv = run_worker_subprocess(script, "head", "partial_void_double", ["default", "bts_contest_streak_auto_v2"])
    b_pv = run_worker_subprocess(script, "base", "partial_void_double", ["horizon", "bts_contest_streak_auto_v1"])
    assert h_pv["streak_json"]["streak"] == 13 == b_pv["streak_json"]["streak"]
    return out


def test_item1_doubleheader_and_game_unverified() -> dict:
    import bts.day_outcome as do_mod

    script = Path(__file__).resolve()
    for date_str in ("2027-07-15", "2027-09-20"):
        h_locked = run_worker_subprocess(script, "head", "dh_locked", ["default", "bts_contest_streak_auto_v2", date_str])
        h_recheck = run_worker_subprocess(script, "head", "dh_unlocked_recheck", ["default", "bts_contest_streak_auto_v2", date_str])
        assert {g["game_pk"]: g["is_doubleheader_game2"] for g in h_locked["scheduler_state"]["games"]} == {8001: False, 8002: True}
        assert "DH game 2 (8002): re-checking" not in h_locked["stderr"]
        assert "DH game 2 (8002): lineup confirmed." in h_recheck["stderr"]

    res = do_mod.classify_day(
        date="2027-07-15", games=2,
        decision={"action": "single", "source": "mdp", "scoreable": True, "delivery_status": "delivered",
                  "primary": {"batter_name": "Aaron Judge", "batter_id": 500002, "game_pk": 8002}},
        decision_sha256="abc",
        receipts=[{
            "outcome": "observed", "et_date": "2027-07-15", "season": 2027, "degraded": [],
            "account": {"username": "stonehengee"}, "selection": {"commit": {"decision_sha256": "abc"}},
            "observation": {"before_cutoff": True, "response_completed_at": "2027-07-15T22:00:00Z"},
            "verifier": {"ok": True, "reason": "match"},
            "qualification": {"all_confirmed": False, "slots": [{"role": "pick", "state": "unit_unverified"}]},
        }],
        intent="enter", expected_username="stonehengee",
    )
    assert res.state == "entered_game_unverified" and res.ok is True
    return res.as_dict()


def test_item1_void_grading_l01() -> dict:
    script = Path(__file__).resolve()
    out = {}
    for date_str in ("2027-07-15", "2027-09-20"):
        out[date_str] = {}
        for scen in ("void_rainout", "void_walk_only", "void_suspended_pre0"):
            h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2", date_str])
            b = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1", date_str])
            assert h["pick_json"]["result"] == "void" and h["streak_json"]["streak"] == 20
            if scen == "void_rainout":
                assert b["pick_json"]["result"] == "void" and b["streak_json"]["streak"] == 20
            else:
                assert b["pick_json"]["result"] == "miss" and b["streak_json"]["streak"] == 0
            out[date_str][scen] = {"head": h["pick_json"]["result"], "base": b["pick_json"]["result"]}
    return out


# ===========================================================================
# ITEM 2 TESTS
# ===========================================================================

def _seed_end_of_2026_snapshot(workdir: Path) -> Path:
    stage_models(workdir)
    picks_dir = workdir / "data" / "picks"
    acc_dir = picks_dir / "account_state"
    health_dir = workdir / "data" / "health_state"
    acc_dir.mkdir(parents=True, exist_ok=True)
    health_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (picks_dir / "streak.json").write_text('{"streak":14,"saver_available":false,"updated":"2026-09-27T23:45:00Z"}\n')
    (acc_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 14, "best_streak": 27,
        "source": "mlb_bts_profile", "source_date": "2026-09-27", "recorded_at": "2026-10-20T17:30:00Z",
        "user_id": 12345, "username": "stonehengee", "saver_available": None,
    }, indent=2) + "\n")
    (acc_dir / "saver_state.json").write_text('{"season":2026,"state":"used","updated_at":"2026-09-12T04:00:00Z","source":"cli"}\n')
    (picks_dir / "2026-09-27.json").write_text(json.dumps({
        "date": "2026-09-27", "run_time": "2026-09-27T18:05:00-04:00",
        "pick": {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "SD", "lineup_position": 1,
                 "pitcher_name": "P", "pitcher_id": 770001, "p_game_hit": 0.825, "flags": [],
                 "projected_lineup": False, "game_pk": 660927, "game_time": "2026-09-27T19:10:00Z"},
        "double_down": None, "runner_up": None, "notification_sent": True,
        "notification_channel": "bluesky_dm", "notification_id": "dm-2026",
        "delivered_at": "2026-09-27T14:15:00-04:00", "result": "hit", "slot_results": {"pick": "hit"},
    }, indent=2) + "\n")
    return picks_dir


def test_item2_first_start_old_config() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod

    with isolated_tmp_root(prefix="snap_old_offday_") as workdir:
        _seed_end_of_2026_snapshot(workdir)
        clock = VirtualClock("2027-03-20T10:00:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-03-20": [], "2027-03-21": []})
        cfg_path = write_toml_config(workdir / "orch.toml", base_live_config(
            workdir, pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        ))
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-03-20", "2027-03-21")) as cap:
            res_off = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-20", "--config", str(cfg_path)])
            assert res_off.exit_code == 0 and cap.dms == [] and "READY=1" in cap.sd_events

    with isolated_tmp_root(prefix="snap_old_gameday_") as workdir:
        picks_dir = _seed_end_of_2026_snapshot(workdir)
        clock = VirtualClock("2027-03-26T10:00:00-04:00")
        world, rows = _single_game_world(clock, "2027-03-26", "2027-03-27", p_hit=0.822)
        cfg_path = write_toml_config(workdir / "orch.toml", base_live_config(
            workdir, pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        ))
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-03-26", "2027-03-27")) as cap:
            res_game = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-26", "--config", str(cfg_path)])
            assert res_game.exit_code == 0
            pick = json.loads((picks_dir / "2027-03-26.json").read_text())
            assert pick["chosen_on"]["streak"] == 14 and pick["chosen_on"]["kind"] == "automatic"
            assert len(cap.dms) == 1 and "- [pick_delivery] pick locked for 2027-03-26" in cap.dms[0]["text"]
            eod_dm = cap.dms[0]["text"]

    with isolated_tmp_root(prefix="snap_old_skip_leak_") as workdir:
        picks_dir = _seed_end_of_2026_snapshot(workdir)
        clock = VirtualClock("2027-03-26T10:00:00-04:00")
        world, rows_785 = _single_game_world(clock, "2027-03-26", "2027-03-27", p_hit=0.785)
        cfg_path = write_toml_config(workdir / "orch.toml", base_live_config(
            workdir, pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        ))
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows_785), ("2027-03-26", "2027-03-27")):
            CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-26", "--config", str(cfg_path)])
            dec_785 = json.loads((picks_dir / "2027-03-26" / "decision.json").read_text())
            assert dec_785["action"] == "skip" and dec_785["streak"] == 14

    return {"chosen_on": pick["chosen_on"], "p785_action": dec_785["action"], "false_eod_dm": eod_dm}


def test_item2_half_edited_dm_no_cs_flag() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod

    spec = importlib.util.spec_from_file_location("ch_mod", str(REPO_ROOT / "scripts" / "check_heartbeat.py"))
    ch_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ch_mod)

    out = {}
    for label, entry_intent in (("dm_enter_no_cs_flag", "enter"), ("dm_research_mismatch", "research")):
        with isolated_tmp_root(prefix=f"half1_{label}_") as workdir:
            _seed_end_of_2026_snapshot(workdir)
            clock = VirtualClock("2027-03-26T10:00:00-04:00")
            world, rows = _single_game_world(clock, "2027-03-26", "2027-03-27")
            cfg_path = write_toml_config(workdir / "orch.toml", base_live_config(
                workdir, pick_delivery="dm", entry_intent=entry_intent,
                contest_state_required=False, contest_state_expected=False,
            ))
            samples: list[dict] = []
            _, _, samples = ch_mod.assess_churn(samples, 0, clock.now_utc())
            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-03-26", "2027-03-27")) as cap:
                codes = []
                for cycle in range(1, 4):
                    clock.advance(30.0)
                    r = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-26", "--config", str(cfg_path)])
                    codes.append(r.exit_code)
                    churn, reason, samples = ch_mod.assess_churn(samples, cycle, clock.now_utc())
                assert codes == [2, 2, 2] and cap.dms == [] and cap.sd_events == [] and churn is True
            out[label] = {"exit_codes": codes, "dms": 0, "churn_reason": reason}
    return out


def test_item2_half_edited_required_no_switch() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod

    out = {}
    for with_preview in (False, True):
        label = "with_3am_preview" if with_preview else "no_preview"
        with isolated_tmp_root(prefix=f"half2_{label}_") as workdir:
            picks_dir = _seed_end_of_2026_snapshot(workdir)
            clock = VirtualClock("2027-07-15T03:05:00-04:00" if with_preview else "2027-07-15T10:00:00-04:00")
            world, rows = _single_game_world(clock, "2027-07-15", "2027-07-16")
            cfg_path = write_toml_config(workdir / "orch.toml", base_live_config(
                workdir, pick_delivery="dm", entry_intent="enter",
                contest_state_required=True, contest_state_expected=True,
            ))
            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-07-15", "2027-07-16")) as cap:
                runner = CliRunner()
                if with_preview:
                    fake_pred_mod = types.ModuleType("bts.model.predict")
                    fake_pred_mod.run_pipeline = lambda *a, **k: pd.DataFrame([dict(rows[0], flags="PROJECTED")])
                    fake_pred_mod.load_blend = lambda *a, **k: None
                    old_mod = sys.modules.get("bts.model.predict")
                    sys.modules["bts.model.predict"] = fake_pred_mod
                    try:
                        rp = runner.invoke(cli_mod.cli, [
                            "preview", "--date", "2027-07-15", "--picks-dir", str(picks_dir),
                            "--data-dir", str(workdir / "data" / "processed"), "--models-dir", str(workdir / "data" / "models"),
                        ])
                        assert rp.exit_code == 0, rp.output
                    finally:
                        if old_mod is None:
                            sys.modules.pop("bts.model.predict", None)
                        else:
                            sys.modules["bts.model.predict"] = old_mod
                    clock.set_et("2027-07-15T10:00:00-04:00")

                rs = runner.invoke(cli_mod.cli, ["schedule", "--date", "2027-07-15", "--config", str(cfg_path)])
                assert rs.exit_code == 0 and "READY=1" in cap.sd_events and clock.idle_seconds > 0
                dm_texts = [d["text"] for d in cap.dms]
                if not with_preview:
                    assert len(dm_texts) == 3, dm_texts
                    assert "Morning check for 2027-07-15" in dm_texts[0]
                    assert "season 2027 switch is absent" in dm_texts[1]
                    assert "[saver_state]" in dm_texts[2]
                else:
                    # 4 DMs: morning CRITICAL, check CRITICAL, T-10m `missed_pick` CRITICAL, and EOD `[saver_state]`
                    assert len(dm_texts) == 4, dm_texts
                    assert "- [missed_pick] NO PICK DELIVERED" in dm_texts[2]
                    assert "[saver_state]" in dm_texts[3]
                out[label] = {"dm_count": len(dm_texts), "dms": dm_texts}
    return out


def _stage_head_written_files(workdir: Path, *, locked_pick: bool, include_auto: bool, switch_status: str) -> Path:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    import bts.daily_decision as dd_mod
    import bts.picks as picks_mod
    from bts import entry_receipt, receipt_io

    picks_dir = workdir / "data" / "picks"
    acc_dir = picks_dir / "account_state"
    health_dir = workdir / "data" / "health_state"
    acc_dir.mkdir(parents=True, exist_ok=True)
    health_dir.mkdir(parents=True, exist_ok=True)
    clock = VirtualClock("2027-07-15T14:00:00-04:00")
    world, _ = _single_game_world(clock, "2027-07-15", "2027-07-16")
    with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")):
        runner = CliRunner()
        assert runner.invoke(cli_mod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks_dir)]).exit_code == 0
        act_id = json.loads((acc_dir / "season_contract.json").read_text())["activation_id"]
        (acc_dir / "saver_state.json").write_text('{"season":2027,"state":"active","updated_at":"2027-07-01T12:00:00Z","source":"cli"}\n')
        assert runner.invoke(cli_mod.cli, [
            "set-contest-streak", "--date", "2027-07-15", "--streak", "19", "--best-streak", "25",
            "--saver", "used", "--picks-dir", str(picks_dir),
        ]).exit_code == 0
        if switch_status == "revoked":
            contract = json.loads((acc_dir / "season_contract.json").read_text())
            contract.update(status="revoked", revoked_at="2027-07-15T18:00:00Z", note="test")
            (acc_dir / "season_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        if include_auto:
            (acc_dir / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": act_id,
                "active_streak": 12, "best_streak": 30, "source": "mlb_bts_profile",
                "source_date": "2027-07-14", "recorded_at": "2027-07-15T14:30:00Z",
                "user_id": 12345, "username": "stonehengee", "saver_available": None,
            }, indent=2) + "\n")
        picks_mod.save_streak(12, picks_dir, saver_available=True, applied_date="2027-07-14", season=2027)
        dp = picks_mod.DailyPick(
            date="2027-07-15", run_time="2027-07-15T18:05:00-04:00",
            pick=picks_mod.Pick("Luis Arraez", 500001, "NYY", 1, "Synth Pitcher", 770001,
                                0.822, [] if locked_pick else ["PROJECTED"], not locked_pick,
                                7001, "2027-07-15T23:35:00Z", "BOS"),
            double_down=None, runner_up=None, notification_sent=locked_pick,
            notification_channel="bluesky_dm" if locked_pick else None,
            notification_id="dm-head-001" if locked_pick else None,
            delivered_at="2027-07-15T18:06:00-04:00" if locked_pick else None,
            chosen_on={"schema": "bts_pick_chosen_on_v1", "required": True, "kind": "automatic",
                       "streak": 12, "saver_available": True, "dated": None},
            automatic_reading={"schema": "bts_pick_automatic_reading_v1", "recorded_at": "2027-07-15T14:30:00Z",
                               "age_hours": 3.58, "stale": False, "reason": None},
        )
        picks_mod.save_pick(dp, picks_dir)
        picks_mod.save_shadow_pick(dp, picks_dir)
        if locked_pick:
            dd_mod.write_decision(
                "2027-07-15", picks_dir,
                action="single", source="mdp", scoreable=True, delivery_status="delivered",
                primary={"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": 7001, "p_game_hit": 0.822},
                double_down=None, streak=12, saver_available=True, state_source="contest_auto",
                state_status="fresh", allow_double=True, contest_source_date="2027-07-14",
            )
        receipt_io.publish(
            entry_receipt.receipts_root(picks_dir) / "2027-07-15" / "20270715T181500-aaa.json",
            b'{"schema":"bts_pick_entry_receipt_v1","outcome":"no_pick_file"}\n',
        )
    return picks_dir


def test_item2_rollback_to_basesha() -> dict:
    import bts.picks as head_picks_mod

    script = Path(__file__).resolve()
    with isolated_tmp_root(prefix="rb_locked_") as workdir:
        picks_dir = _stage_head_written_files(workdir, locked_pick=True, include_auto=True, switch_status="revoked")
        r_auto = run_worker_subprocess(script, "base", "rb_inspect_with_auto", [str(workdir)])
        assert r_auto["decision_streak_state"]["streak"] == 12
        assert r_auto["decision_streak_state"]["saver_available"] is True
        assert r_auto["decision_streak_state"]["best_status"] == "untrusted"
        assert "chosen_on" not in r_auto["pick_keys_after_basesha_save"]
        assert "automatic_reading" not in r_auto["pick_keys_after_basesha_save"]
        assert "season" not in r_auto["streak_after_basesha_save"]
        assert head_picks_mod.load_streak(picks_dir, season=2027) == 0

    with isolated_tmp_root(prefix="rb_v3_only_") as workdir:
        _stage_head_written_files(workdir, locked_pick=True, include_auto=False, switch_status="revoked")
        r_v3 = run_worker_subprocess(script, "base", "rb_v3_without_auto", [str(workdir)])
        assert r_v3["day1"]["streak"] == 19 and r_v3["day1"]["saver_available"] is True
        assert r_v3["day1"]["best_status"] == "untrusted"
        assert any(a["level"] == "CRITICAL" and "no source_date" in a["message"] for a in r_v3["day1"]["alerts"])
        assert r_v3["day6_still_active"]["streak"] == 19

    with isolated_tmp_root(prefix="rb_unlocked_") as workdir:
        _stage_head_written_files(workdir, locked_pick=False, include_auto=True, switch_status="automatic")
        r_un = run_worker_subprocess(script, "base", "rb_unlocked_day", [str(workdir)])
        assert r_un["decision"]["action"] == "skip" and any("No pick yet" in m for m in r_un["dms"])

    return {"with_auto": r_auto["decision_streak_state"], "v3_only": r_v3, "unlocked": r_un["decision"]["action"]}


# ===========================================================================
# ITEM 3 & 4 TESTS
# ===========================================================================

def _setup_cron_shim_home(workdir: Path, cfg: dict) -> tuple[Path, Path, dict[str, str]]:
    home = workdir / "home"
    bts_dir = home / "projects" / "bts"
    uv_dir = home / ".local" / "bin"
    shim_bin = workdir / "shim_bin"
    crontab_store = workdir / "crontab_store.txt"
    bts_dir.mkdir(parents=True, exist_ok=True)
    uv_dir.mkdir(parents=True, exist_ok=True)
    shim_bin.mkdir(parents=True, exist_ok=True)
    (bts_dir / ".env").write_text("HEALTHCHECKS_PING_URL=https://hc-ping.com/00000000-0000-0000-0000-000000000000\n")
    write_toml_config(home / ".bts-orchestrator.toml", cfg)

    uv_script = uv_dir / "uv"
    uv_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [ "${{1:-}}" = "run" ]; then
    shift
    if [ "${{1:-}}" = "bts" ]; then
        shift
        exec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"
    elif [ "${{1:-}}" = "python" ]; then
        shift
        exec "{sys.executable}" "$@"
    fi
fi
exit 2
""",
        encoding="utf-8",
    )
    uv_script.chmod(uv_script.stat().st_mode | stat.S_IXUSR)

    crontab_script = shim_bin / "crontab"
    crontab_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
STORE="{crontab_store}"
if [ "${{1:-}}" = "-l" ]; then
    [ -f "$STORE" ] && {{ cat "$STORE"; exit 0; }}
    echo "no crontab for synthuser" >&2
    exit 1
elif [ "${{1:-}}" = "-" ]; then
    cat > "$STORE"
    exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    crontab_script.chmod(crontab_script.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env.update(
        HOME=str(home),
        PYTHONPATH=f"{HEAD_SRC}:{REPO_ROOT}",
        PATH=f"{shim_bin}:{uv_dir}:{env.get('PATH', '')}",
        HEALTHCHECKS_PING_URL="https://hc-ping.com/00000000-0000-0000-0000-000000000000",
        TMPDIR=str(workdir),
    )
    return home, crontab_store, env


def _run_cron_setup(env: dict[str, str], action: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "cron-setup-hetzner.sh"), action],
        env=env, capture_output=True, text=True, timeout=20,
    )


def _cron_job_lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln.endswith("# BTS-HETZNER") and not ln.startswith("#")]


def test_item3_cron_show_and_install() -> dict:
    with isolated_tmp_root(prefix="cron_intents_") as workdir:
        _, store_res, env_res = _setup_cron_shim_home(
            workdir / "res",
            base_live_config(workdir / "res", pick_delivery="private", entry_intent="research",
                             contest_state_required=False, contest_state_expected=False),
        )
        store_res.write_text("15 4 * * * /usr/local/bin/unrelated-job\n")
        show_res = _run_cron_setup(env_res, "show")
        assert show_res.returncode == 0
        res_lines = _cron_job_lines(show_res.stdout)
        assert len(res_lines) == 18
        assert not any("check-pick-entered" in ln for ln in res_lines)
        assert all("--dm-recipient" not in ln for ln in res_lines if "fetch-contest-streak" in ln)
        assert any("day-outcome" in ln and "--dm-recipient stonehengee.bsky.social" in ln for ln in res_lines)
        assert _run_cron_setup(env_res, "install").returncode == 0
        assert "/usr/local/bin/unrelated-job" in store_res.read_text()

        _, store_ent, env_ent = _setup_cron_shim_home(
            workdir / "ent",
            base_live_config(workdir / "ent", pick_delivery="dm", entry_intent="enter",
                             contest_state_required=True, contest_state_expected=True),
        )
        show_ent = _run_cron_setup(env_ent, "show")
        assert show_ent.returncode == 0
        ent_lines = _cron_job_lines(show_ent.stdout)
        assert len(ent_lines) == 19
        assert sum("fetch-contest-streak" in ln and "--dm-recipient stonehengee.bsky.social" in ln for ln in ent_lines) == 4
        assert sum("check-pick-entered" in ln and "--dm-recipient stonehengee.bsky.social" in ln for ln in ent_lines) == 1
        assert _run_cron_setup(env_ent, "install").returncode == 0

        _, store_bad, env_bad = _setup_cron_shim_home(
            workdir / "bad",
            base_live_config(workdir / "bad", pick_delivery="dm", entry_intent="research",
                             contest_state_required=True, contest_state_expected=True),
        )
        store_bad.write_text("0 0 * * * /bin/true\n")
        assert _run_cron_setup(env_bad, "install").returncode != 0
        assert store_bad.read_text() == "0 0 * * * /bin/true\n"
    return {"research_jobs": len(res_lines), "enter_jobs": len(ent_lines)}


def test_item3_cron_cli_commands_verified() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod

    substitutions = {
        r"$(date -d yesterday +\%Y-\%m-\%d)": "2027-07-14",
        r'$(date -d "7 days ago" +\%Y-\%m-\%d)': "2027-07-08",
        r"$(date +\%Y-\%m-\%d)": "2027-07-15",
        r"$(date +\%Y)": "2027",
        r'"$BTS_SCHEDULER_HEARTBEAT_PING_URL"': "https://hc-ping.com/hb-uuid",
    }
    with isolated_tmp_root(prefix="cron_cmds_") as workdir:
        cfg = base_live_config(workdir, pick_delivery="dm", entry_intent="enter")
        orch_path = write_toml_config(workdir / "orch.toml", cfg)
        _, _, env = _setup_cron_shim_home(workdir / "shim", cfg)
        ent_lines = _cron_job_lines(_run_cron_setup(env, "show").stdout)

        unique_invocations: list[list[str]] = []
        for ln in ent_lines:
            s = ln
            for k, v in substitutions.items():
                s = s.replace(k, v)
            for m in re.finditer(r"\.local/bin/uv run ([^>&;)#]+)", s):
                tokens = [str(orch_path) if t.endswith(".bts-orchestrator.toml") else t
                          for t in shlex.split(m.group(1).strip())]
                if tokens not in unique_invocations:
                    unique_invocations.append(tokens)
        assert len(unique_invocations) >= 14

        runner = CliRunner()
        verified: list[str] = []
        for tokens in unique_invocations:
            if tokens[0] == "python":
                proc = subprocess.run([sys.executable, str(REPO_ROOT / tokens[1]), "--help"],
                                      capture_output=True, text=True, timeout=10)
                assert proc.returncode == 0
                verified.append(" ".join(tokens))
                continue
            bts_args = tokens[1:]
            cmd_path, grp, idx = [], cli_mod.cli, 0
            while idx < len(bts_args) and not bts_args[idx].startswith("-"):
                sub = grp.get_command(None, bts_args[idx])  # type: ignore[arg-type]
                if sub is None:
                    break
                cmd_path.append(bts_args[idx])
                grp = sub
                idx += 1
            assert runner.invoke(cli_mod.cli, [*cmd_path, "--help"]).exit_code == 0
            parent_ctx = cli_mod.cli.make_context("bts", list(bts_args))
            cur_cmd = cli_mod.cli
            rem_args = list(parent_ctx.protected_args + parent_ctx.args)
            while isinstance(cur_cmd, cli_mod.click.Group) and rem_args:
                cname, cobj, rem = cur_cmd.resolve_command(parent_ctx, rem_args)
                assert cobj is not None
                parent_ctx = cobj.make_context(cname, rem, parent=parent_ctx)
                cur_cmd = cobj
                rem_args = list(parent_ctx.protected_args + parent_ctx.args)
            assert rem_args == []
            verified.append("bts " + " ".join(bts_args))

        clock = VirtualClock("2027-07-15T08:30:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-07-14": [], "2027-07-15": []})
        write_covered_auto_state(workdir / "data" / "picks", season=2027, streak=5, best_streak=5,
                                 source_date="2027-07-14", recorded_at_utc="2027-07-15T06:10:00Z")
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-14", "2027-07-15")):
            for bts_args in (
                ["check-results", "--date", "2027-07-14", "--wait-deadline-et", "06:00"],
                ["reconcile"],
                ["skip-policy-shadow-update"],
                ["day-outcome", "--date", "2027-07-14", "--config", str(orch_path),
                 "--picks-dir", "data/picks", "--expected-username", "stonehengee",
                 "--dm-recipient", "stonehengee.bsky.social"],
            ):
                assert runner.invoke(cli_mod.cli, bts_args).exit_code == 0
    return {"verified_count": len(verified), "verified_commands": verified}


def test_item3_cron_timing_and_heartbeat() -> dict:
    import bts.heartbeat as hb_mod
    from bts.contest_state import AUTOMATIC_READING_STALE_HOURS

    spec = importlib.util.spec_from_file_location("ch_mod", str(REPO_ROOT / "scripts" / "check_heartbeat.py"))
    ch_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ch_mod)

    with isolated_tmp_root(prefix="hb_window_") as workdir:
        hb_path = workdir / ".heartbeat"
        wake_utc = RealDatetime.fromisoformat("2027-07-15T14:00:00+00:00")
        hb_mod.write_heartbeat(hb_path, state=hb_mod.HeartbeatState.SLEEPING,
                               now_utc=RealDatetime.fromisoformat("2027-07-15T02:30:00+00:00"),
                               sleeping_until=wake_utc)
        assert ch_mod.is_stale(hb_path, now=wake_utc)[0] is False
        assert ch_mod.is_stale(hb_path, now=RealDatetime.fromisoformat("2027-07-15T14:00:25+00:00"))[0] is False
        samples: list[dict] = []
        _, _, samples = ch_mod.assess_churn(samples, 5, RealDatetime.fromisoformat("2027-07-15T13:55:00+00:00"))
        assert ch_mod.assess_churn(samples, 6, RealDatetime.fromisoformat("2027-07-15T14:05:00+00:00"))[0] is False

    age_after_0210 = round((RealDatetime.fromisoformat("2027-07-15T10:00:00-04:00")
                            - RealDatetime.fromisoformat("2027-07-15T02:10:00-04:00")).total_seconds() / 3600, 2)
    age_if_overnight_fetches_fail = round((RealDatetime.fromisoformat("2027-07-15T10:00:00-04:00")
                                           - RealDatetime.fromisoformat("2027-07-14T13:30:00-04:00")).total_seconds() / 3600, 2)
    assert age_after_0210 < AUTOMATIC_READING_STALE_HOURS < age_if_overnight_fetches_fail
    return {
        "age_at_1000_after_0210": age_after_0210,
        "age_at_1000_if_overnight_fetches_fail": age_if_overnight_fetches_fail,
    }


def test_item3_enter_cron_dm_matrix() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    import bts.daily_decision as dd_mod
    import bts.picks as picks_mod
    from bts.leaderboard.auth import AuthError, AuthSession

    with isolated_tmp_root(prefix="cron_dm_matrix_") as workdir:
        picks_dir = workdir / "data" / "picks"
        clock = VirtualClock("2027-07-15T18:15:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-07-15": [GameSpec(game_pk=7001, official_date="2027-07-15", game_date_utc="2027-07-15T23:05:00Z")]})
        write_covered_auto_state(picks_dir, season=2027, streak=12, best_streak=15, source_date="2027-07-14", recorded_at_utc="2027-07-15T14:30:00Z")
        dp = picks_mod.DailyPick(
            date="2027-07-15", run_time="2027-07-15T18:05:00-04:00",
            pick=picks_mod.Pick("Luis Arraez", 500001, "NYY", 1, "P", 770001, 0.822, [], False, 7001, "2027-07-15T23:05:00Z"),
            double_down=None, runner_up=None, notification_sent=True, notification_channel="bluesky_dm",
            notification_id="dm-1", delivered_at="2027-07-15T18:05:00-04:00",
        )
        picks_mod.save_pick(dp, picks_dir)
        dd_mod.write_decision(
            "2027-07-15", picks_dir,
            action="single", source="mdp", scoreable=True, delivery_status="delivered",
            primary={"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": 7001, "p_game_hit": 0.822},
            double_down=None, streak=12, saver_available=True, state_source="contest_auto",
            state_status="fresh", allow_double=True, contest_source_date="2027-07-14",
        )
        runner = CliRunner()
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")) as cap:
            with patch("bts.leaderboard.auth.load_session_cookies", side_effect=AuthError("expired")):
                assert runner.invoke(cli_mod.cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir), "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social"]).exit_code == 2
                assert runner.invoke(cli_mod.cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir), "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social"]).exit_code == 2
            assert len(cap.dms) == 1

            with patch("bts.leaderboard.auth.load_session_cookies", side_effect=AuthError("flap")):
                assert runner.invoke(cli_mod.cli, ["check-pick-entered", "--picks-dir", str(picks_dir), "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social", "--now-et", "2027-07-15T18:15:00"]).exit_code == 0
            assert len(cap.dms) == 1

            fake_session = AuthSession(xsid="xs", user_id=12345, username="stonehengee")
            today_date = RealDatetime(2027, 7, 15).date()
            with patch("bts.leaderboard.auth.load_session_cookies", return_value={"ipid": "1"}), \
                 patch("bts.leaderboard.auth.extract_uid", return_value="u1"), \
                 patch("bts.leaderboard.auth.fetch_login_session", return_value=fake_session), \
                 patch("bts.contest_fetch.fetch_profile", return_value={"predictions": []}), \
                 patch("bts.cli._fetch_rounds", return_value={101: today_date}), \
                 patch("bts.cli._fetch_bts_to_mlb", return_value={999: 500001}):
                with patch("bts.contest_fetch.fetch_pending_predictions", return_value=[]):
                    for ts in ("2027-07-15T18:15:00", "2027-07-15T18:35:00", "2027-07-15T18:48:00"):
                        assert runner.invoke(cli_mod.cli, ["check-pick-entered", "--picks-dir", str(picks_dir), "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social", "--now-et", ts]).exit_code == 1
                with patch("bts.contest_fetch.fetch_pending_predictions", return_value=[{"roundId": 101, "playerId": 999, "unitId": 1, "number": 1}]):
                    assert runner.invoke(cli_mod.cli, ["check-pick-entered", "--picks-dir", str(picks_dir), "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social", "--now-et", "2027-07-15T18:52:00"]).exit_code == 0
            assert len(cap.dms) == 5

    for mod_path in (
        HEAD_SRC / "bts" / "data" / "pull.py",
        HEAD_SRC / "bts" / "data" / "build.py",
        HEAD_SRC / "bts" / "data" / "sync.py",
        HEAD_SRC / "bts" / "data" / "backup.py",
        HEAD_SRC / "bts" / "data" / "backup_cli.py",
        HEAD_SRC / "bts" / "features" / "park_drag_producer.py",
        HEAD_SRC / "bts" / "skip_policy_shadow.py",
        REPO_ROOT / "scripts" / "check_heartbeat.py",
    ):
        for node in ast.walk(ast.parse(mod_path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and "dm" in node.module.split("."):
                raise AssertionError(f"Unexpected DM import in {mod_path}")
    return {"dms_exercised": [d["text"] for d in cap.dms]}


def test_item4_leaderboard_systemd_drift_blindspot() -> dict:
    import bts.health.unit_drift as ud_mod

    head_lb = (REPO_ROOT / "deploy" / "systemd" / "bts-leaderboard.service").read_text()
    stale_snapshot_lb = head_lb.replace("stonehengee.bsky.social", "stoneericm.bsky.social")
    with isolated_tmp_root(prefix="unit_drift_lb_") as workdir:
        home = workdir / "home"
        installed_dir = home / ".config" / "systemd" / "user"
        installed_dir.mkdir(parents=True, exist_ok=True)
        for name in ("bts-scheduler.service", "bts-dashboard.service"):
            (installed_dir / name).write_text((REPO_ROOT / "scripts" / "systemd" / name).read_text())
        (installed_dir / "bts-leaderboard.service").write_text(stale_snapshot_lb)
        (installed_dir / "bts-leaderboard.timer").write_text((REPO_ROOT / "deploy" / "systemd" / "bts-leaderboard.timer").read_text())

        alerts = ud_mod.check(installed_dir=installed_dir, repo_units_dir=REPO_ROOT / "scripts" / "systemd")
        assert alerts == []
        env = dict(os.environ, HOME=str(home), TMPDIR=str(workdir))
        proc = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "install-systemd-hetzner.sh"), "--diff"],
                              env=env, capture_output=True, text=True, timeout=10)
        assert proc.returncode == 0 and "bts-leaderboard" not in proc.stdout
    return {"unit_drift_alerts": 0, "persistent_timer": True}


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        res = _worker_entry(sys.argv[2], sys.argv[3], sys.argv[4:])
        sys.stdout.write(json.dumps(res))
        return 0

    logs_dir = REVIEW_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "item1": {
            "single_hit_and_miss": test_item1_single_hit_and_miss(),
            "stale_and_multiday_saver_trap": test_item1_stale_and_multiday_saver_trap(),
            "double_down_and_day_outcome": test_item1_double_down_and_day_outcome(),
            "skip_days": test_item1_skip_days(),
            "postponed_and_duplicate_eod_alert": test_item1_postponed_and_duplicate_eod_alert(),
            "doubleheader": test_item1_doubleheader_and_game_unverified(),
            "void_grading_l01": test_item1_void_grading_l01(),
        },
        "item2": {
            "first_start_old_config": test_item2_first_start_old_config(),
            "half_edited_dm_no_cs_flag": test_item2_half_edited_dm_no_cs_flag(),
            "half_edited_required_no_switch": test_item2_half_edited_required_no_switch(),
            "rollback_to_basesha": test_item2_rollback_to_basesha(),
        },
        "item3_and_4": {
            "cron_show_and_install": test_item3_cron_show_and_install(),
            "cron_cli_commands_verified": test_item3_cron_cli_commands_verified(),
            "cron_timing_and_heartbeat": test_item3_cron_timing_and_heartbeat(),
            "enter_cron_dm_matrix": test_item3_enter_cron_dm_matrix(),
            "leaderboard_systemd_drift_blindspot": test_item4_leaderboard_systemd_drift_blindspot(),
        },
    }
    out_path = logs_dir / "slice2_verified_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"ALL 16 SLICE 2 CHECKS PASSED -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
