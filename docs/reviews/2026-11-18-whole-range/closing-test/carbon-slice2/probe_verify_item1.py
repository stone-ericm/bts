"""Verified Item 1 whole-range probes: mid-season (2027-07-15) and late-season
(2027-09-20) automatic-path comparison between BASESHA (619abda7) and HEAD (bc143131).

Host-independent (stubs `shutil.disk_usage` and stages synthetic `blend_<date>.pkl`
files so EOD health checks do not depend on developer laptop disk fullness).
Adoptable as pytest regression tests: no module-level import of `bts.model.predict`,
no process-wide state left changed, no network, no writes outside
`.carbon-review/whole-range/`.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from harness_lib import (  # noqa: E402
    BASE_SRC,
    HEAD_SRC,
    REVIEW_DIR,
    GameSpec,
    RealDatetime,
    SyntheticMLBWorld,
    VirtualClock,
    base_live_config,
    harness_env,
    isolated_tmp_root,
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
    """Wrap `harness_env` with synthetic `blend_<date>.pkl` files and deterministic
    30% `shutil.disk_usage` so EOD `run_all_checks` is 100% host-independent."""
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


def _worker_run_scenario(code_tree: str, scenario: str, extra: list[str]) -> dict:
    import bts
    import bts.scheduler as sch_mod

    expected_src = (HEAD_SRC if code_tree == "head" else BASE_SRC).resolve()
    actual_bts = Path(bts.__file__).resolve()
    assert actual_bts.is_relative_to(expected_src), f"Expected {expected_src}, imported {actual_bts}"

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

                clock = VirtualClock(f"{date_str}T10:00:00-04:00")
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
                    workdir / "data" / "picks",
                    season=2027, streak=streak, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T06:10:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": 7001, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p1},
                    {"batter_name": "Yordan Alvarez", "batter_id": 500002, "team": "HOU", "game_pk": 7002, "game_time": f"{date_str}T23:40:00Z", "p_game_hit": p2},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
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
                    workdir / "data" / "picks",
                    season=2027, streak=12, best_streak=15,
                    source_date="2027-07-14", recorded_at_utc="2027-07-14T18:00:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": 7001, "game_time": "2027-07-15T23:05:00Z", "p_game_hit": 0.818},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("double_mid", "double_late"):
                is_late = scenario == "double_late"
                date_str = "2027-09-20" if is_late else "2027-07-15"
                prev_date = "2027-09-19" if is_late else "2027-07-14"
                clock = VirtualClock(f"{date_str}T10:00:00-04:00")
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
                    workdir / "data" / "picks",
                    season=2027, streak=12, best_streak=15,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T06:10:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Bobby Witt Jr.", "batter_id": 500001, "team": "NYY", "game_pk": 7101, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": 0.835},
                    {"batter_name": "Freddie Freeman", "batter_id": 500002, "team": "LAD", "game_pk": 7102, "game_time": f"{date_str}T23:25:00Z", "p_game_hit": 0.831},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("skip_mid", "skip_late_stop"):
                is_late = scenario == "skip_late_stop"
                date_str = "2027-09-20" if is_late else "2027-07-15"
                prev_date = "2027-09-19" if is_late else "2027-07-14"
                best = 30 if is_late else 15
                p1 = 0.860 if is_late else 0.780
                clock = VirtualClock(f"{date_str}T10:00:00-04:00")
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
                    picks_dir,
                    season=2027, streak=12, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T06:10:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                (picks_dir / f"{date_str}.json").write_text(json.dumps({
                    "date": date_str,
                    "run_time": f"{date_str}T07:05:00Z",
                    "pick": {
                        "batter_name": "Steven Kwan", "batter_id": 500001, "team": "NYY",
                        "lineup_position": 1, "pitcher_name": "Synth Pitcher", "pitcher_id": 770001,
                        "p_game_hit": 0.815, "flags": ["PROJECTED"], "projected_lineup": True,
                        "game_pk": 7201, "game_time": f"{date_str}T23:05:00Z",
                    },
                    "double_down": None,
                    "runner_up": None,
                }, indent=2))
                rows = [
                    {"batter_name": "Steven Kwan", "batter_id": 500001, "team": "NYY", "game_pk": 7201, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p1},
                    {"batter_name": "Ronald Acuna Jr.", "batter_id": 500002, "team": "ATL", "game_pk": 7202, "game_time": f"{date_str}T23:35:00Z", "p_game_hit": p1 - 0.01},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
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

                clock = VirtualClock(f"{date_str}T10:00:00-04:00")
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
                    workdir / "data" / "picks",
                    season=2027, streak=streak, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T06:10:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )

                def _preds(*a, **k):
                    rows = [
                        {"batter_name": "Corey Seager", "batter_id": 500001, "team": "NYY", "game_pk": 7301, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p_g1},
                        {"batter_name": "Julio Rodriguez", "batter_id": 500002, "team": "SEA", "game_pk": 7302, "game_time": f"{date_str}T23:50:00Z", "p_game_hit": p_g2},
                    ]
                    return _pred_df(world, rows)

                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, _preds, (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario == "partial_void_double":
                date_str = "2027-07-15"
                clock = VirtualClock("2027-07-15T10:00:00-04:00")
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
                    workdir / "data" / "picks",
                    season=2027, streak=12, best_streak=15,
                    source_date="2027-07-14", recorded_at_utc="2027-07-15T06:10:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Bobby Witt Jr.", "batter_id": 500001, "team": "NYY", "game_pk": 7401, "game_time": "2027-07-15T23:05:00Z", "p_game_hit": 0.836},
                    {"batter_name": "Ian Happ", "batter_id": 500002, "team": "CHC", "game_pk": 7402, "game_time": "2027-07-15T23:35:00Z", "p_game_hit": 0.831},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
                    import bts.picks as picks_mod
                    clock.set_et("2027-07-16T02:00:00-04:00")
                    picks_mod.reconcile_results(
                        workdir / "data" / "picks", lookback_days=3, clock=clock.now_et
                    )
                return _collect_run_artifacts(workdir, date_str, cap, buf)

            if scenario in ("dh_locked", "dh_unlocked_recheck"):
                date_str = extra[2] if len(extra) > 2 else "2027-07-15"
                prev_date = "2027-09-19" if date_str == "2027-09-20" else "2027-07-14"
                is_skip = scenario == "dh_unlocked_recheck"
                streak = 16 if date_str == "2027-09-20" else 12
                best = 30 if (is_skip and date_str == "2027-09-20") else (18 if date_str == "2027-09-20" else 15)
                p1 = 0.775 if is_skip else (0.835 if date_str == "2027-09-20" else 0.818)

                clock = VirtualClock(f"{date_str}T10:00:00-04:00")
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
                            double_header="S",
                            away_confirm_at_et=g2_confirm,
                            home_confirm_at_et=g2_confirm,
                            batters=[{"id": 500002, "name": "Aaron Judge", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks",
                    season=2027, streak=streak, best_streak=best,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T06:10:00Z",
                    saver_state="active", auto_schema=auto_schema,
                )
                rows = [
                    {"batter_name": "Rafael Devers", "batter_id": 500001, "team": "BOS", "side": "home", "game_pk": 8001, "game_time": f"{date_str}T17:10:00Z", "p_game_hit": p1},
                    {"batter_name": "Aaron Judge", "batter_id": 500002, "team": "NYY", "side": "away", "game_pk": 8002, "game_time": f"{date_str}T23:10:00Z", "p_game_hit": p1 - 0.005},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
                out = _collect_run_artifacts(workdir, date_str, cap, buf)
                out["sleeps"] = clock.sleeps
                return out

            if scenario in ("void_rainout", "void_walk_only", "void_suspended_pre0"):
                date_str = extra[2] if len(extra) > 2 else "2027-07-15"
                prev_date = "2027-09-19" if date_str == "2027-09-20" else "2027-07-14"
                clock = VirtualClock(f"{date_str}T10:00:00-04:00")
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
                            status_override=status_override,
                            resume_datetime_utc=resume_dt,
                            plays=plays,
                            batters=[{"id": 500001, "name": "Mookie Betts", "side": "away", "batting": batting}],
                        ),
                    ],
                    nxt: nxt_games,
                }
                world = SyntheticMLBWorld(clock, games)
                write_covered_auto_state(
                    workdir / "data" / "picks",
                    season=2027, streak=20, best_streak=22,
                    source_date=prev_date, recorded_at_utc=f"{date_str}T06:10:00Z",
                    saver_state="used", model_streak=20, model_saver=False,
                    auto_schema=auto_schema,
                )
                p1 = 0.835 if date_str == "2027-09-20" else 0.818
                rows = [
                    {"batter_name": "Mookie Betts", "batter_id": 500001, "team": "NYY", "game_pk": 7501, "game_time": f"{date_str}T23:05:00Z", "p_game_hit": p1},
                ]
                cfg = base_live_config(workdir)
                buf = io.StringIO()
                with _host_clean_env(workdir, clock, world, lambda *a, **k: _pred_df(world, rows), (date_str, nxt)) as cap, \
                     contextlib.redirect_stderr(buf):
                    sch_mod.run_day(date=date_str, config=cfg)
                return _collect_run_artifacts(workdir, date_str, cap, buf)

    raise ValueError(f"Unknown scenario: {scenario}")


def test_mid_and_late_single_hit_and_miss() -> dict:
    script = Path(__file__).resolve()
    out = {}
    for scen in ("single_mid_hit", "single_mid_miss", "single_late_hit"):
        h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2"])
        b_lit = run_worker_subprocess(script, "base", scen, ["default", "bts_contest_streak_auto_v2"])
        b_hor = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1"])

        # Unpatched BASESHA skips ALL 2027 dates (`SEASON_END_DATE="2026-09-28"` -> d=0)
        assert b_lit["decision_json"]["action"] == "skip", b_lit["decision_json"]
        assert len(b_lit["dms"]) == 1 and "No pick yet" in b_lit["dms"][0]["text"], b_lit["dms"]

        # Horizon-matched BASESHA sends byte-for-byte identical pick DM as HEAD
        assert len(h["dms"]) == 1 and len(b_hor["dms"]) == 1
        assert h["dms"][0]["text"] == b_hor["dms"][0]["text"]
        assert h["decision_json"]["action"] == "single" == b_hor["decision_json"]["action"]

        head_only_pick_keys = set(h["pick_json"]) - set(b_hor["pick_json"])
        assert head_only_pick_keys == {"chosen_on", "automatic_reading"}, head_only_pick_keys
        assert h["pick_json"]["chosen_on"]["kind"] == "automatic"
        assert h["pick_json"]["automatic_reading"]["stale"] is False

        head_only_streak_keys = set(h["streak_json"]) - set(b_hor["streak_json"])
        assert head_only_streak_keys == {"season", "applied_dates"}, head_only_streak_keys

        if scen == "single_mid_miss":
            assert h["streak_json"]["streak"] == 12 and h["streak_json"]["saver_available"] is False
            assert h["saver_json"]["state"] == "active"
            assert b_hor["streak_json"]["streak"] == 12 and b_hor["streak_json"]["saver_available"] is False
        elif scen == "single_mid_hit":
            assert h["streak_json"]["streak"] == 13 == b_hor["streak_json"]["streak"]
            assert h["decision_json"]["objective"] == "reach57"
        elif scen == "single_late_hit":
            assert h["streak_json"]["streak"] == 17 == b_hor["streak_json"]["streak"]
            assert h["decision_json"]["objective"] == "emax_season_best"

        out[scen] = {
            "head_dm": h["dms"][0]["text"],
            "base_literal_action": b_lit["decision_json"]["action"],
            "base_literal_dm": b_lit["dms"][0]["text"],
            "base_horizon_dm_identical": h["dms"][0]["text"] == b_hor["dms"][0]["text"],
            "head_only_pick_keys": sorted(head_only_pick_keys),
            "head_only_streak_keys": sorted(head_only_streak_keys),
            "head_streak_after": h["streak_json"],
        }
    return out


def test_stale_reading_and_multiday_saver_trap() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    import bts.saver_state as sv_mod
    import bts.scheduler as sch_mod

    script = Path(__file__).resolve()
    h_stale = run_worker_subprocess(script, "head", "stale_mid", ["default", "bts_contest_streak_auto_v2"])
    b_stale = run_worker_subprocess(script, "base", "stale_mid", ["horizon", "bts_contest_streak_auto_v1"])

    assert len(h_stale["dms"]) == 2, h_stale["dms"]
    assert "BTS health WARN attention:\n- [contest_state] Morning check for 2027-07-15" in h_stale["dms"][0]["text"]
    assert "bts set-contest-streak --date 2027-07-15 --streak N --best-streak B --saver S" in h_stale["dms"][0]["text"]
    assert "saver-state" not in h_stale["dms"][0]["text"]
    assert h_stale["pick_json"]["automatic_reading"]["stale"] is True
    assert len(b_stale["dms"]) == 1

    with isolated_tmp_root(prefix="multiday_saver_trap_") as workdir:
        clock = VirtualClock("2027-07-15T10:00:00-04:00")
        games = {
            "2027-07-15": [
                GameSpec(
                    game_pk=7601, official_date="2027-07-15", game_date_utc="2027-07-15T23:05:00Z",
                    batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                ),
            ],
            "2027-07-16": [
                GameSpec(
                    game_pk=7602, official_date="2027-07-16", game_date_utc="2027-07-16T23:05:00Z",
                    batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                ),
            ],
            "2027-07-17": [
                GameSpec(
                    game_pk=7603, official_date="2027-07-17", game_date_utc="2027-07-17T23:05:00Z",
                    batters=[{"id": 500001, "name": "Luis Arraez", "side": "away", "batting": {"atBats": 4, "hits": 1}}],
                ),
            ],
        }
        world = SyntheticMLBWorld(clock, games)
        picks_dir = workdir / "data" / "picks"
        write_covered_auto_state(
            picks_dir,
            season=2027, streak=45, best_streak=45,
            source_date="2027-07-13", recorded_at_utc="2027-07-14T18:00:00Z",
            saver_state="active",
        )
        cfg = base_live_config(workdir)
        buf15 = io.StringIO()
        buf16 = io.StringIO()

        def _preds_day(*a, **k):
            cur = clock.now_et().date().isoformat()
            pk = 7601 if cur == "2027-07-15" else 7602
            return _pred_df(world, [
                {"batter_name": "Luis Arraez", "batter_id": 500001, "team": "NYY", "game_pk": pk, "game_time": f"{cur}T23:05:00Z", "p_game_hit": 0.830},
            ])

        with _host_clean_env(workdir, clock, world, _preds_day, ("2027-07-15", "2027-07-16", "2027-07-17")) as cap:
            clock.set_et("2027-07-15T10:15:00-04:00")
            runner = CliRunner()
            res_cli = runner.invoke(cli_mod.cli, [
                "set-contest-streak",
                "--date", "2027-07-15",
                "--streak", "45",
                "--best-streak", "45",
                "--saver", "used",
                "--picks-dir", str(picks_dir),
            ])
            assert res_cli.exit_code == 0, res_cli.output

            with contextlib.redirect_stderr(buf15):
                sch_mod.run_day(date="2027-07-15", config=cfg)
            dec15 = json.loads((picks_dir / "2027-07-15" / "decision.json").read_text())
            assert dec15["action"] == "skip" and dec15["saver_available"] is False

            clock.set_et("2027-07-16T02:10:00-04:00")
            obs_path = picks_dir / "account_state" / "contest_streak.json"
            obs = json.loads(obs_path.read_text())
            obs.update(active_streak=45, best_streak=45, source_date="2027-07-15", recorded_at="2027-07-16T06:10:00Z")
            obs_path.write_text(json.dumps(obs, indent=2) + "\n")
            sv_mod.maybe_auto_earn_saver(picks_dir, best_streak=45, season=2027)

            dms_before_16 = len(cap.dms)
            clock.set_et("2027-07-16T10:00:00-04:00")
            with contextlib.redirect_stderr(buf16):
                sch_mod.run_day(date="2027-07-16", config=cfg)
            dec16 = json.loads((picks_dir / "2027-07-16" / "decision.json").read_text())
            pick16 = json.loads((picks_dir / "2027-07-16.json").read_text())
            dms_16 = cap.dms[dms_before_16:]

            assert "Morning state check: automatic contest state accepted (streak 45)." in buf16.getvalue()
            assert dec16["action"] == "single" and dec16["saver_available"] is True
            assert pick16["chosen_on"]["saver_available"] is True
            assert len(dms_16) == 1 and "Today's BTS pick: Luis Arraez" in dms_16[0]["text"], dms_16

    return {
        "stale_mid_head_dms": [d["text"] for d in h_stale["dms"]],
        "multiday_day1_action": dec15["action"],
        "multiday_day1_saver": dec15["saver_available"],
        "multiday_day2_morning_stderr": [
            ln.strip() for ln in buf16.getvalue().splitlines() if "Morning state check" in ln
        ],
        "multiday_day2_action": dec16["action"],
        "multiday_day2_saver": dec16["saver_available"],
    }


def test_mid_and_late_double_down_and_day_outcome() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod

    script = Path(__file__).resolve()
    out = {}
    for scen in ("double_mid", "double_late"):
        h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2"])
        b_hor = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1"])

        assert len(h["scheduler_state"]["runs_completed"]) == 2
        assert len(h["dms"]) == 1 and "DOUBLE DOWN" in h["dms"][0]["text"]
        assert h["dms"][0]["text"] == b_hor["dms"][0]["text"]
        assert h["decision_json"]["action"] == "double"
        assert h["pick_json"]["slot_results"] == {"pick": "hit", "double_down": "hit"}
        assert h["streak_json"]["streak"] == 14 == b_hor["streak_json"]["streak"]
        out[scen] = {
            "runs_completed": len(h["scheduler_state"]["runs_completed"]),
            "dm": h["dms"][0]["text"],
            "streak_after": h["streak_json"]["streak"],
            "objective": h["decision_json"]["objective"],
        }

    with isolated_tmp_root(prefix="day_outcome_dd_") as workdir:
        date_str = "2027-07-15"
        clock = VirtualClock("2027-07-16T08:30:00-04:00")
        games = {
            date_str: [
                GameSpec(game_pk=7101, official_date=date_str, game_date_utc="2027-07-15T23:05:00Z"),
                GameSpec(game_pk=7102, official_date=date_str, game_date_utc="2027-07-15T23:25:00Z"),
            ]
        }
        world = SyntheticMLBWorld(clock, games)
        picks_dir = workdir / "data" / "picks"
        (picks_dir / date_str).mkdir(parents=True, exist_ok=True)
        dec_bytes = (json.dumps({
            "schema_version": "bts_daily_decision_v1",
            "date": date_str,
            "action": "double",
            "source": "mdp",
            "scoreable": True,
            "delivery_status": "delivered",
            "primary": {"batter_name": "Bobby Witt Jr.", "batter_id": 500001, "game_pk": 7101},
            "double_down": {"batter_name": "Freddie Freeman", "batter_id": 500002, "game_pk": 7102},
        }, indent=2) + "\n").encode("utf-8")
        (picks_dir / date_str / "decision.json").write_bytes(dec_bytes)
        dec_sha = hashlib.sha256(dec_bytes).hexdigest()

        cfg = base_live_config(workdir)
        cfg_path = write_toml_config(workdir / "orch.toml", cfg)

        from bts import entry_receipt, receipt_io

        def _write_sealed_receipt(attempt_id: str, all_conf: bool, reason: str, slots: list[dict]) -> None:
            rec_dir = entry_receipt.receipts_root(picks_dir) / date_str
            body = {
                "schema": "bts_pick_entry_receipt_v1",
                "attempt_id": attempt_id,
                "et_date": date_str,
                "season": 2027,
                "outcome": "observed",
                "degraded": [],
                "account": {"user_id": 12345, "username": "stonehengee"},
                "selection": {"commit": {"decision_sha256": dec_sha}},
                "observation": {"before_cutoff": True, "response_completed_at": "2027-07-15T22:30:00Z"},
                "verifier": {"ok": all_conf, "reason": reason},
                "qualification": {"all_confirmed": all_conf, "slots": slots},
            }
            data = (json.dumps(body, indent=1, sort_keys=True) + "\n").encode("utf-8")
            receipt_io.publish(rec_dir / f"20270715T223000-{attempt_id}.json", data)

        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")) as cap:
            runner = CliRunner()
            r0 = runner.invoke(cli_mod.cli, [
                "day-outcome", "--date", date_str, "--config", str(cfg_path),
                "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
                "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r0.exit_code == 1 and json.loads(r0.output)["state"] == "entry_unverifiable", r0.output
            assert len(cap.dms) == 1 and "ENTRY UNVERIFIABLE" in cap.dms[0]["text"]

            _write_sealed_receipt("001", False, "mismatch", [
                {"role": "pick", "state": "confirmed"},
                {"role": "double_down", "state": "missing"},
            ])
            r1 = runner.invoke(cli_mod.cli, [
                "day-outcome", "--date", date_str, "--config", str(cfg_path),
                "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
                "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r1.exit_code == 1 and json.loads(r1.output)["state"] == "not_entered", r1.output
            assert len(cap.dms) == 2 and "NOT ENTERED" in cap.dms[1]["text"]

            _write_sealed_receipt("002", True, "match", [
                {"role": "pick", "state": "confirmed"},
                {"role": "double_down", "state": "confirmed"},
            ])
            r2 = runner.invoke(cli_mod.cli, [
                "day-outcome", "--date", date_str, "--config", str(cfg_path),
                "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
                "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r2.exit_code == 0 and json.loads(r2.output)["state"] == "entered", r2.output
            assert len(cap.dms) == 2

        out["day_outcome_double"] = {
            "no_receipts": json.loads(r0.output),
            "one_leg_missing": json.loads(r1.output),
            "both_legs_confirmed": json.loads(r2.output),
        }
    return out


def test_mid_and_late_skip_days() -> dict:
    script = Path(__file__).resolve()
    h_mid = run_worker_subprocess(script, "head", "skip_mid", ["default", "bts_contest_streak_auto_v2"])
    b_mid = run_worker_subprocess(script, "base", "skip_mid", ["horizon", "bts_contest_streak_auto_v1"])

    assert h_mid["decision_json"]["action"] == "skip" == b_mid["decision_json"]["action"]
    assert h_mid["decision_json"]["scoreable"] is False
    assert len(h_mid["dms"]) == 1 and "No pick yet" in h_mid["dms"][0]["text"], h_mid["dms"]
    assert h_mid["dms"][0]["text"] == b_mid["dms"][0]["text"]

    h_late = run_worker_subprocess(script, "head", "skip_late_stop", ["default", "bts_contest_streak_auto_v2"])
    b_late_v1 = run_worker_subprocess(script, "base", "skip_late_stop", ["horizon", "bts_contest_streak_auto_v1"])
    b_late_v2 = run_worker_subprocess(script, "base", "skip_late_stop", ["horizon", "bts_contest_streak_auto_v2"])

    assert h_late["decision_json"]["action"] == "skip"
    assert h_late["decision_json"]["best_status"] == "trusted"
    assert b_late_v1["decision_json"]["action"] == "skip"
    assert b_late_v2["decision_json"]["best_status"] == "untrusted"
    assert b_late_v2["decision_json"]["action"] == "double"

    return {
        "mid_skip_dm": h_mid["dms"][0]["text"],
        "late_head_action": h_late["decision_json"]["action"],
        "late_base_auto_v1_action": b_late_v1["decision_json"]["action"],
        "late_base_auto_v2_best_status": b_late_v2["decision_json"]["best_status"],
        "late_base_auto_v2_action": b_late_v2["decision_json"]["action"],
    }


def test_postponed_games_and_duplicate_eod_alert() -> dict:
    script = Path(__file__).resolve()
    out = {}

    # Pre-selection postponed game excluded cleanly on both dates
    for date_str in ("2027-07-15", "2027-09-20"):
        hp = run_worker_subprocess(script, "head", "postponed_pre", ["default", "bts_contest_streak_auto_v2", date_str])
        bp = run_worker_subprocess(script, "base", "postponed_pre", ["horizon", "bts_contest_streak_auto_v1", date_str])
        assert len(hp["dms"]) == 1 and "Julio Rodriguez" in hp["dms"][0]["text"]
        assert hp["dms"][0]["text"] == bp["dms"][0]["text"]

    # Mid-day postponement between selection and delivery: proves duplicate EOD CRITICAL DM!
    for scen in ("postponed_midday_mid", "postponed_midday_late"):
        h = run_worker_subprocess(script, "head", scen, ["default", "bts_contest_streak_auto_v2"])
        b_hor = run_worker_subprocess(script, "base", scen, ["horizon", "bts_contest_streak_auto_v1"])

        assert len(h["dms"]) == 3, [d["text"] for d in h["dms"]]
        assert "BTS health CRITICAL alert(s):\n- [postponed_pick] DELIVERY REFUSED" in h["dms"][0]["text"]
        assert "Today's BTS pick: Julio Rodriguez" in h["dms"][1]["text"]
        assert "BTS health CRITICAL alert(s):\n- [late_delivery] delivery REFUSED" in h["dms"][2]["text"]
        assert sorted(h["health_dm_status"]["sent_sources"]) == ["CRITICAL:late_delivery", "CRITICAL:postponed_pick"]
        assert h["pick_json"]["result"] == "hit"

        assert len(b_hor["dms"]) == 1 and "Corey Seager" in b_hor["dms"][0]["text"]
        assert b_hor["pick_json"]["result"] == "void"

        out[scen] = {
            "head_dms": [d["text"] for d in h["dms"]],
            "head_sent_sources": h["health_dm_status"]["sent_sources"],
            "head_result": h["pick_json"]["result"],
            "head_streak": h["streak_json"]["streak"],
            "base_horizon_dm": b_hor["dms"][0]["text"],
            "base_horizon_result": b_hor["pick_json"]["result"],
            "base_horizon_streak": b_hor["streak_json"]["streak"],
        }

    # Control proof: `_refuse_late_delivery` (`past_submission_cutoff`) uses `late_delivery` at both sites -> 0 duplicate EOD DMs!
    import bts.scheduler as sch_mod
    from bts.health.runner import run_all_checks
    from bts.picks import DailyPick, Pick
    with isolated_tmp_root(prefix="late_refusal_dedup_ctrl_") as workdir:
        clock = VirtualClock("2027-07-15T19:02:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-07-15": []})
        picks_dir = workdir / "data" / "picks"
        write_covered_auto_state(
            picks_dir, season=2027, streak=12, best_streak=15,
            source_date="2027-07-14", recorded_at_utc="2027-07-15T06:10:00Z",
        )
        cfg = base_live_config(workdir)
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")) as cap:
            st = sch_mod.SchedulerState(
                date="2027-07-15", schedule_fetched_at=clock.now_et().isoformat(),
                games=[], confirmed_game_pks=[], runs_completed=[],
                pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            dp = DailyPick(
                date="2027-07-15", run_time=clock.now_et().isoformat(),
                pick=Pick("Late Batter", 500099, "NYY", 1, "P", 770001, 0.82, [], False, 7999, "2027-07-15T23:05:00Z"),
                double_down=None, runner_up=None,
            )
            cutoff = RealDatetime.fromisoformat("2027-07-15T19:00:00-04:00")
            sch_mod._refuse_late_delivery(dp, cfg, picks_dir, st, "2027-07-15", "lineup", cutoff)
            assert len(cap.dms) == 1 and "- [late_delivery] LATE DELIVERY REFUSED" in cap.dms[0]["text"]
            clock.set_et("2027-07-15T23:00:00-04:00")
            run_all_checks(
                picks_dir=picks_dir,
                models_dir=workdir / "data" / "models",
                today=RealDatetime(2027, 7, 15).date(),
                data_dir=workdir / "data" / "processed",
                dm_recipient="stonehengee.bsky.social",
                live_forward_capture_enabled=False,
                contest_state_expected=True,
            )
            assert len(cap.dms) == 1

    h_pv = run_worker_subprocess(script, "head", "partial_void_double", ["default", "bts_contest_streak_auto_v2"])
    b_pv = run_worker_subprocess(script, "base", "partial_void_double", ["horizon", "bts_contest_streak_auto_v1"])
    assert h_pv["pick_json"]["result"] == "hit" == b_pv["pick_json"]["result"]
    assert h_pv["pick_json"]["slot_results"] == {"pick": "hit", "double_down": "void"}
    assert h_pv["streak_json"]["streak"] == 13 == b_pv["streak_json"]["streak"]
    out["partial_void_double_streak"] = h_pv["streak_json"]["streak"]
    return out


def test_doubleheader_and_game_unverified() -> dict:
    import bts.day_outcome as do_mod

    script = Path(__file__).resolve()
    out = {}
    for date_str in ("2027-07-15", "2027-09-20"):
        h_locked = run_worker_subprocess(
            script, "head", "dh_locked", ["default", "bts_contest_streak_auto_v2", date_str]
        )
        h_recheck = run_worker_subprocess(
            script, "head", "dh_unlocked_recheck", ["default", "bts_contest_streak_auto_v2", date_str]
        )
        games_meta = {g["game_pk"]: g["is_doubleheader_game2"] for g in h_locked["scheduler_state"]["games"]}
        assert games_meta == {8001: False, 8002: True}
        assert "DH game 2 (8002): re-checking" not in h_locked["stderr"]
        assert "DH game 2 (8002): re-checking every 15min" in h_recheck["stderr"]
        assert "DH game 2 (8002): lineup confirmed." in h_recheck["stderr"]
        out[date_str] = {
            "games_is_dh_game2": games_meta,
            "locked_skipped_dh_loop": True,
            "unlocked_confirmed_on_recheck": True,
        }

    dec = {
        "action": "single", "source": "mdp", "scoreable": True, "delivery_status": "delivered",
        "primary": {"batter_name": "Aaron Judge", "batter_id": 500002, "game_pk": 8002},
    }
    rec = {
        "outcome": "observed", "et_date": "2027-07-15", "season": 2027, "degraded": [],
        "account": {"username": "stonehengee"},
        "selection": {"commit": {"decision_sha256": "abc"}},
        "observation": {"before_cutoff": True, "response_completed_at": "2027-07-15T22:00:00Z"},
        "verifier": {"ok": True, "reason": "match"},
        "qualification": {"all_confirmed": False, "slots": [{"role": "pick", "state": "unit_unverified"}]},
    }
    res = do_mod.classify_day(
        date="2027-07-15", games=2, decision=dec, decision_sha256="abc",
        receipts=[rec], intent="enter", expected_username="stonehengee",
    )
    assert res.state == "entered_game_unverified" and res.ok is True
    out["day_outcome_dh_batter_only"] = res.as_dict()
    return out


def test_void_grading_l01() -> dict:
    script = Path(__file__).resolve()
    out = {}
    for date_str in ("2027-07-15", "2027-09-20"):
        out[date_str] = {}
        for scen in ("void_rainout", "void_walk_only", "void_suspended_pre0"):
            h = run_worker_subprocess(
                script, "head", scen, ["default", "bts_contest_streak_auto_v2", date_str]
            )
            b = run_worker_subprocess(
                script, "base", scen, ["horizon", "bts_contest_streak_auto_v1", date_str]
            )
            assert h["pick_json"]["result"] == "void" and h["streak_json"]["streak"] == 20
            if scen == "void_rainout":
                assert b["pick_json"]["result"] == "void" and b["streak_json"]["streak"] == 20
            else:
                assert b["pick_json"]["result"] == "miss" and b["streak_json"]["streak"] == 0
            out[date_str][scen] = {
                "head_result": h["pick_json"]["result"],
                "head_streak": h["streak_json"]["streak"],
                "base_result": b["pick_json"]["result"],
                "base_streak": b["streak_json"]["streak"],
            }
    return out


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        code_tree = sys.argv[2]
        scenario = sys.argv[3]
        extra = sys.argv[4:]
        res = _worker_run_scenario(code_tree, scenario, extra)
        sys.stdout.write(json.dumps(res))
        return 0

    logs_dir = REVIEW_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "single_hit_and_miss": test_mid_and_late_single_hit_and_miss(),
        "stale_and_multiday_saver_trap": test_stale_reading_and_multiday_saver_trap(),
        "double_down_and_day_outcome": test_mid_and_late_double_down_and_day_outcome(),
        "skip_days": test_mid_and_late_skip_days(),
        "postponed_and_duplicate_eod_alert": test_postponed_games_and_duplicate_eod_alert(),
        "doubleheader": test_doubleheader_and_game_unverified(),
        "void_grading_l01": test_void_grading_l01(),
    }
    out_path = logs_dir / "item1_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified Item 1 probes PASSED -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
