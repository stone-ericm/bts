"""Verified Item 2 whole-range probes: deploy behavior, first start on 2026 snapshot
state under old (`private`, `research`) and half-edited configs, systemd
Restart=always / Type=notify behavior, and rollback to BASESHA (619abda7) after
HEAD wrote 2027 files.

Host-independent (stubs `shutil.disk_usage` and stages synthetic `blend_<date>.pkl`
files). Adoptable as pytest regression tests: no module-level import of
`bts.model.predict`, no process-wide state left changed, no network, no writes
outside `.carbon-review/whole-range/`.
"""
from __future__ import annotations

import contextlib
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
    SyntheticMLBWorld,
    VirtualClock,
    base_live_config,
    harness_env,
    isolated_tmp_root,
    run_worker_subprocess,
    stage_models,
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


def _seed_end_of_2026_snapshot(workdir: Path) -> Path:
    stage_models(workdir)
    picks_dir = workdir / "data" / "picks"
    acc_dir = picks_dir / "account_state"
    health_dir = workdir / "data" / "health_state"
    acc_dir.mkdir(parents=True, exist_ok=True)
    health_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "data" / "processed").mkdir(parents=True, exist_ok=True)

    (picks_dir / "streak.json").write_text(json.dumps({
        "streak": 14,
        "saver_available": False,
        "updated": "2026-09-27T23:45:00Z",
    }, indent=2) + "\n")

    (acc_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 14,
        "best_streak": 27,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-27",
        "recorded_at": "2026-10-20T17:30:00Z",
        "user_id": 12345,
        "username": "stonehengee",
        "saver_available": None,
    }, indent=2) + "\n")

    (acc_dir / "saver_state.json").write_text(json.dumps({
        "season": 2026,
        "state": "used",
        "updated_at": "2026-09-12T04:00:00Z",
        "source": "cli",
    }, indent=2) + "\n")

    (picks_dir / "2026-09-27.json").write_text(json.dumps({
        "date": "2026-09-27",
        "run_time": "2026-09-27T18:05:00-04:00",
        "pick": {
            "batter_name": "Luis Arraez", "batter_id": 500001, "team": "SD",
            "lineup_position": 1, "pitcher_name": "P", "pitcher_id": 770001,
            "p_game_hit": 0.825, "flags": [], "projected_lineup": False,
            "game_pk": 660927, "game_time": "2026-09-27T19:10:00Z",
        },
        "double_down": None,
        "runner_up": None,
        "notification_sent": True,
        "notification_channel": "bluesky_dm",
        "notification_id": "dm-2026",
        "delivered_at": "2026-09-27T14:15:00-04:00",
        "result": "hit",
        "slot_results": {"pick": "hit"},
    }, indent=2) + "\n")
    (picks_dir / "2026-09-27").mkdir(parents=True, exist_ok=True)
    (picks_dir / "2026-09-27" / "decision.json").write_text(json.dumps({
        "schema_version": "bts_daily_decision_v1",
        "date": "2026-09-27",
        "action": "single",
        "source": "mdp",
        "scoreable": True,
        "delivery_status": "delivered",
    }, indent=2) + "\n")
    (health_dir / "health_dm_delivery_status.json").write_text(json.dumps({
        "date": "2026-09-27",
        "sent_sources": [],
    }, indent=2) + "\n")
    return picks_dir


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


def test_head_first_start_snapshot_old_config_private_research() -> dict:
    """First start of HEAD on restored 2026 snapshot with OLD config (`private`, `research`):
    Demonstrates THREE concrete behaviors:
    1. Off-day (2027-03-20): starts cleanly (`exit 0`), emits `READY=1` + `.heartbeat`, sends 0 DMs, idles.
    2. Game-day (2027-03-26) AND `03:00 ET` `bts preview` (`require_contest_state=False`):
       `load_decision_streak_state(picks_dir, require_contest_state=False)` (`src/bts/contest_state.py:597-670`)
       has NO season or contract-switch guard on `contest_streak.json`! While `streak.json` and
       `saver_state.json` reset to 2027 (`model_streak=0`, `saver_flag_state='uninitialized'`),
       `load_decision_streak_state(require_contest_state=False)` selects 2026's `contest_streak.json`
       (`streak=14`, `saver_available=False`, `kind='automatic'`), causing a `p=0.785` 2027 day
       (which `streak=0` would pick) to **SKIP** on 2026's streak!
    3. On a game day where a pick IS locked privately (`p=0.822`, `delivery_status='private_locked'`),
       EOD `post_failure.check` (`src/bts/health/post_failure.py:55-81`) fails to exclude
       `delivery_status == 'private_locked'`, sending a false-positive
       `BTS health CRITICAL alert(s):\n- [pick_delivery] pick locked...` DM to `bluesky.dm_recipient`!
    """
    from click.testing import CliRunner
    import bts.cli as cli_mod

    # 1a. Off-day (2027-03-20)
    with isolated_tmp_root(prefix="snap_old_offday_") as workdir:
        _seed_end_of_2026_snapshot(workdir)
        clock = VirtualClock("2027-03-20T10:00:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-03-20": [], "2027-03-21": []})
        cfg = base_live_config(
            workdir, pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        )
        cfg_path = write_toml_config(workdir / "orch.toml", cfg)
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-03-20", "2027-03-21")) as cap:
            res_off = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-20", "--config", str(cfg_path)])
            assert res_off.exit_code == 0, res_off.output
            assert "READY=1" in cap.sd_events
            assert cap.dms == []
            assert clock.idle_seconds is not None and clock.idle_seconds > 0

    # 1b. Game-day (2027-03-26, p=0.822) under snapshot's old config (`private`, `research`)
    with isolated_tmp_root(prefix="snap_old_gameday_") as workdir:
        picks_dir = _seed_end_of_2026_snapshot(workdir)
        clock = VirtualClock("2027-03-26T10:00:00-04:00")
        world, rows = _single_game_world(clock, "2027-03-26", "2027-03-27", p_hit=0.822)
        cfg = base_live_config(
            workdir, pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        )
        cfg_path = write_toml_config(workdir / "orch.toml", cfg)
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-03-26", "2027-03-27")) as cap:
            res_game = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-26", "--config", str(cfg_path)])
            assert res_game.exit_code == 0, res_game.output
            dec = json.loads((picks_dir / "2027-03-26" / "decision.json").read_text())
            pick = json.loads((picks_dir / "2027-03-26.json").read_text())
            streak_after = json.loads((picks_dir / "streak.json").read_text())

            assert dec["delivery_status"] == "private_locked" and dec["scoreable"] is True
            # 2026 contest_streak.json (streak=14) carried into 2027 under require_contest_state=False!
            assert pick["chosen_on"] == {
                "schema": "bts_pick_chosen_on_v1",
                "required": False,
                "kind": "automatic",
                "streak": 14,
                "saver_available": False,
                "dated": None,
            }
            assert streak_after["streak"] == 1 and streak_after["season"] == 2027
            # EOD post_failure CRITICAL DM fires on private_locked day!
            assert len(cap.dms) == 1, cap.dms
            assert "BTS health CRITICAL alert(s):\n- [pick_delivery] pick locked for 2027-03-26" in cap.dms[0]["text"]
            eod_dm = cap.dms[0]["text"]

    # 1c. Game-day with p=0.785 under snapshot's old config (`private`, `research`):
    # At true 2027 streak=0, MDP plays (bar=0.0%); because 2026's contest_streak=14 leaks in, MDP SKIPS!
    with isolated_tmp_root(prefix="snap_old_skip_leak_") as workdir:
        picks_dir = _seed_end_of_2026_snapshot(workdir)
        clock = VirtualClock("2027-03-26T10:00:00-04:00")
        world, rows_785 = _single_game_world(clock, "2027-03-26", "2027-03-27", p_hit=0.785)
        cfg = base_live_config(
            workdir, pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        )
        cfg_path = write_toml_config(workdir / "orch.toml", cfg)
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows_785), ("2027-03-26", "2027-03-27")):
            res_785 = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-03-26", "--config", str(cfg_path)])
            assert res_785.exit_code == 0, res_785.output
            dec_785 = json.loads((picks_dir / "2027-03-26" / "decision.json").read_text())
            assert dec_785["action"] == "skip" and dec_785["streak"] == 14

    return {
        "offday_exit_code": res_off.exit_code,
        "gameday_chosen_on_leaks_2026_streak": pick["chosen_on"],
        "gameday_p785_action_under_leaked_2026_streak": dec_785["action"],
        "gameday_private_locked_false_eod_dm": eod_dm,
    }


def test_head_half_edited_dm_without_contest_state_required_and_systemd_loop() -> dict:
    from click.testing import CliRunner
    import importlib.util
    import bts.cli as cli_mod

    spec = importlib.util.spec_from_file_location("check_heartbeat", str(HEAD_SRC.parent / "scripts" / "check_heartbeat.py"))
    ch_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ch_mod)

    results = {}
    for label, entry_intent in (("dm_intent_enter_no_cs_flag", "enter"), ("dm_intent_research_mismatch", "research")):
        with isolated_tmp_root(prefix=f"half1_{label}_") as workdir:
            _seed_end_of_2026_snapshot(workdir)
            clock = VirtualClock("2027-03-26T10:00:00-04:00")
            world, rows = _single_game_world(clock, "2027-03-26", "2027-03-27")
            cfg = base_live_config(
                workdir, pick_delivery="dm", entry_intent=entry_intent,
                contest_state_required=False, contest_state_expected=False,
            )
            cfg_path = write_toml_config(workdir / "orch.toml", cfg)
            hb_path = workdir / "data" / ".heartbeat"

            samples: list[dict] = []
            _, _, samples = ch_mod.assess_churn(samples, 0, clock.now_utc())

            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-03-26", "2027-03-27")) as cap:
                runner = CliRunner()
                codes = []
                churn_at_cycle = []
                for cycle in range(1, 4):
                    clock.advance(30.0, "systemd_restart_sec")
                    r = runner.invoke(cli_mod.cli, ["schedule", "--date", "2027-03-26", "--config", str(cfg_path)])
                    codes.append(r.exit_code)
                    churn, reason, samples = ch_mod.assess_churn(samples, cycle, clock.now_utc())
                    churn_at_cycle.append((cycle, churn, reason))

                assert codes == [2, 2, 2], codes
                assert cap.dms == []
                assert cap.sd_events == []
                assert not hb_path.exists()
                assert churn_at_cycle[-1][1] is True

            results[label] = {
                "exit_codes": codes,
                "stderr_sample": r.output.strip(),
                "dms_sent_across_3_restarts": len(cap.dms),
                "sd_events": cap.sd_events,
                "heartbeat_written": hb_path.exists(),
                "churn_tripped_on_restart_3": churn_at_cycle[-1][2],
            }
    return results


def test_head_half_edited_required_but_no_switch_file() -> dict:
    from click.testing import CliRunner
    import types as _types
    import bts.cli as cli_mod

    out = {}
    for with_preview in (False, True):
        label = "with_3am_preview" if with_preview else "no_preview"
        with isolated_tmp_root(prefix=f"half2_{label}_") as workdir:
            picks_dir = _seed_end_of_2026_snapshot(workdir)
            clock = VirtualClock("2027-07-15T03:05:00-04:00" if with_preview else "2027-07-15T10:00:00-04:00")
            world, rows = _single_game_world(clock, "2027-07-15", "2027-07-16")
            cfg = base_live_config(
                workdir, pick_delivery="dm", entry_intent="enter",
                contest_state_required=True, contest_state_expected=True,
            )
            cfg_path = write_toml_config(workdir / "orch.toml", cfg)

            with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-07-15", "2027-07-16")) as cap:
                runner = CliRunner()
                if with_preview:
                    fake_pred_mod = _types.ModuleType("bts.model.predict")
                    proj_rows = [dict(rows[0], flags="PROJECTED")]
                    fake_pred_mod.run_pipeline = lambda *a, **k: pd.DataFrame(proj_rows)
                    fake_pred_mod.load_blend = lambda *a, **k: None
                    old_mod = sys.modules.get("bts.model.predict")
                    sys.modules["bts.model.predict"] = fake_pred_mod
                    try:
                        rp = runner.invoke(cli_mod.cli, [
                            "preview", "--date", "2027-07-15",
                            "--picks-dir", str(picks_dir),
                            "--data-dir", str(workdir / "data" / "processed"),
                            "--models-dir", str(workdir / "data" / "models"),
                        ])
                        assert rp.exit_code == 0, rp.output
                    finally:
                        if old_mod is None:
                            sys.modules.pop("bts.model.predict", None)
                        else:
                            sys.modules["bts.model.predict"] = old_mod
                    assert (picks_dir / "2027-07-15.json").exists()
                    clock.set_et("2027-07-15T10:00:00-04:00")

                rs = runner.invoke(cli_mod.cli, ["schedule", "--date", "2027-07-15", "--config", str(cfg_path)])
                assert rs.exit_code == 0, rs.output
                assert "READY=1" in cap.sd_events
                assert (workdir / "data" / ".heartbeat").exists()
                assert clock.idle_seconds is not None and clock.idle_seconds > 0
                assert not (picks_dir / "2027-07-15" / "decision.json").exists()

                dm_texts = [d["text"] for d in cap.dms]
                if not with_preview:
                    assert len(dm_texts) == 2, dm_texts
                    assert "- [contest_state] Morning check for 2027-07-15: no usable contest state" in dm_texts[0]
                    assert "- [contest_state] live pick path requires contest-account state" in dm_texts[1]
                else:
                    assert len(dm_texts) == 4, dm_texts
                    assert "- [contest_state] Morning check for 2027-07-15: no usable contest state" in dm_texts[0]
                    assert "- [contest_state] live pick path requires contest-account state" in dm_texts[1]
                    assert "- [missed_pick] NO PICK DELIVERED" in dm_texts[2]
                    assert "- [pick_delivery] pick locked for 2027-07-15" in dm_texts[3]

                out[label] = {
                    "ready_notified": "READY=1" in cap.sd_events,
                    "idle_seconds": clock.idle_seconds,
                    "dm_count": len(dm_texts),
                    "dms": dm_texts,
                }
    return out


def _stage_head_written_files(workdir: Path, *, locked_pick: bool, include_auto: bool, switch_status: str = "automatic") -> Path:
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
    (workdir / "data" / "processed").mkdir(parents=True, exist_ok=True)

    clock = VirtualClock("2027-07-15T14:00:00-04:00")
    world, _ = _single_game_world(clock, "2027-07-15", "2027-07-16")
    with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(), ("2027-07-15", "2027-07-16")):
        runner = CliRunner()
        r_en = runner.invoke(cli_mod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks_dir)])
        assert r_en.exit_code == 0, r_en.output
        act_id = json.loads((acc_dir / "season_contract.json").read_text())["activation_id"]

        (acc_dir / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "active", "updated_at": "2027-07-01T12:00:00Z", "source": "cli",
        }, indent=2) + "\n")

        r_man = runner.invoke(cli_mod.cli, [
            "set-contest-streak", "--date", "2027-07-15",
            "--streak", "19", "--best-streak", "25", "--saver", "used",
            "--picks-dir", str(picks_dir),
        ])
        assert r_man.exit_code == 0, r_man.output

        if switch_status == "revoked":
            contract = json.loads((acc_dir / "season_contract.json").read_text())
            contract["status"] = "revoked"
            contract["revoked_at"] = "2027-07-15T18:00:00Z"
            contract["note"] = "test revoke"
            (acc_dir / "season_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

        if include_auto:
            (acc_dir / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v2",
                "season": 2027,
                "activation_id": act_id,
                "active_streak": 12,
                "best_streak": 30,
                "source": "mlb_bts_profile",
                "source_date": "2027-07-14",
                "recorded_at": "2027-07-15T06:10:00Z",
                "user_id": 12345,
                "username": "stonehengee",
                "saver_available": None,
            }, indent=2) + "\n")

        picks_mod.save_streak(12, picks_dir, saver_available=True, applied_date="2027-07-14", season=2027)

        dp = picks_mod.DailyPick(
            date="2027-07-15",
            run_time="2027-07-15T18:05:00-04:00",
            pick=picks_mod.Pick(
                "Luis Arraez", 500001, "NYY", 1, "Synth Pitcher", 770001,
                0.822, [] if locked_pick else ["PROJECTED"], not locked_pick,
                7001, "2027-07-15T23:35:00Z", "BOS",
            ),
            double_down=None,
            runner_up=None,
            notification_sent=locked_pick,
            notification_channel="bluesky_dm" if locked_pick else None,
            notification_id="dm-head-001" if locked_pick else None,
            delivered_at="2027-07-15T18:06:00-04:00" if locked_pick else None,
            chosen_on={
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "automatic",
                "streak": 12,
                "saver_available": True,
                "dated": None,
            },
            automatic_reading={
                "schema": "bts_pick_automatic_reading_v1",
                "recorded_at": "2027-07-15T06:10:00Z",
                "age_hours": 11.92,
                "stale": False,
                "reason": None,
            },
        )
        picks_mod.save_pick(dp, picks_dir)
        picks_mod.save_shadow_pick(dp, picks_dir)

        if locked_pick:
            dd_mod.write_decision(
                picks_dir,
                dd_mod.build_pick_decision(
                    date="2027-07-15",
                    action="single",
                    source="mdp",
                    scoreable=True,
                    delivery_status="delivered",
                    primary={"batter_name": "Luis Arraez", "batter_id": 500001, "game_pk": 7001, "p_game_hit": 0.822},
                    double_down=None,
                    streak=12,
                    saver_available=True,
                    state_source="contest_auto",
                    state_status="fresh",
                    allow_double=True,
                    contest_source_date="2027-07-14",
                ),
            )

        (health_dir / "contest_morning_check_dm_delivery_status.json").write_text('{"date":"2027-07-15","sent_incidents":["contest_state:morning"]}\n')
        (health_dir / "contest_stale_reading_dm_delivery_status.json").write_text('{"date":"2027-07-15","sent_incidents":["contest_state:stale_reading"]}\n')
        (health_dir / "health_dm_delivery_status.json").write_text('{"date":"2027-07-15","sent_sources":[]}\n')
        (health_dir / "contest_streak_fetch_status.json").write_text('{"last_success_at":"2027-07-15T06:10:00Z","last_error":null}\n')
        receipt_io.publish(
            entry_receipt.receipts_root(picks_dir) / "2027-07-15" / "20270715T181500-aaa.json",
            b'{"schema":"bts_pick_entry_receipt_v1","outcome":"no_pick_file"}\n',
        )
    return picks_dir


def _base_worker(scenario: str, extra: list[str]) -> dict:
    import bts
    import bts.contest_state as cs_mod
    import bts.health.contest_state as h_cs_mod
    import bts.picks as picks_mod
    import bts.scheduler as sch_mod
    import bts.web as web_mod

    assert Path(bts.__file__).resolve().is_relative_to(BASE_SRC.resolve())
    workdir = Path(extra[0])
    picks_dir = workdir / "data" / "picks"

    if scenario == "inspect_files_with_auto":
        clock = VirtualClock("2027-07-15T18:10:00-04:00")
        world, rows = _single_game_world(clock, "2027-07-15", "2027-07-16")
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
                    from harness_lib import make_pinned_datetime_cls
                    web_stack.enter_context(patch.object(web_mod, "datetime", make_pinned_datetime_cls(clock)))
                html = web_mod.render_page()
                assert "Luis Arraez" in html

            ds = cs_mod.load_decision_streak_state(picks_dir, require_contest_state=True, now=clock.now_utc())
            h_alerts = [
                {"level": a.level, "source": a.source, "message": a.message}
                for a in h_cs_mod.check(picks_dir, expected=True, now=clock.now_utc())
            ]

            cfg = base_live_config(workdir)
            sch_mod.run_day(date="2027-07-15", config=cfg)

            pick_after = json.loads((picks_dir / "2027-07-15.json").read_text())
            streak_after = json.loads((picks_dir / "streak.json").read_text())
            return {
                "decision_streak_state": {
                    "streak": ds.streak,
                    "saver_available": ds.saver_available,
                    "source": ds.source,
                    "best_streak": ds.best_streak,
                    "best_status": ds.best_status,
                },
                "health_contest_alerts": h_alerts,
                "dms_sent_by_basesha": [d["text"] for d in cap.dms],
                "pick_keys_after_basesha_save": sorted(pick_after.keys()),
                "pick_result_after_basesha": pick_after["result"],
                "streak_after_basesha_save": streak_after,
            }

    if scenario == "inspect_v3_manual_without_auto":
        clock = VirtualClock("2027-07-15T18:10:00-04:00")
        world, _ = _single_game_world(clock, "2027-07-15", "2027-07-16")
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
                    "streak": ds_day1.streak,
                    "saver_available": ds_day1.saver_available,
                    "best_status": ds_day1.best_status,
                    "alerts": alerts_day1,
                },
                "day6_still_active": {
                    "streak": ds_day6.streak,
                    "saver_available": ds_day6.saver_available,
                },
            }

    if scenario == "run_unlocked_day_after_rollback":
        clock = VirtualClock("2027-07-15T18:10:00-04:00")
        world, rows = _single_game_world(clock, "2027-07-15", "2027-07-16")
        cfg = base_live_config(workdir)
        buf = io.StringIO()
        with _host_clean_env(workdir, clock, world, lambda *a, **k: pd.DataFrame(rows), ("2027-07-15", "2027-07-16")) as cap, \
             contextlib.redirect_stderr(buf):
            sch_mod.run_day(date="2027-07-15", config=cfg)
        dec = json.loads((picks_dir / "2027-07-15" / "decision.json").read_text())
        return {
            "dms": [d["text"] for d in cap.dms],
            "decision": dec,
        }

    raise ValueError(f"Unknown base worker scenario: {scenario}")


def test_rollback_to_basesha_after_head_wrote_files() -> dict:
    import bts.picks as head_picks_mod

    script = Path(__file__).resolve()

    with isolated_tmp_root(prefix="rb_locked_") as workdir:
        picks_dir = _stage_head_written_files(workdir, locked_pick=True, include_auto=True, switch_status="revoked")
        r_auto = run_worker_subprocess(script, "base", "inspect_files_with_auto", [str(workdir)])

        assert r_auto["decision_streak_state"]["streak"] == 12
        assert r_auto["decision_streak_state"]["saver_available"] is True
        assert r_auto["decision_streak_state"]["best_status"] == "untrusted"
        assert any("manual override" in a["message"] for a in r_auto["health_contest_alerts"]), r_auto["health_contest_alerts"]

        assert r_auto["dms_sent_by_basesha"] == [], r_auto["dms_sent_by_basesha"]
        assert r_auto["pick_result_after_basesha"] == "hit"
        assert "chosen_on" not in r_auto["pick_keys_after_basesha_save"]
        assert "automatic_reading" not in r_auto["pick_keys_after_basesha_save"]
        assert "season" not in r_auto["streak_after_basesha_save"]
        assert "applied_dates" not in r_auto["streak_after_basesha_save"]

        head_streak_on_redeploy = head_picks_mod.load_streak(picks_dir, season=2027)
        assert head_streak_on_redeploy == 0

    with isolated_tmp_root(prefix="rb_v3_only_") as workdir:
        _stage_head_written_files(workdir, locked_pick=True, include_auto=False, switch_status="revoked")
        r_v3 = run_worker_subprocess(script, "base", "inspect_v3_manual_without_auto", [str(workdir)])
        assert r_v3["day1"]["streak"] == 19
        assert r_v3["day1"]["saver_available"] is True
        assert r_v3["day1"]["best_status"] == "untrusted"
        assert any(a["level"] == "CRITICAL" and "no source_date" in a["message"] for a in r_v3["day1"]["alerts"])
        assert r_v3["day6_still_active"]["streak"] == 19

    with isolated_tmp_root(prefix="rb_unlocked_") as workdir:
        _stage_head_written_files(workdir, locked_pick=False, include_auto=True, switch_status="automatic")
        r_un = run_worker_subprocess(script, "base", "run_unlocked_day_after_rollback", [str(workdir)])
        assert r_un["decision"]["action"] == "skip"
        assert any("No pick yet" in m for m in r_un["dms"]), r_un["dms"]

    return {
        "locked_with_auto_and_v3": r_auto,
        "head_model_streak_after_basesha_save": head_streak_on_redeploy,
        "v3_without_auto": r_v3,
        "unlocked_day_after_rollback": r_un,
    }


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        code_tree = sys.argv[2]
        assert code_tree == "base"
        res = _base_worker(sys.argv[3], sys.argv[4:])
        sys.stdout.write(json.dumps(res))
        return 0

    logs_dir = REVIEW_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "first_start_old_config": test_head_first_start_snapshot_old_config_private_research(),
        "half_edited_dm_no_cs_flag": test_head_half_edited_dm_without_contest_state_required_and_systemd_loop(),
        "half_edited_required_no_switch": test_head_half_edited_required_but_no_switch_file(),
        "rollback_to_basesha": test_rollback_to_basesha_after_head_wrote_files(),
    }
    out_path = logs_dir / "item2_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified Item 2 probes PASSED -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
