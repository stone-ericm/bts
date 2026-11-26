"""Verified Item 3 & 4 whole-range probes: cron setup under `research` and `enter`,
CLI flag verification (`--help` and stubbed runtime), restart/timing collision
checks, `enter` DM trigger matrix, and systemd `bts-leaderboard.service` drift
blindspot.

Adoptable as pytest regression tests: no module-level import of `bts.model.predict`,
no process-wide state left changed, no network, no writes outside
`.carbon-review/whole-range/`.
"""
from __future__ import annotations

import ast
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from harness_lib import (  # noqa: E402
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
    write_covered_auto_state,
    write_toml_config,
)

if str(HEAD_SRC) not in sys.path:
    sys.path.insert(0, str(HEAD_SRC))


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
echo "unexpected uv args: $*" >&2
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
    if [ ! -f "$STORE" ]; then
        echo "no crontab for synthuser" >&2
        exit 1
    fi
    cat "$STORE"
    exit 0
elif [ "${{1:-}}" = "-" ]; then
    cat > "$STORE"
    exit 0
fi
echo "unexpected crontab args: $*" >&2
exit 2
""",
        encoding="utf-8",
    )
    crontab_script.chmod(crontab_script.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env["HOME"] = str(home)
    env["PYTHONPATH"] = f"{HEAD_SRC}:{REPO_ROOT}"
    env["PATH"] = f"{shim_bin}:{uv_dir}:{env.get('PATH', '')}"
    env["HEALTHCHECKS_PING_URL"] = "https://hc-ping.com/00000000-0000-0000-0000-000000000000"
    env["TMPDIR"] = str(workdir)
    return home, crontab_store, env


def _run_cron_setup(env: dict[str, str], action: str) -> subprocess.CompletedProcess[str]:
    script = REPO_ROOT / "scripts" / "cron-setup-hetzner.sh"
    return subprocess.run(
        ["bash", str(script), action],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_cron_show_and_install_both_intents() -> dict:
    out = {}
    with isolated_tmp_root(prefix="cron_intents_") as workdir:
        cfg_res = base_live_config(
            workdir / "res", pick_delivery="private", entry_intent="research",
            contest_state_required=False, contest_state_expected=False,
        )
        _, store_res, env_res = _setup_cron_shim_home(workdir / "res", cfg_res)
        store_res.write_text("15 4 * * * /usr/local/bin/unrelated-job\n")

        show_res = _run_cron_setup(env_res, "show")
        assert show_res.returncode == 0, show_res.stderr
        res_lines = [ln for ln in show_res.stdout.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        assert len(res_lines) == 17, len(res_lines)
        assert not any("check-pick-entered" in ln for ln in res_lines)
        fetch_res = [ln for ln in res_lines if "fetch-contest-streak" in ln]
        assert len(fetch_res) == 4 and all("--dm-recipient" not in ln for ln in fetch_res)
        day_out_res = [ln for ln in res_lines if "day-outcome" in ln]
        assert len(day_out_res) == 1 and "--dm-recipient stonehengee.bsky.social" in day_out_res[0]

        inst_res = _run_cron_setup(env_res, "install")
        assert inst_res.returncode == 0, inst_res.stderr
        assert "/usr/local/bin/unrelated-job" in store_res.read_text()

        cfg_ent = base_live_config(
            workdir / "ent", pick_delivery="dm", entry_intent="enter",
            contest_state_required=True, contest_state_expected=True,
        )
        _, store_ent, env_ent = _setup_cron_shim_home(workdir / "ent", cfg_ent)
        show_ent = _run_cron_setup(env_ent, "show")
        assert show_ent.returncode == 0, show_ent.stderr
        ent_lines = [ln for ln in show_ent.stdout.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        assert len(ent_lines) == 18, len(ent_lines)
        fetch_ent = [ln for ln in ent_lines if "fetch-contest-streak" in ln]
        assert len(fetch_ent) == 4 and all("--dm-recipient stonehengee.bsky.social" in ln for ln in fetch_ent)
        cpe_ent = [ln for ln in ent_lines if "check-pick-entered" in ln]
        assert len(cpe_ent) == 1 and "--dm-recipient stonehengee.bsky.social" in cpe_ent[0]

        inst_ent = _run_cron_setup(env_ent, "install")
        assert inst_ent.returncode == 0, inst_ent.stderr

        cfg_bad = base_live_config(
            workdir / "bad", pick_delivery="dm", entry_intent="research",
            contest_state_required=True, contest_state_expected=True,
        )
        _, store_bad, env_bad = _setup_cron_shim_home(workdir / "bad", cfg_bad)
        before_bad = "0 0 * * * /bin/true\n"
        store_bad.write_text(before_bad)
        bad_res = _run_cron_setup(env_bad, "install")
        assert bad_res.returncode != 0
        assert store_bad.read_text() == before_bad

        out["research_line_count"] = len(res_lines)
        out["enter_line_count"] = len(ent_lines)
        out["enter_lines"] = ent_lines
    return out


def _extract_cli_invocations(cron_lines: list[str], orch_path: str) -> list[list[str]]:
    substitutions = {
        r"$(date -d yesterday +\%Y-\%m-\%d)": "2027-07-14",
        r'$(date -d "7 days ago" +\%Y-\%m-\%d)': "2027-07-08",
        r"$(date +\%Y-\%m-\%d)": "2027-07-15",
        r"$(date +\%Y)": "2027",
        r'"$BTS_SCHEDULER_HEARTBEAT_PING_URL"': "https://hc-ping.com/hb-uuid",
    }
    extracted: list[list[str]] = []
    for ln in cron_lines:
        s = ln
        for k, v in substitutions.items():
            s = s.replace(k, v)
        for m in re.finditer(r"\.local/bin/uv run ([^>&;)#]+)", s):
            raw_cmd = m.group(1).strip()
            tokens = shlex.split(raw_cmd)
            tokens = [orch_path if t.endswith(".bts-orchestrator.toml") else t for t in tokens]
            extracted.append(tokens)
    return extracted


def test_every_cron_command_exists_and_executes_dry_or_stubbed() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod

    with isolated_tmp_root(prefix="cron_cmds_") as workdir:
        cfg = base_live_config(workdir, pick_delivery="dm", entry_intent="enter")
        orch_path = write_toml_config(workdir / "orch.toml", cfg)
        _, _, env = _setup_cron_shim_home(workdir / "shim", cfg)
        show = _run_cron_setup(env, "show")
        assert show.returncode == 0
        ent_lines = [ln for ln in show.stdout.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        invocations = _extract_cli_invocations(ent_lines, str(orch_path))
        unique_invocations: list[list[str]] = []
        for inv in invocations:
            if inv not in unique_invocations:
                unique_invocations.append(inv)

        assert len(unique_invocations) == 14, unique_invocations

        runner = CliRunner()
        verified_help: list[str] = []
        verified_stubbed: list[str] = []

        for tokens in unique_invocations:
            if tokens[0] == "python":
                proc = subprocess.run(
                    [sys.executable, str(REPO_ROOT / tokens[1]), "--help"],
                    capture_output=True, text=True, timeout=10,
                )
                assert proc.returncode == 0, proc.stderr
                verified_help.append(" ".join(tokens))
                continue

            assert tokens[0] == "bts"
            bts_args = tokens[1:]
            cmd_path = []
            grp = cli_mod.cli
            idx = 0
            while idx < len(bts_args) and not bts_args[idx].startswith("-"):
                sub = grp.get_command(None, bts_args[idx])  # type: ignore[arg-type]
                if sub is None:
                    break
                cmd_path.append(bts_args[idx])
                grp = sub
                idx += 1
            assert cmd_path, f"Could not resolve command in {bts_args}"
            r_help = runner.invoke(cli_mod.cli, [*cmd_path, "--help"])
            assert r_help.exit_code == 0, f"{cmd_path} --help failed: {r_help.output}"

            parent_ctx = cli_mod.cli.make_context("bts", list(bts_args))
            cur_cmd = cli_mod.cli
            rem_args = list(parent_ctx.protected_args + parent_ctx.args)
            while isinstance(cur_cmd, cli_mod.click.Group) and rem_args:
                cname, cobj, rem = cur_cmd.resolve_command(parent_ctx, rem_args)
                assert cobj is not None, f"Unresolved subcommand in {bts_args}"
                parent_ctx = cobj.make_context(cname, rem, parent=parent_ctx)
                cur_cmd = cobj
                rem_args = list(parent_ctx.protected_args + parent_ctx.args)
            assert rem_args == [], f"Unparsed leftover args in {bts_args}: {rem_args}"
            verified_help.append("bts " + " ".join(bts_args))

        clock = VirtualClock("2027-07-15T08:30:00-04:00")
        world = SyntheticMLBWorld(clock, {"2027-07-14": [], "2027-07-15": []})
        write_covered_auto_state(
            workdir / "data" / "picks", season=2027, streak=5, best_streak=5,
            source_date="2027-07-14", recorded_at_utc="2027-07-15T06:10:00Z",
        )
        with harness_env(workdir, clock, world, lambda *a, **k: pd.DataFrame()):
            for bts_args in (
                ["check-results", "--date", "2027-07-14", "--wait-deadline-et", "06:00"],
                ["reconcile"],
                ["skip-policy-shadow-update"],
                ["day-outcome", "--date", "2027-07-14", "--config", str(orch_path),
                 "--picks-dir", "data/picks", "--expected-username", "stonehengee",
                 "--dm-recipient", "stonehengee.bsky.social"],
            ):
                r = runner.invoke(cli_mod.cli, bts_args)
                assert r.exit_code == 0, f"{bts_args} failed: {r.output}"
                verified_stubbed.append("bts " + " ".join(bts_args))

    return {
        "verified_cli_commands_and_flags": verified_help,
        "verified_runtime_invocations": verified_stubbed,
    }


def test_cron_timing_collisions_and_heartbeat_restart_window() -> dict:
    import importlib.util
    import bts.heartbeat as hb_mod
    from bts.contest_state import AUTOMATIC_READING_STALE_HOURS

    spec = importlib.util.spec_from_file_location("ch_mod", str(REPO_ROOT / "scripts" / "check_heartbeat.py"))
    ch_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ch_mod)

    with isolated_tmp_root(prefix="hb_window_") as workdir:
        hb_path = workdir / ".heartbeat"
        wake_utc = RealDatetime.fromisoformat("2027-07-15T14:00:00+00:00")
        hb_mod.write_heartbeat(
            hb_path, state=hb_mod.HeartbeatState.SLEEPING,
            now_utc=RealDatetime.fromisoformat("2027-07-15T02:30:00+00:00"),
            sleeping_until=wake_utc,
        )
        stale_1000, _ = ch_mod.is_stale(hb_path, now=wake_utc)
        stale_1025, _ = ch_mod.is_stale(hb_path, now=RealDatetime.fromisoformat("2027-07-15T14:00:25+00:00"))
        assert stale_1000 is False and stale_1025 is False

        samples: list[dict] = []
        _, _, samples = ch_mod.assess_churn(samples, 5, RealDatetime.fromisoformat("2027-07-15T13:55:00+00:00"))
        churn_1005, _, _ = ch_mod.assess_churn(samples, 6, RealDatetime.fromisoformat("2027-07-15T14:05:00+00:00"))
        assert churn_1005 is False

    age_after_0210 = round((RealDatetime.fromisoformat("2027-07-15T10:00:00-04:00")
                            - RealDatetime.fromisoformat("2027-07-15T02:10:00-04:00")).total_seconds() / 3600, 2)
    age_if_overnight_fetches_fail = round((RealDatetime.fromisoformat("2027-07-15T10:00:00-04:00")
                                           - RealDatetime.fromisoformat("2027-07-14T13:30:00-04:00")).total_seconds() / 3600, 2)
    assert age_after_0210 < AUTOMATIC_READING_STALE_HOURS < age_if_overnight_fetches_fail

    return {
        "heartbeat_at_1000_et_stale": stale_1000,
        "heartbeat_in_30s_restart_gap_stale": stale_1025,
        "daily_restart_churn": churn_1005,
        "age_hours_at_1000_after_0210_fetch": age_after_0210,
        "age_hours_at_1000_if_0110_0210_fail_before_1030_fetch": age_if_overnight_fetches_fail,
    }


def test_enter_cron_dm_matrix() -> dict:
    from click.testing import CliRunner
    import bts.cli as cli_mod
    import bts.daily_decision as dd_mod
    import bts.picks as picks_mod
    from bts.leaderboard.auth import AuthError, SessionInfo

    with isolated_tmp_root(prefix="cron_dm_matrix_") as workdir:
        picks_dir = workdir / "data" / "picks"
        clock = VirtualClock("2027-07-15T18:15:00-04:00")
        world = SyntheticMLBWorld(clock, {
            "2027-07-15": [GameSpec(game_pk=7001, official_date="2027-07-15", game_date_utc="2027-07-15T23:05:00Z")],
        })
        write_covered_auto_state(
            picks_dir, season=2027, streak=12, best_streak=15,
            source_date="2027-07-14", recorded_at_utc="2027-07-15T06:10:00Z",
        )
        dp = picks_mod.DailyPick(
            date="2027-07-15", run_time="2027-07-15T18:05:00-04:00",
            pick=picks_mod.Pick("Luis Arraez", 500001, "NYY", 1, "P", 770001, 0.822, [], False, 7001, "2027-07-15T23:05:00Z"),
            double_down=None, runner_up=None,
            notification_sent=True, notification_channel="bluesky_dm", notification_id="dm-1",
            delivered_at="2027-07-15T18:05:00-04:00",
        )
        picks_mod.save_pick(dp, picks_dir)
        dd_mod.write_decision(
            picks_dir,
            dd_mod.build_pick_decision(
                date="2027-07-15", action="single", source="mdp", scoreable=True, delivery_status="delivered",
                primary={"batter_name": "Luis Arraez", "batter_id": 500001, "game_pk": 7001, "p_game_hit": 0.822},
                double_down=None, streak=12, saver_available=True, state_source="contest_auto",
                state_status="fresh", allow_double=True, contest_source_date="2027-07-14",
            ),
        )

        runner = CliRunner()
        with harness_env(workdir, clock, world, lambda *a, **k: pd.DataFrame()) as cap:
            with patch("bts.leaderboard.auth.load_session_cookies", side_effect=AuthError("expired cookies")):
                rf1 = runner.invoke(cli_mod.cli, [
                    "fetch-contest-streak", "--picks-dir", str(picks_dir),
                    "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                ])
                rf2 = runner.invoke(cli_mod.cli, [
                    "fetch-contest-streak", "--picks-dir", str(picks_dir),
                    "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                ])
            assert rf1.exit_code == 2 and rf2.exit_code == 2
            assert len(cap.dms) == 1 and "contest-streak" in cap.dms[0]["text"].lower(), cap.dms

            with patch("bts.leaderboard.auth.load_session_cookies", side_effect=AuthError("flap")):
                rc_err = runner.invoke(cli_mod.cli, [
                    "check-pick-entered", "--picks-dir", str(picks_dir),
                    "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                    "--now-et", "2027-07-15T18:15:00",
                ])
            assert rc_err.exit_code == 0 and len(cap.dms) == 1

            fake_session = SessionInfo(user_id=12345, username="stonehengee", xsid="xs")
            today_date = RealDatetime(2027, 7, 15).date()
            with patch("bts.leaderboard.auth.load_session_cookies", return_value={"ipid": "1"}), \
                 patch("bts.leaderboard.auth.extract_uid", return_value="u1"), \
                 patch("bts.leaderboard.auth.fetch_login_session", return_value=fake_session), \
                 patch("bts.contest_fetch.fetch_profile", return_value={"predictions": []}), \
                 patch("bts.cli._fetch_rounds", return_value={101: today_date}), \
                 patch("bts.cli._fetch_bts_to_mlb", return_value={999: 500001}):
                with patch("bts.contest_fetch.fetch_pending_predictions", return_value=[]):
                    for ts in ("2027-07-15T18:15:00", "2027-07-15T18:35:00", "2027-07-15T18:48:00"):
                        rc = runner.invoke(cli_mod.cli, [
                            "check-pick-entered", "--picks-dir", str(picks_dir),
                            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                            "--now-et", ts,
                        ])
                        assert rc.exit_code == 1
                with patch("bts.contest_fetch.fetch_pending_predictions", return_value=[{"roundId": 101, "playerId": 999, "unitId": 1, "number": 1}]):
                    rc_ok = runner.invoke(cli_mod.cli, [
                        "check-pick-entered", "--picks-dir", str(picks_dir),
                        "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                        "--now-et", "2027-07-15T18:52:00",
                    ])
                    assert rc_ok.exit_code == 0

            cpe_dms = [d["text"] for d in cap.dms[1:]]
            assert len(cpe_dms) == 4, cpe_dms
            assert "NOT entered" in cpe_dms[0]
            assert "STILL NOT entered" in cpe_dms[1]
            assert "STILL NOT entered" in cpe_dms[2]
            assert "pick entry confirmed" in cpe_dms[3]

    non_dm_modules = [
        HEAD_SRC / "bts" / "data" / "pull.py",
        HEAD_SRC / "bts" / "data" / "build.py",
        HEAD_SRC / "bts" / "data" / "sync.py",
        HEAD_SRC / "bts" / "data" / "backup.py",
        HEAD_SRC / "bts" / "data" / "backup_cli.py",
        HEAD_SRC / "bts" / "features" / "park_drag_producer.py",
        HEAD_SRC / "bts" / "skip_policy_shadow.py",
        REPO_ROOT / "scripts" / "check_heartbeat.py",
    ]
    for mod_path in non_dm_modules:
        tree = ast.parse(mod_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "dm" in node.module.split("."):
                raise AssertionError(f"Unexpected DM import in {mod_path}")

    return {
        "fetch_contest_streak_dm": cap.dms[0]["text"],
        "check_pick_entered_ladder": cpe_dms,
    }


def test_leaderboard_systemd_unit_drift_blindspot() -> dict:
    import bts.health.unit_drift as ud_mod

    head_lb = (REPO_ROOT / "deploy" / "systemd" / "bts-leaderboard.service").read_text()
    assert "--dm-recipient stonehengee.bsky.social" in head_lb
    stale_snapshot_lb = head_lb.replace("stonehengee.bsky.social", "stoneericm.bsky.social")
    assert stale_snapshot_lb != head_lb

    with isolated_tmp_root(prefix="unit_drift_lb_") as workdir:
        home = workdir / "home"
        installed_dir = home / ".config" / "systemd" / "user"
        installed_dir.mkdir(parents=True, exist_ok=True)
        for name in ("bts-scheduler.service", "bts-dashboard.service"):
            (installed_dir / name).write_text((REPO_ROOT / "scripts" / "systemd" / name).read_text())
        (installed_dir / "bts-leaderboard.service").write_text(stale_snapshot_lb)
        (installed_dir / "bts-leaderboard.timer").write_text(
            (REPO_ROOT / "deploy" / "systemd" / "bts-leaderboard.timer").read_text()
        )

        alerts = ud_mod.check(installed_dir=installed_dir, repo_units_dir=REPO_ROOT / "scripts" / "systemd")
        assert alerts == []

        env = dict(os.environ)
        env["HOME"] = str(home)
        env["TMPDIR"] = str(workdir)
        proc = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "install-systemd-hetzner.sh"), "--diff"],
            env=env, capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == 0
        assert "bts-leaderboard" not in proc.stdout

    return {
        "unit_drift_alerts": len(alerts),
        "install_systemd_diff_stdout": proc.stdout.strip(),
        "leaderboard_timer_persistent": "Persistent=true" in (REPO_ROOT / "deploy" / "systemd" / "bts-leaderboard.timer").read_text(),
    }


def main() -> int:
    logs_dir = REVIEW_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cron_show_and_install": test_cron_show_and_install_both_intents(),
        "cron_cli_commands_verified": test_every_cron_command_exists_and_executes_dry_or_stubbed(),
        "cron_timing_and_heartbeat": test_cron_timing_collisions_and_heartbeat_restart_window(),
        "enter_cron_dm_matrix": test_enter_cron_dm_matrix(),
        "leaderboard_systemd_drift_blindspot": test_leaderboard_systemd_unit_drift_blindspot(),
    }
    out_path = logs_dir / "item3_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified Item 3 & 4 probes PASSED -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
