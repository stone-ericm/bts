"""Probe 1f: Run Part 1c shims cleanly (escaping `/home/bts` in bash substitution) and save JSON results."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[2]
OCT_DIR = REPO / ".carbon-review/wr-e/oct_619abda7"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bts import day_outcome


def main() -> None:
    findings = {}

    # 1. Checklist Row A5 on restored 619abda7 box (steps 2 and 5-7 without steps 1, 3, 4)
    with tempfile.TemporaryDirectory() as tmp:
        box_home = Path(tmp) / "home_bts"
        bts_dir = box_home / "projects/bts"
        bts_dir.mkdir(parents=True)
        os.symlink(OCT_DIR / "scripts", bts_dir / "scripts")
        os.symlink(OCT_DIR / "src", bts_dir / "src")
        (bts_dir / ".env").write_text("HEALTHCHECKS_PING_URL=https://hc-ping.com/fake\nBTS_BLUESKY_APP_PASSWORD=fake\n")
        (box_home / ".bts-orchestrator.toml").write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\nentry_intent = "enter"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        bin_dir = box_home / ".local/bin"
        bin_dir.mkdir(parents=True)
        crontab_store = box_home / "crontab.txt"
        uv_shim = bin_dir / "uv"
        uv_shim.write_text(
            f'#!/usr/bin/env bash\nshift\nshift\n'
            f'PYTHONPATH="{OCT_DIR / "src"}" exec "{REPO / ".venv/bin/python"}" -c "from bts.cli import cli; cli()" "$@"\n'
        )
        uv_shim.chmod(0o755)
        crontab_shim = bin_dir / "crontab"
        crontab_shim.write_text(
            f'#!/usr/bin/env bash\n'
            f'if [ "${{1:-}}" = "-l" ]; then\n'
            f'  if [ -f "{crontab_store}" ]; then cat "{crontab_store}"; exit 0; else echo "no crontab for bts" >&2; exit 1; fi\n'
            f'elif [ "${{1:-}}" = "-" ]; then cat > "{crontab_store}"; fi\n'
        )
        crontab_shim.chmod(0o755)

        env = {**os.environ, "HOME": str(box_home), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
        p_ei = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", "from bts.cli import cli; cli()", "entry-intent", "--config", str(box_home / ".bts-orchestrator.toml")],
            env={**env, "PYTHONPATH": str(OCT_DIR / "src")},
            capture_output=True, text=True, timeout=15,
        )
        p_cron = subprocess.run(
            ["bash", "-c", "cd ~/projects/bts && set -a && . ./.env && set +a && bash scripts/cron-setup-hetzner.sh show && bash scripts/cron-setup-hetzner.sh install"],
            env=env, capture_output=True, text=True, timeout=15,
        )
        installed_cron = crontab_store.read_text() if crontab_store.exists() else ""
        p_days = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", "from bts import strategy; print(strategy._days_remaining('2027-03-24'))"],
            env={**env, "PYTHONPATH": str(OCT_DIR / "src")},
            capture_output=True, text=True, timeout=15, check=True,
        )
        findings["repro_a5_on_619abda7"] = {
            "entry_intent_exit": p_ei.returncode,
            "entry_intent_stdout": p_ei.stdout.strip(),
            "cron_install_exit": p_cron.returncode,
            "has_hardcoded_2026_build": "--seasons 2026" in installed_cron,
            "missing_day_outcome": "day-outcome" not in installed_cron,
            "has_entry_checker": "check-pick-entered" in installed_cron,
            "october_2027_days_left": int(p_days.stdout.strip()),
        }

    # 2. Checklist Row B4 without .env exported on Linux
    with tempfile.TemporaryDirectory() as tmp:
        drill = Path(tmp) / "drill"
        picks_copy = drill / "picks"
        (picks_copy / "2026-09-10").mkdir(parents=True)
        cfg_file = Path(tmp) / "orch.toml"
        cfg_file.write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\nentry_intent = "enter"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        code_b4 = "\n".join([
            "import json, subprocess",
            "from unittest.mock import patch",
            "from click.testing import CliRunner",
            "from bts.cli import cli",
            "from bts import scheduler",
            "orig_run = subprocess.run",
            "def no_keychain(args, *a, **kw):",
            "    if args and args[0] == 'security': raise FileNotFoundError('security not on linux')",
            "    return orig_run(args, *a, **kw)",
            "with patch.object(subprocess, 'run', no_keychain), patch.object(scheduler, 'fetch_schedule', lambda d: [{'gamePk': 1, 'status': {'detailedState': 'Final'}}]):",
            f"    res = CliRunner().invoke(cli, ['day-outcome', '--date', '2026-09-10', '--config', '{cfg_file}', '--picks-dir', '{picks_copy}', '--expected-username', 'stonehengee', '--dm-recipient', 'stonehengee.bsky.social'])",
            "    print(json.dumps({'exit_code': res.exit_code, 'output': res.output}))",
        ])
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("BTS_BLUESKY")}
        clean_env["PYTHONPATH"] = str(REPO / "src")
        p_b4 = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", code_b4],
            env=clean_env, capture_output=True, text=True, timeout=15, check=True,
        )
        findings["repro_b4_missing_dotenv"] = json.loads(p_b4.stdout)

    # 3. INCIDENT.md §4 (`INCIDENT.md:129-131`) `install-systemd-hetzner.sh` while held
    with tempfile.TemporaryDirectory() as tmp:
        box_home = Path(tmp) / "home"
        systemd_dir = box_home / ".config/systemd/user"
        systemd_dir.mkdir(parents=True)
        mask_target = Path(tmp) / "null_sink"
        mask_target.write_text("")
        for u in ("bts-scheduler.service", "bts-dashboard.service", "bts-leaderboard.timer", "bts-leaderboard.service", "bts-lineup-collect.service"):
            os.symlink(mask_target, systemd_dir / u)
        shim_bin = Path(tmp) / "bin"
        shim_bin.mkdir()
        systemctl_offline = shim_bin / "systemctl"
        systemctl_offline.write_text('#!/usr/bin/env bash\necho "Failed to connect to bus: No medium found" >&2\nexit 1\n')
        systemctl_offline.chmod(0o755)
        p_inst_offline = subprocess.run(
            ["bash", str(REPO / "scripts/install-systemd-hetzner.sh")],
            env={**os.environ, "HOME": str(box_home), "PATH": f"{shim_bin}:{os.environ['PATH']}"},
            capture_output=True, text=True, timeout=15,
        )
        offline_sched_unmasked = not (systemd_dir / "bts-scheduler.service").is_symlink()

        for u in ("bts-scheduler.service", "bts-dashboard.service", "bts-lineup-collect.service"):
            (systemd_dir / u).unlink(missing_ok=True)
            os.symlink(mask_target, systemd_dir / u)
        systemctl_online = shim_bin / "systemctl"
        systemctl_online.write_text('#!/usr/bin/env bash\nexit 0\n')
        systemctl_online.chmod(0o755)
        p_inst_online = subprocess.run(
            ["bash", str(REPO / "scripts/install-systemd-hetzner.sh")],
            env={**os.environ, "HOME": str(box_home), "PATH": f"{shim_bin}:{os.environ['PATH']}"},
            capture_output=True, text=True, timeout=15,
        )
        findings["repro_incident_s4_install_while_held"] = {
            "offline_exit_code": p_inst_offline.returncode,
            "offline_stderr": p_inst_offline.stderr.strip(),
            "offline_scheduler_mask_destroyed": offline_sched_unmasked,
            "online_exit_code": p_inst_online.returncode,
            "scheduler_mask_destroyed": not (systemd_dir / "bts-scheduler.service").is_symlink(),
            "dashboard_mask_destroyed": not (systemd_dir / "bts-dashboard.service").is_symlink(),
            "writer_service_mask_destroyed": not (systemd_dir / "bts-lineup-collect.service").is_symlink(),
        }

    # 4. INCIDENT.md §3 (`INCIDENT.md:93`) `python scripts/capture_bts_cookies.py`
    p_py = subprocess.run(["bash", "-c", "command -v python"], capture_output=True, text=True, timeout=5)
    fake_pw = MagicMock()
    pw_mod = ModuleType("playwright")
    pw_sync = ModuleType("playwright.sync_api")
    pw_sync.sync_playwright = fake_pw
    with patch.dict(sys.modules, {"playwright": pw_mod, "playwright.sync_api": pw_sync}):
        import scripts.capture_bts_cookies as cap
        calls = []
        ctx = MagicMock()
        ctx.cookies.return_value = [{"name": "oktaid", "value": "123"}]
        browser = MagicMock()
        browser.new_context.return_value = ctx
        fake_pw.return_value.__enter__.return_value.chromium.launch.return_value = browser
        out_no_print, err_no_print = io.StringIO(), io.StringIO()
        with patch.object(sys, "platform", "darwin"), \
             patch.object(sys, "argv", ["capture_bts_cookies.py"]), \
             patch("builtins.input", return_value=""), \
             patch.object(subprocess, "run", lambda a, **kw: calls.append(a) or MagicMock(returncode=0)), \
             redirect_stdout(out_no_print), redirect_stderr(err_no_print):
            cap.main()
    findings["repro_incident_s3_cookies"] = {
        "bare_python_exists_on_mac": p_py.returncode == 0,
        "capture_without_print_stdout": out_no_print.getvalue(),
        "capture_stderr": err_no_print.getvalue().strip(),
        "keychain_calls": len(calls),
    }

    # 5. Full B8c deploy.yml shell body under shims (canary pass, canary fail, early sync stop)
    workflow_text = (REPO / ".github/workflows/deploy.yml").read_text()
    ssh_script = workflow_text.split("script: |\n", 1)[1]
    ssh_script = "\n".join(line[12:] if line.startswith("            ") else line for line in ssh_script.splitlines())

    deploy_outcomes = {}
    for scenario in ("canary_pass", "canary_fail", "early_sync_fail"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin_repo = root / "origin.git"
            box_home = root / "home_bts"
            box_repo = box_home / "projects/bts"
            (box_home / "projects").mkdir(parents=True)
            subprocess.run(["git", "init", "--bare", str(origin_repo)], check=True, capture_output=True, timeout=10)
            work = root / "work"
            subprocess.run(["git", "clone", str(origin_repo), str(work)], check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(work), "config", "user.email", "test@local"], check=True, timeout=5)
            subprocess.run(["git", "-C", str(work), "config", "user.name", "test"], check=True, timeout=5)
            (work / "rev.txt").write_text("october-619abda7\n")
            subprocess.run(["git", "-C", str(work), "add", "rev.txt"], check=True, timeout=5)
            subprocess.run(["git", "-C", str(work), "commit", "-m", "october"], check=True, capture_output=True, timeout=5)
            old_sha = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            (work / "rev.txt").write_text("season-2027-target\n")
            subprocess.run(["git", "-C", str(work), "add", "rev.txt"], check=True, timeout=5)
            subprocess.run(["git", "-C", str(work), "commit", "-m", "target2027"], check=True, capture_output=True, timeout=5)
            target_sha = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            subprocess.run(["git", "-C", str(work), "push", "origin", "HEAD:refs/heads/deploy"], check=True, capture_output=True, timeout=10)

            subprocess.run(["git", "clone", str(origin_repo), str(box_repo)], check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(box_repo), "reset", "--hard", old_sha], check=True, capture_output=True, timeout=5)
            (box_repo / "data").mkdir(parents=True)

            shims = root / "shims"
            shims.mkdir()
            state_log = root / "shim_events.log"
            (box_home / ".local/bin").mkdir(parents=True)
            uv_box = box_home / ".local/bin/uv"
            uv_box.write_text(
                f'#!/usr/bin/env bash\n'
                f'echo "uv $*" >> "{state_log}"\n'
                f'if [ "{scenario}" = "early_sync_fail" ] && [ "$1" = "sync" ]; then exit 1; fi\n'
                f'exit 0\n'
            )
            uv_box.chmod(0o755)

            for name, body in {
                "sleep": "#!/usr/bin/env bash\nexit 0\n",
                "id": "#!/usr/bin/env bash\necho 1000\n",
                "tailscale": "#!/usr/bin/env bash\necho 100.64.0.1\n",
                "curl": (
                    f'#!/usr/bin/env bash\n'
                    f'if [ "{scenario}" = "canary_fail" ]; then printf "500"; exit 0; else printf "200"; exit 0; fi\n'
                ),
                "systemctl": (
                    f'#!/usr/bin/env bash\n'
                    f'echo "systemctl $*" >> "{state_log}"\n'
                    f'if [[ "$*" == *"is-active --quiet"* ]]; then exit 0; fi\n'
                    f'if [[ "$*" == *"is-active"* ]]; then echo "active"; exit 0; fi\n'
                    f'exit 0\n'
                ),
                "sudo": (
                    f'#!/usr/bin/env bash\n'
                    f'while [[ "$1" == -* ]]; do\n'
                    f'  if [[ "$1" == "-u" || "$1" == "-Hu" ]]; then shift 2; else shift; fi\n'
                    f'done\n'
                    f'while [[ "$1" == *=* ]]; do export "$1"; shift; done\n'
                    f'HOME="{box_home}" GIT_CONFIG_GLOBAL="{box_home}/.gitconfig" exec "$@"\n'
                ),
            }.items():
                p = shims / name
                p.write_text(body)
                p.chmod(0o755)

            adapted_script = ssh_script.replace("/home/bts", str(box_home))
            proc = subprocess.run(
                ["bash", "-c", adapted_script],
                env={**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "TARGET_SHA": target_sha},
                capture_output=True, text=True, timeout=20,
            )
            final_sha = subprocess.run(
                ["git", "-C", str(box_repo), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            deploy_outcomes[scenario] = {
                "exit_code": proc.returncode,
                "final_equals_target": final_sha == target_sha,
                "final_equals_old": final_sha == old_sha,
                "canary_passed_printed": "CANARY PASSED" in proc.stdout,
                "rolled_back_printed": "Rolled back cleanly" in proc.stdout,
            }

    assert deploy_outcomes["canary_pass"] == {
        "exit_code": 0, "final_equals_target": True, "final_equals_old": False,
        "canary_passed_printed": True, "rolled_back_printed": False,
    }
    assert deploy_outcomes["canary_fail"] == {
        "exit_code": 1, "final_equals_target": False, "final_equals_old": True,
        "canary_passed_printed": False, "rolled_back_printed": True,
    }
    assert deploy_outcomes["early_sync_fail"] == {
        "exit_code": 1, "final_equals_target": True, "final_equals_old": False,
        "canary_passed_printed": False, "rolled_back_printed": False,
    }
    findings["deploy_shim_outcomes"] = deploy_outcomes

    # 6. B8c steps 5-6 & A7 revert with HEAD cron-setup-hetzner.sh
    with tempfile.TemporaryDirectory() as tmp:
        box_home = Path(tmp) / "home_bts"
        bts_dir = box_home / "projects/bts"
        bts_dir.mkdir(parents=True)
        os.symlink(REPO / "scripts", bts_dir / "scripts")
        os.symlink(REPO / "src", bts_dir / "src")
        (bts_dir / ".env").write_text("HEALTHCHECKS_PING_URL=https://hc-ping.com/ok\n")
        bin_dir = box_home / ".local/bin"
        bin_dir.mkdir(parents=True)
        crontab_store = box_home / "crontab.txt"
        uv_shim = bin_dir / "uv"
        uv_shim.write_text(
            f'#!/usr/bin/env bash\nshift\nshift\n'
            f'PYTHONPATH="{REPO / "src"}" exec "{REPO / ".venv/bin/python"}" -c "from bts.cli import cli; cli()" "$@"\n'
        )
        uv_shim.chmod(0o755)
        crontab_shim = bin_dir / "crontab"
        crontab_shim.write_text(
            f'#!/usr/bin/env bash\n'
            f'if [ "${{1:-}}" = "-l" ]; then\n'
            f'  if [ -f "{crontab_store}" ]; then cat "{crontab_store}"; exit 0; else echo "no crontab for bts" >&2; exit 1; fi\n'
            f'elif [ "${{1:-}}" = "-" ]; then cat > "{crontab_store}"; fi\n'
        )
        crontab_shim.chmod(0o755)
        env = {**os.environ, "HOME": str(box_home), "PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONPATH": str(REPO / "src")}

        (box_home / ".bts-orchestrator.toml").write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\nentry_intent = "enter"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        subprocess.run(
            ["bash", "-c", "cd ~/projects/bts && set -a && . ./.env && set +a && bash scripts/cron-setup-hetzner.sh show && bash scripts/cron-setup-hetzner.sh install"],
            env=env, capture_output=True, text=True, timeout=15, check=True,
        )
        cron_enter = [ln for ln in crontab_store.read_text().splitlines() if ln and not ln.startswith("#")]
        assert len(cron_enter) == 19

        (box_home / ".bts-orchestrator.toml").write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "private"\nentry_intent = "research"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        subprocess.run(
            ["bash", "-c", "cd ~/projects/bts && set -a && . ./.env && set +a && bash scripts/cron-setup-hetzner.sh install"],
            env=env, capture_output=True, text=True, timeout=15, check=True,
        )
        cron_res = [ln for ln in crontab_store.read_text().splitlines() if ln and not ln.startswith("#")]
        assert len(cron_res) == 18

        res_do = day_outcome.classify_day(
            date="2027-07-15", games=10, void_games=0,
            decision={"action": "single", "source": "mdp", "scoreable": True, "delivery_status": "delivered",
                      "primary": {"batter_name": "Steven Kwan"}},
            decision_sha256="abc", receipts=[], intent="research", expected_username="stonehengee",
        )
        assert res_do.state == "delivered_under_research" and not res_do.ok
    findings["head_cron_and_a7_revert"] = "PASS"

    out_path = REPO / ".carbon-review/wr-e/probe_1f_results.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
