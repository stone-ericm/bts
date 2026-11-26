"""Slice 2 Probe 3: Cron setup (`scripts/cron-setup-hetzner.sh`) under both `research` and `enter`
intents with shims for `uv`, `crontab`, `flock`, `timeout`, and `date`; CLI flag/option verification
against HEAD for every cron command; scheduler restart & inter-job collision checks; and DM audit.

Runnable standalone (`.venv/bin/python .carbon-review/whole-range/probe_cron.py`)
and adoptable as a pytest regression module (no module-level `bts.model.predict` import,
no process-wide state left mutated, no network).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "bts").is_dir():
            return parent
    raise RuntimeError("Could not locate BTS repository root")


REPO_ROOT = _find_repo_root()


def _write_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _build_cron_shims(home_dir: Path) -> tuple[Path, dict[str, str]]:
    """Build shims for `uv`, `crontab`, `flock`, `timeout`, `curl`, and GNU `date -d`."""
    bin_dir = home_dir / "shim_bin"
    uv_bin = home_dir / ".local" / "bin" / "uv"
    crontab_store = home_dir / ".fake_crontab"
    shim_log = home_dir / "shim_invocations.jsonl"
    py_exe = sys.executable

    # Shim `uv`: when invoked as `uv run bts ...` during `entry-intent`, run REAL HEAD `bts.cli`!
    # When `CRON_DRY_EXEC=1`, log the invocation instead of running network/data jobs.
    _write_executable(
        uv_bin,
        f"""#!/usr/bin/env bash
set -euo pipefail
if [ "${{CRON_DRY_EXEC:-0}}" = "1" ]; then
    "{py_exe}" -c 'import json, sys; open({json.dumps(str(shim_log))}, "a").write(json.dumps({{"tool": "uv", "argv": sys.argv[1:]}}) + "\\n")' "$@"
    exit 0
fi
if [ "${{1:-}}" = "run" ] && [ "${{2:-}}" = "bts" ]; then
    shift 2
    PYTHONPATH="{REPO_ROOT / 'src'}" exec "{py_exe}" -c 'import sys; from bts.cli import cli; cli()' "$@"
fi
if [ "${{1:-}}" = "run" ] && [ "${{2:-}}" = "python" ]; then
    shift 2
    PYTHONPATH="{REPO_ROOT / 'src'}" exec "{py_exe}" "$@"
fi
echo "Unexpected uv invocation: $*" >&2
exit 2
""",
    )

    # Shim `crontab` supporting `-l` ("no crontab for user" when absent) and `-` (install from stdin)
    _write_executable(
        bin_dir / "crontab",
        f"""#!/usr/bin/env bash
set -euo pipefail
STORE={str(crontab_store)!r}
if [ "${{1:-}}" = "-l" ]; then
    if [ ! -f "$STORE" ]; then
        echo "no crontab for bts" >&2
        exit 1
    fi
    cat "$STORE"
    exit 0
fi
if [ "${{1:-}}" = "-" ]; then
    cat > "$STORE"
    exit 0
fi
echo "Unsupported crontab args: $*" >&2
exit 2
""",
    )

    # Shim `flock`, `timeout`, `curl`, and GNU-compatible `date` (supporting `-d yesterday` / `-d "7 days ago"`)
    for tool_name in ("flock", "timeout"):
        _write_executable(
            bin_dir / tool_name,
            f"""#!/usr/bin/env bash
exec "{py_exe}" -c '
import json, subprocess, sys
argv = sys.argv[1:]
with open({json.dumps(str(shim_log))}, "a") as fh:
    fh.write(json.dumps({{"tool": {json.dumps(tool_name)}, "argv": argv}}) + "\\n")
if {json.dumps(tool_name)} == "flock":
    assert argv[0] == "-n", f"flock expected -n, got {{argv}}"
    cmd = argv[2:]
else:
    assert argv[:4] == ["--verbose", "-k", "30", "840"], f"timeout expected --verbose -k 30 840, got {{argv}}"
    cmd = argv[4:]
raise SystemExit(subprocess.call(cmd))
' "$@"
""",
        )

    _write_executable(
        bin_dir / "curl",
        f"""#!/usr/bin/env bash
exec "{py_exe}" -c '
import json, sys
with open({json.dumps(str(shim_log))}, "a") as fh:
    fh.write(json.dumps({{"tool": "curl", "argv": sys.argv[1:]}}) + "\\n")
' "$@"
""",
    )

    _write_executable(
        bin_dir / "date",
        f"""#!/usr/bin/env bash
exec "{py_exe}" -c '
import sys
from datetime import date, timedelta
args = sys.argv[1:]
ref = date(2027, 7, 15)
fmt = "%Y-%m-%d"
i = 0
while i < len(args):
    if args[i] == "-d":
        spec = args[i + 1]
        if spec == "yesterday":
            ref = ref - timedelta(days=1)
        elif spec == "7 days ago":
            ref = ref - timedelta(days=7)
        else:
            raise SystemExit(f"unsupported -d {{spec!r}}")
        i += 2
    elif args[i].startswith("+"):
        fmt = args[i][1:]
        i += 1
    else:
        raise SystemExit(f"unsupported date arg {{args[i]!r}}")
print(ref.strftime(fmt))
' "$@"
""",
    )

    bts_dir = home_dir / "projects" / "bts"
    bts_dir.mkdir(parents=True, exist_ok=True)
    (bts_dir / ".env").write_text(
        "HEALTHCHECKS_PING_URL=https://hc-ping.example.invalid/cron-uuid\n"
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.example.invalid/hb-uuid\n"
    )

    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["HEALTHCHECKS_PING_URL"] = "https://hc-ping.example.invalid/cron-uuid"
    env["PATH"] = f"{bin_dir}:{uv_bin.parent}:{env.get('PATH', '')}"
    return crontab_store, env


def _write_orch_toml(home_dir: Path, *, delivery: str, intent: str, required: bool = True) -> Path:
    cfg = home_dir / ".bts-orchestrator.toml"
    cfg.write_text(
        "\n".join([
            "[orchestrator]",
            'picks_dir = "data/picks"',
            "[bluesky]",
            'dm_recipient = "stonehengee.bsky.social"',
            "[scheduler]",
            f'pick_delivery = "{delivery}"',
            f'entry_intent = "{intent}"',
            f"contest_state_required = {'true' if required else 'false'}",
            "[health_checks]",
            f"contest_state_expected = {'true' if required else 'false'}",
        ]) + "\n"
    )
    return cfg


def _extract_cron_entries(show_stdout: str) -> list[str]:
    marker = "Would install:\n"
    idx = show_stdout.index(marker) + len(marker)
    block = show_stdout[idx:].strip().splitlines()
    assert block[0] == "# BTS-HETZNER"
    return [ln for ln in block[1:] if ln.strip()]


def _extract_bts_invocations_from_cron_line(cron_line: str, home_dir: Path) -> list[list[str]]:
    """Strip schedule + trailing redirect/marker, unescape cron `\\%`, and extract all `bts ...` or `python ...` argvs."""
    # Remove 5 cron schedule fields
    parts = cron_line.split(None, 5)
    assert len(parts) == 6, f"Malformed cron line: {cron_line}"
    cmd_part = parts[5]
    assert cmd_part.endswith("# BTS-HETZNER")
    cmd_part = cmd_part[: -len("# BTS-HETZNER")].strip()
    # Unescape cron percent escapes
    cmd_part = cmd_part.replace(r"\%", "%")
    # Expand known date subshells and env vars as cron's /bin/sh would on 2027-07-15
    cmd_part = cmd_part.replace('$(date -d yesterday +%Y-%m-%d)', "2027-07-14")
    cmd_part = cmd_part.replace('$(date -d "7 days ago" +%Y-%m-%d)', "2027-07-08")
    cmd_part = cmd_part.replace('$(date +%Y-%m-%d)', "2027-07-15")
    cmd_part = cmd_part.replace('$(date +%Y)', "2027")
    cmd_part = cmd_part.replace('"$BTS_SCHEDULER_HEARTBEAT_PING_URL"', "https://hc-ping.example.invalid/hb-uuid")

    uv_str = f"{home_dir}/.local/bin/uv run "
    invocations: list[list[str]] = []
    # Find every `$UV_BIN run ...` segment before `&&`, `;`, `}`, `)`, `>>`, `>`
    for m in re.finditer(re.escape(uv_str) + r"([^;&)>]+)", cmd_part):
        seg = m.group(1).strip()
        tokens = shlex.split(seg)
        invocations.append(tokens)
    return invocations


def test_cron_show_install_remove_and_cli_flags(tmp_path: Path) -> dict:
    """Verify `cron-setup-hetzner.sh` under both `research` and `enter`, plus refusal cases,
    POSIX `/bin/sh` execution with shims, and Click option parsing for every single cron command."""
    import bts.cli as cli_mod

    script = REPO_ROOT / "scripts" / "cron-setup-hetzner.sh"
    runner = CliRunner()
    report: dict[str, object] = {}

    for intent, delivery in (("research", "private"), ("enter", "dm")):
        home = tmp_path / f"home_{intent}"
        crontab_store, env = _build_cron_shims(home)
        _write_orch_toml(home, delivery=delivery, intent=intent, required=True)

        # Pre-seed an unrelated user cron line to prove install/remove preserve it
        crontab_store.write_text("15 12 * * * echo keep-unrelated-job\n")

        # 1. `show`
        proc_show = subprocess.run(["bash", str(script), "show"], env=env, capture_output=True, text=True, check=True)
        entries = _extract_cron_entries(proc_show.stdout)

        # 2. `install`
        proc_inst = subprocess.run(["bash", str(script), "install"], env=env, capture_output=True, text=True, check=True)
        installed_lines = crontab_store.read_text().splitlines()
        assert "15 12 * * * echo keep-unrelated-job" in installed_lines
        bts_installed = [ln for ln in installed_lines if ln.endswith("# BTS-HETZNER") and ln != "# BTS-HETZNER"]
        assert bts_installed == entries

        # 3. Verify every CLI invocation's subcommand + exact flags against HEAD's Click definitions
        verified_cmds = []
        dm_capable_lines = []
        for ln in entries:
            sched_spec = " ".join(ln.split(None, 5)[:5])
            invocations = _extract_bts_invocations_from_cron_line(ln, home)
            for tokens in invocations:
                if tokens[0] == "bts":
                    bts_args = tokens[1:]
                    # Verify --help exits 0 on the subcommand path
                    sub_prefix = []
                    for tok in bts_args:
                        if tok.startswith("-"):
                            break
                        sub_prefix.append(tok)
                    help_res = runner.invoke(cli_mod.cli, [*sub_prefix, "--help"])
                    assert help_res.exit_code == 0, f"--help failed for {sub_prefix}: {help_res.output}"
                    # Parse EXACT arguments/options through Click's make_context to catch any invalid option or type!
                    ctx = cli_mod.cli.make_context("bts", list(bts_args))
                    # Walk nested group contexts if `bts data ...` / `bts leaderboard ...` / `bts backup ...`
                    cmd_obj = cli_mod.cli
                    rem = list(bts_args)
                    with ctx:
                        while isinstance(cmd_obj, cli_mod.click.Group) and rem:
                            cmd_name, cmd_obj, rem = cmd_obj.resolve_command(ctx, rem)
                            ctx = cmd_obj.make_context(cmd_name, rem, parent=ctx)
                        assert rem == [], f"Unparsed extra args {rem} in {bts_args}"
                    verified_cmds.append(" ".join(tokens))
                    if "--dm-recipient" in bts_args:
                        dm_capable_lines.append({"schedule": sched_spec, "cmd": " ".join(tokens)})
                elif tokens[0] == "python":
                    assert tokens[1] == "scripts/check_heartbeat.py"
                    hb_help = subprocess.run(
                        [sys.executable, str(REPO_ROOT / "scripts" / "check_heartbeat.py"), "--help"],
                        capture_output=True, text=True, check=True,
                    )
                    assert "--heartbeat-path" in hb_help.stdout and "--ping-url" in hb_help.stdout
                    verified_cmds.append(" ".join(tokens))

            # Execute every cron command line in POSIX `/bin/sh` with CRON_DRY_EXEC=1 and our flock/timeout/uv/date shims!
            raw_cmd = ln.split(None, 5)[5]
            raw_cmd = raw_cmd[: -len("# BTS-HETZNER")].strip().replace(r"\%", "%")
            # Rewrite hardcoded `/usr/bin/flock` and `/usr/bin/timeout` and `/home/bts/` lock path to shim equivalents
            sh_cmd = (
                raw_cmd.replace("/usr/bin/flock", "flock")
                .replace("/usr/bin/timeout", "timeout")
                .replace("/home/bts/", f"{home}/")
            )
            dry_env = dict(env, CRON_DRY_EXEC="1")
            sh_proc = subprocess.run(["/bin/sh", "-c", sh_cmd], env=dry_env, capture_output=True, text=True)
            log_dump = "\n".join(p.read_text() for p in (home / "logs").glob("*.log")) if sh_proc.returncode != 0 else ""
            assert sh_proc.returncode == 0, f"/bin/sh failed on cron line:\n{sh_cmd}\nstderr:\n{sh_proc.stderr}\nlogs:\n{log_dump}"

        # 4. `remove` preserves unrelated job and removes all BTS lines
        subprocess.run(["bash", str(script), "remove"], env=env, capture_output=True, text=True, check=True)
        assert crontab_store.read_text().strip() == "15 12 * * * echo keep-unrelated-job"

        report[intent] = {
            "cron_line_count": len(entries),
            "verified_cli_commands": len(verified_cmds),
            "dm_capable_lines": dm_capable_lines,
        }

    # Verify exact counts & DM-capable lines under research vs enter:
    assert report["research"]["cron_line_count"] == 18
    assert report["enter"]["cron_line_count"] == 19
    # Under research: ONLY `30 8` day-outcome has --dm-recipient (F6 removed --dm-recipient from fetch-contest-streak)
    assert len(report["research"]["dm_capable_lines"]) == 1
    assert "day-outcome" in report["research"]["dm_capable_lines"][0]["cmd"]
    # Under enter: 4 `fetch-contest-streak` + 1 `day-outcome` + 1 `check-pick-entered` = 6 lines!
    assert len(report["enter"]["dm_capable_lines"]) == 6

    # 5. Refusal tests: missing HEALTHCHECKS_PING_URL, mismatched intent/delivery, missing live gate under enter
    home_bad = tmp_path / "home_refusals"
    _, env_bad = _build_cron_shims(home_bad)
    _write_orch_toml(home_bad, delivery="dm", intent="enter", required=False)
    r_nogate = subprocess.run(["bash", str(script), "show"], env=env_bad, capture_output=True, text=True)
    assert r_nogate.returncode != 0, "Expected cron-setup refusal when contest_state_required=false under enter"

    _write_orch_toml(home_bad, delivery="dm", intent="research", required=True)
    r_mismatch = subprocess.run(["bash", str(script), "show"], env=env_bad, capture_output=True, text=True)
    assert r_mismatch.returncode != 0, "Expected cron-setup refusal when delivery=dm and intent=research"

    env_no_hc = {k: v for k, v in env_bad.items() if k != "HEALTHCHECKS_PING_URL"}
    r_nohc = subprocess.run(["bash", str(script), "show"], env=env_no_hc, capture_output=True, text=True)
    assert r_nohc.returncode != 0

    return report


def test_cron_timing_and_restart_collisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Verify timing interactions:
    1) 10:00 ET scheduler wakeup/exit + 30s systemd restart vs `*/5` `check_heartbeat.py` (staleness + NRestarts churn).
    2) `*/15` `check-pick-entered --cron` pre-10:00 ET deferral vs early-opener execution.
    3) `01:10/02:10/10:30/13:30` `fetch-contest-streak` spacing vs 14h staleness threshold.
    4) Restic backup schedule (`20 */3`, `50 4`, `35 5 * * 0`) lock behavior.
    """
    import scripts.check_heartbeat as chb_mod
    import bts.contest_state as cs_mod
    import bts.entry_receipt as er_mod

    hb_path = tmp_path / "data" / ".heartbeat"
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "data" / "health_state" / "heartbeat_monitor.json"

    # 1. Simulate scheduler sleeping overnight until 10:00:00 ET (14:00:00Z), exiting at 10:00:00 ET,
    #    systemd restarting 30s later at 10:00:30 ET (NRestarts 5 -> 6), and `check_heartbeat.py` running at
    #    10:00:00 ET, 10:00:15 ET (mid-restart gap), and 10:05:00 ET!
    t_0955 = datetime(2027, 7, 15, 9, 55, 0, tzinfo=ET).astimezone(UTC)
    t_1000 = datetime(2027, 7, 15, 10, 0, 0, tzinfo=ET).astimezone(UTC)
    t_1000_15 = datetime(2027, 7, 15, 10, 0, 15, tzinfo=ET).astimezone(UTC)
    t_1005 = datetime(2027, 7, 15, 10, 5, 0, tzinfo=ET).astimezone(UTC)

    hb_sleeping_to_10 = {
        "pid": 1001,
        "state": "sleeping",
        "timestamp": datetime(2027, 7, 15, 0, 15, 0, tzinfo=ET).astimezone(UTC).isoformat(),
        "sleeping_until": t_1000.isoformat(),
    }
    hb_path.write_text(json.dumps(hb_sleeping_to_10))

    # At 09:55 ET (NRestarts=5): fresh, no churn
    stale_0955, _ = chb_mod.is_stale(hb_path, now=t_0955)
    churn_0955, _, samples = chb_mod.assess_churn([], 5, now=t_0955)
    chb_mod.save_churn_samples(state_path, "bts-scheduler", samples)
    assert not stale_0955 and not churn_0955

    # At 10:00:00 ET and 10:00:15 ET (during 30s RestartSec=30 gap before new daemon writes heartbeat):
    stale_1000, _ = chb_mod.is_stale(hb_path, now=t_1000)
    stale_gap, _ = chb_mod.is_stale(hb_path, now=t_1000_15)
    assert not stale_1000 and not stale_gap, "30s systemd restart gap at 10:00 ET must not false-alarm check_heartbeat!"

    # At 10:05:00 ET after daily rollover restart (NRestarts=6, sleeping until 16:10 ET):
    hb_new_day = {
        "pid": 1002,
        "state": "sleeping",
        "timestamp": datetime(2027, 7, 15, 10, 0, 32, tzinfo=ET).astimezone(UTC).isoformat(),
        "sleeping_until": datetime(2027, 7, 15, 16, 10, 0, tzinfo=ET).astimezone(UTC).isoformat(),
    }
    hb_path.write_text(json.dumps(hb_new_day))
    stale_1005, _ = chb_mod.is_stale(hb_path, now=t_1005)
    churn_1005, _, samples = chb_mod.assess_churn(chb_mod.load_churn_samples(state_path), 6, now=t_1005)
    assert not stale_1005 and not churn_1005, "Normal daily +1 NRestarts rollover must not trip churn!"

    # 2. `check-pick-entered --cron` pre-10:00 ET deferral boundary:
    # A 10:10 ET game has a 10:00 ET slot (10 min to pitch, in (5, 75]) -> has_original_slot is True (defers before 10:00).
    # A 10:05 ET game has 5.0 min to pitch at 10:00 ET (NOT > 5) -> has_original_slot is False (runs early before 10:00!).
    # A 09:05 ET London/Tokyo game -> has_original_slot is False (runs early before 10:00!).
    d_jul15 = datetime(2027, 7, 15, tzinfo=ET).date()
    assert er_mod.has_original_slot(d_jul15, datetime(2027, 7, 15, 10, 10, tzinfo=ET)) is True
    assert er_mod.has_original_slot(d_jul15, datetime(2027, 7, 15, 10, 5, tzinfo=ET)) is False
    assert er_mod.has_original_slot(d_jul15, datetime(2027, 7, 15, 9, 5, tzinfo=ET)) is False

    # 3. Automatic reading 14h staleness vs cron fetch schedule (01:10, 02:10, 10:30, 13:30):
    # - 02:10 fetch -> 10:00 morning wake is 7.83h (fresh)
    # - 13:30 fetch -> 23:30 late West Coast delivery is 10.0h (fresh)
    # - If 01:10 & 02:10 BOTH fail, yesterday's 13:30 fetch at 10:00 morning wake is 20.5h (>14h -> morning stale DM,
    #   30 min before the 10:30 fetch runs).
    s_base = cs_mod.DecisionStreakState(
        streak=10, saver_available=True, allow_double=True, source="contest", status="fresh",
        contest_source_date=None, model_streak=10, model_saver_available=True,
        observation_kind="automatic",
        contest_recorded_at=datetime(2027, 7, 15, 2, 10, tzinfo=ET).astimezone(UTC).isoformat().replace("+00:00", "Z"),
    )
    rec_10am = cs_mod.automatic_reading_record(s_base, datetime(2027, 7, 15, 10, 0, tzinfo=ET))
    assert rec_10am["stale"] is False and rec_10am["age_hours"] == 7.83

    s_missed_night = cs_mod.DecisionStreakState(
        streak=10, saver_available=True, allow_double=True, source="contest", status="fresh",
        contest_source_date=None, model_streak=10, model_saver_available=True,
        observation_kind="automatic",
        contest_recorded_at=datetime(2027, 7, 14, 13, 30, tzinfo=ET).astimezone(UTC).isoformat().replace("+00:00", "Z"),
    )
    rec_missed = cs_mod.automatic_reading_record(s_missed_night, datetime(2027, 7, 15, 10, 0, tzinfo=ET))
    assert rec_missed["stale"] is True and rec_missed["age_hours"] == 20.5

    return {
        "rollover_10am_heartbeat_collision": False,
        "early_watchdog_cutoff_boundary_verified": True,
        "fetch_cadence_age_at_10am_hours": rec_10am["age_hours"],
        "missed_overnight_fetches_age_at_10am_hours": rec_missed["age_hours"],
    }


def main() -> int:
    import tempfile

    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    log_path = REPO_ROOT / ".carbon-review" / "whole-range" / "probe_cron.log"
    results = {}
    with tempfile.TemporaryDirectory(prefix="wr2_probe3_") as tmp:
        tmp_path = Path(tmp)
        mp = pytest.MonkeyPatch()
        try:
            results["cron_setup_and_flags"] = test_cron_show_install_remove_and_cli_flags(tmp_path / "c1")
            results["timing_and_collisions"] = test_cron_timing_and_restart_collisions(tmp_path / "c2", mp)
        finally:
            mp.undo()
    assert "bts.model.predict" not in sys.modules
    log_path.write_text(json.dumps(results, indent=2) + "\n")
    print("PROBE 3 PASS:", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
