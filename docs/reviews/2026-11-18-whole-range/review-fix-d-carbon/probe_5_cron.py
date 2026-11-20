#!/usr/bin/env python3
"""Item 5 probe: cron-setup-hetzner.sh rendering, Vixie cron % transform + /bin/sh -c replay, refusals, and test_entry_intent.py AST diff."""
from __future__ import annotations

import ast
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEAD_CRON = REPO / "scripts" / "cron-setup-hetzner.sh"
BASE_CRON = Path(__file__).resolve().parent / "base" / "scripts" / "cron-setup-hetzner.sh"
INSTALLER = REPO / "scripts" / "install-deploy-guard-hetzner.sh"
PYTHON = sys.executable


def extract_guard_source() -> str:
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def vixie_cron_transform(command_part: str) -> tuple[str, bool]:
    """Simulate Vixie cron's % handling: unescaped % splits command vs stdin; \\% becomes %."""
    out = []
    i = 0
    has_unescaped_percent = False
    while i < len(command_part):
        if command_part[i] == "\\" and i + 1 < len(command_part) and command_part[i + 1] == "%":
            out.append("%")
            i += 2
        elif command_part[i] == "%":
            has_unescaped_percent = True
            break
        else:
            out.append(command_part[i])
            i += 1
    return "".join(out), has_unescaped_percent


def setup_cron_env(root: Path, intent: str = "enter") -> dict:
    home = root / "home"
    proj = home / "projects" / "bts"
    proj.mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(parents=True, exist_ok=True)
    (proj / ".env").write_text(
        "BTS_SECRET_TOKEN=secret_val_42\n"
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.com/hb_tok\n"
    )
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)

    trace_file = root / "trace.jsonl"
    fail_mode_file = root / "fail_mode"
    fail_mode_file.write_text("")

    # uv shim: supports `run bts entry-intent` via real CLI, and records all cron invocations!
    uv_bin = home / ".local" / "bin" / "uv"
    uv_bin.write_text(
        f'#!{PYTHON}\n'
        'import fcntl, json, os, subprocess, sys\n'
        'args = sys.argv[1:]\n'
        'if args[:3] == ["run", "bts", "entry-intent"]:\n'
        f'    raise SystemExit(subprocess.call(["{PYTHON}", "-c", "from bts.cli import cli; cli()", *args[2:]])) \n'
        'lock_path = os.environ.get("WRD_WRITERS_LOCK", "")\n'
        'writer_locked = False\n'
        'if lock_path and os.path.exists(lock_path):\n'
        '    with open(lock_path, "rb") as lf:\n'
        '        try:\n'
        '            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
        '        except BlockingIOError:\n'
        '            writer_locked = True\n'
        'rec = {\n'
        '    "tool": "uv",\n'
        '    "argv": args,\n'
        '    "cwd": os.getcwd(),\n'
        '    "env_token": os.environ.get("BTS_SECRET_TOKEN"),\n'
        '    "writer_sh_lock_held": writer_locked,\n'
        '}\n'
        f'with open("{trace_file}", "a") as tf:\n'
        '    tf.write(json.dumps(rec) + "\\n")\n'
        'print("UV_STDOUT:" + " ".join(args))\n'
        'sys.stderr.write("UV_STDERR:" + " ".join(args) + "\\n")\n'
        f'fail_mode = open("{fail_mode_file}").read().strip()\n'
        'if fail_mode and fail_mode in " ".join(args):\n'
        '    raise SystemExit(17)\n'
        'raise SystemExit(0)\n'
    )
    uv_bin.chmod(0o755)

    shims = root / "shims"
    shims.mkdir(exist_ok=True)

    # GNU date shim for macOS replay
    date_shim = shims / "date"
    date_shim.write_text(
        f'#!{PYTHON}\n'
        'import sys\n'
        'args = sys.argv[1:]\n'
        'if args == ["-d", "yesterday", "+%Y-%m-%d"]: print("2027-07-14")\n'
        'elif args == ["-d", "7 days ago", "+%Y-%m-%d"]: print("2027-07-08")\n'
        'elif args == ["+%Y-%m-%d"]: print("2027-07-15")\n'
        'elif args == ["+%Y"]: print("2027")\n'
        'else: raise SystemExit(f"bad date args: {args}")\n'
    )
    date_shim.chmod(0o755)

    # Real flock shim using fcntl.flock for macOS
    flock_shim = shims / "flock"
    flock_shim.write_text(
        f'#!{PYTHON}\n'
        'import fcntl, os, sys\n'
        'assert sys.argv[1] == "-n", sys.argv\n'
        'lockfile = sys.argv[2]\n'
        'cmd = sys.argv[3:]\n'
        'fd = os.open(lockfile, os.O_CREAT | os.O_RDWR, 0o644)\n'
        'try:\n'
        '    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
        'except BlockingIOError:\n'
        '    raise SystemExit(1)\n'
        'os.set_inheritable(fd, True)\n'
        'os.execv(cmd[0], cmd)\n'
    )
    flock_shim.chmod(0o755)

    # Python3 shim for heartbeat line
    py3_shim = shims / "python3"
    py3_shim.write_text(
        f'#!{PYTHON}\n'
        'import json, os, sys\n'
        'rec = {"tool": "python3", "argv": sys.argv[1:], "cwd": os.getcwd(), "env_token": os.environ.get("BTS_SECRET_TOKEN")}\n'
        f'with open("{trace_file}", "a") as tf: tf.write(json.dumps(rec) + "\\n")\n'
        'print("HB_STDOUT")\n'
    )
    py3_shim.chmod(0o755)

    curl_shim = shims / "curl"
    curl_shim.write_text(
        f'#!{PYTHON}\n'
        'import json, sys\n'
        f'with open("{trace_file}", "a") as tf: tf.write(json.dumps({{"tool": "curl", "argv": sys.argv[1:]}}) + "\\n")\n'
    )
    curl_shim.chmod(0o755)

    crontab_file = root / "crontab.txt"
    crontab_shim = shims / "crontab"
    crontab_shim.write_text(
        f'#!/bin/sh\nf="{crontab_file}"\n'
        'case "$1" in\n'
        '  -l) [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
        '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n'
        '  *) exit 98 ;;\n'
        'esac\n'
    )
    crontab_shim.chmod(0o755)

    guard_dir = root / "guard"
    guard_dir.mkdir(mode=0o755)
    for n in ("writers.lock", "transaction.lock", "state.lock"):
        (guard_dir / n).touch(mode=0o644)
    guard_bin = shims / "bts-deploy-hold"
    guard_bin.write_text(f"#!{PYTHON} -I\n" + extract_guard_source())
    guard_bin.chmod(0o755)

    cfg = home / ".bts-orchestrator.toml"
    if intent == "enter":
        cfg.write_text(
            '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
    else:
        cfg.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')

    return {
        "home": home, "proj": proj, "shims": shims, "guard_dir": guard_dir,
        "guard_bin": guard_bin, "crontab_file": crontab_file,
        "trace_file": trace_file, "fail_mode_file": fail_mode_file,
    }


def replay_cron_line(line: str, ctx: dict) -> tuple[int, list[dict]]:
    parts = line.split(maxsplit=5)
    assert len(parts) == 6, line
    cmd_raw = parts[5]
    cmd, had_unescaped_pct = vixie_cron_transform(cmd_raw)
    assert not had_unescaped_pct, f"Unescaped % in cron line: {line}"
    # Route hardcoded /usr/bin/flock, /home/bts/.check-results.lock, and /usr/bin/python3 to local shims
    cmd = cmd.replace("/usr/bin/flock", str(ctx["shims"] / "flock"))
    cmd = cmd.replace("/home/bts/.check-results.lock", str(ctx["home"] / ".check-results.lock"))
    cmd = cmd.replace("/usr/bin/python3", str(ctx["shims"] / "python3"))

    ctx["trace_file"].unlink(missing_ok=True)
    res = subprocess.run(
        ["/bin/sh", "-c", cmd],
        cwd="/tmp",
        env={
            "HOME": str(ctx["home"]),
            "PATH": f"{ctx['shims']}:/usr/bin:/bin",
            "BTS_TEST_DEPLOY_GUARD_DIR": str(ctx["guard_dir"]),
            "WRD_WRITERS_LOCK": str(ctx["guard_dir"] / "writers.lock"),
        },
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
    )
    traces = [json.loads(ln) for ln in ctx["trace_file"].read_text().splitlines()] if ctx["trace_file"].exists() else []
    return res.returncode, traces


def check_test_entry_intent_unchanged() -> dict:
    base_src = subprocess.run(
        ["git", "show", "b128854d:tests/test_entry_intent.py"],
        cwd=REPO, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    ).stdout
    head_src = (REPO / "tests" / "test_entry_intent.py").read_text()
    base_tree = ast.parse(base_src)
    head_tree = ast.parse(head_src)
    base_funcs = {n.name: ast.unparse(n) for n in base_tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    head_funcs = {n.name: ast.unparse(n) for n in head_tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    changed = [name for name, body in base_funcs.items() if head_funcs.get(name) != body]
    added = sorted(set(head_funcs) - set(base_funcs))
    return {"base_test_count": len(base_funcs), "changed_existing_tests": changed, "added_tests": added}


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="p5_", dir=Path(__file__).resolve().parent))
    try:
        ctx = setup_cron_env(work_dir, "enter")
        env = {
            "HOME": str(ctx["home"]),
            "PATH": f"{ctx['shims']}:/usr/bin:/bin",
            "HEALTHCHECKS_PING_URL": "https://hc-ping.com/boxping",
            "BTS_DEPLOY_HOLD_BIN": str(ctx["guard_bin"]),
            "BTS_TEST_DEPLOY_GUARD_DIR": str(ctx["guard_dir"]),
        }

        base_show = subprocess.run(["/bin/bash", str(BASE_CRON), "show"], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        head_show = subprocess.run(["/bin/bash", str(HEAD_CRON), "show"], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        assert base_show.returncode == 0 and head_show.returncode == 0
        base_hdr, base_body = base_show.stdout.split("Would install:\n")
        head_hdr, head_body = head_show.stdout.split("Would install:\n")
        assert base_hdr == head_hdr, "show header changed!"

        base_lines = [ln for ln in base_body.splitlines() if ln and not ln.startswith("#")]
        head_lines = [ln for ln in head_body.splitlines() if ln and not ln.startswith("#")]
        assert len(base_lines) == len(head_lines) == 19  # 18 + check-pick-entered

        # Replay EVERY line on BASE vs HEAD (clear) and compare cwd, env, dates, argv, and log redirects!
        gated_indices = []
        for idx, (b_ln, h_ln) in enumerate(zip(base_lines, head_lines)):
            assert b_ln.split()[:5] == h_ln.split()[:5], f"Schedule changed on line {idx}"
            b_rc, b_tr = replay_cron_line(b_ln, ctx)
            h_rc, h_tr = replay_cron_line(h_ln, ctx)
            assert b_rc == h_rc == 0, (idx, b_rc, h_rc)
            assert len(b_tr) == len(h_tr), (idx, b_tr, h_tr)

            if f"{ctx['guard_bin']} run " in h_ln:
                gated_indices.append(idx)
                # Gated writer: inner command is identical (no --no-sync), and writer_sh_lock_held MUST be True!
                for bt, ht in zip(b_tr, h_tr):
                    assert bt["argv"] == ht["argv"], (idx, bt, ht)
                    assert bt["cwd"] == ht["cwd"] == str(ctx["proj"].resolve())
                    assert bt["env_token"] == ht["env_token"] == "secret_val_42"
                    assert ht["writer_sh_lock_held"] is True, f"Line {idx} did not inherit writers.lock!"
            elif "scripts/check_heartbeat.py" in h_ln:
                assert h_tr[0]["tool"] == "python3"
                assert h_tr[0]["argv"] == [
                    "scripts/check_heartbeat.py", "--heartbeat-path", "data/.heartbeat",
                    "--deploy-hold", "/var/lib/bts-deploy/hold", "--ping-url", "https://hc-ping.com/hb_tok",
                ]
                assert h_tr[0]["cwd"] == str(ctx["proj"].resolve())
            elif h_tr[0]["tool"] == "uv":
                # Unheld uv line: identical except `--no-sync` inserted right after `run`!
                assert h_tr[0]["argv"] == ["run", "--no-sync", *b_tr[0]["argv"][1:]], (idx, b_tr, h_tr)
                assert h_tr[0]["cwd"] == str(ctx["proj"].resolve())
                assert h_tr[0]["env_token"] == "secret_val_42"

        assert len(gated_indices) == 8

        # Replay ALL 8 gated lines + unheld lines while HELD!
        (ctx["guard_dir"] / "hold").write_text("held-sentinel")
        for idx in gated_indices:
            rc, tr = replay_cron_line(head_lines[idx], ctx)
            assert rc == 75 and tr == [], f"Gated line {idx} ran while held!"
        assert "bts-deploy-hold:" in (ctx["home"] / "logs" / "cron.log").read_text()
        (ctx["guard_dir"] / "hold").unlink()

        # Verify 01:00 grader's flock singleton!
        grader_line = head_lines[0]
        assert "check-results --date" in grader_line
        lock_path = ctx["home"] / ".check-results.lock"
        with lock_path.open("wb") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rc_singleton, tr_singleton = replay_cron_line(grader_line, ctx)
            assert rc_singleton == 1 and tr_singleton == [], "Grader ran despite .check-results.lock held!"

        # Verify 03:00 chain's pull/build gate and preview/R2 independence!
        chain_line, = (ln for ln in head_lines if "nightly-preview" in ln)
        chain_cases = {}
        for fail_sub in ("data pull", "data build", "bts preview", "sync-to-r2"):
            ctx["fail_mode_file"].write_text(fail_sub)
            rc_c, tr_c = replay_cron_line(chain_line, ctx)
            chain_cases[fail_sub] = {"rc": rc_c, "commands_run": [" ".join(t["argv"][1:]) for t in tr_c]}
        ctx["fail_mode_file"].write_text("")
        assert chain_cases["data pull"]["commands_run"] == ["bts data pull --start 2027-07-08 --end 2027-07-15"]
        assert chain_cases["data build"]["commands_run"] == [
            "bts data pull --start 2027-07-08 --end 2027-07-15",
            "bts data build --seasons 2027",
        ]
        assert chain_cases["bts preview"]["commands_run"] == [
            "bts data pull --start 2027-07-08 --end 2027-07-15",
            "bts data build --seasons 2027",
            "bts preview",
            "bts data sync-to-r2",
        ]
        assert chain_cases["sync-to-r2"]["commands_run"] == [
            "bts data pull --start 2027-07-08 --end 2027-07-15",
            "bts data build --seasons 2027",
            "bts preview",
            "bts data sync-to-r2",
        ]

        # Verify install refusals leave crontab untouched!
        orig_cron = "15 4 * * * echo keep_me\n"
        ctx["crontab_file"].write_text(orig_cron)
        refusal_checks = {}

        # (a) writers.lock symlink
        (ctx["guard_dir"] / "writers.lock").unlink()
        (ctx["guard_dir"] / "writers.lock").symlink_to(ctx["proj"] / ".env")
        r_sym = subprocess.run(["/bin/bash", str(HEAD_CRON), "install"], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        assert r_sym.returncode == 1 and ctx["crontab_file"].read_text() == orig_cron
        refusal_checks["symlink_writers_lock"] = r_sym.returncode
        (ctx["guard_dir"] / "writers.lock").unlink()
        (ctx["guard_dir"] / "writers.lock").touch(mode=0o644)

        # (b) extra newline from guard version
        bad_guard = work_dir / "bad_guard"
        bad_guard.write_text('#!/bin/sh\nprintf "bts-deploy-hold 1\\n\\n"\n')
        bad_guard.chmod(0o755)
        r_nl = subprocess.run(["/bin/bash", str(HEAD_CRON), "install"], env={**env, "BTS_DEPLOY_HOLD_BIN": str(bad_guard)}, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        assert r_nl.returncode == 1 and ctx["crontab_file"].read_text() == orig_cron
        refusal_checks["extra_newline_version"] = r_nl.returncode

        ast_diff = check_test_entry_intent_unchanged()
        assert ast_diff["changed_existing_tests"] == []

        report = {
            "total_lines_replayed": len(head_lines),
            "gated_lines_verified": len(gated_indices),
            "show_header_identical": True,
            "grader_flock_singleton_verified": True,
            "chain_0300_independence": chain_cases,
            "install_refusals_preserve_crontab": refusal_checks,
            "test_entry_intent_ast_diff": ast_diff,
        }
        out_path = Path(__file__).resolve().parent / "probe_5_cron.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
