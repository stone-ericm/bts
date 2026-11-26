"""Item 5 probe: run all 12 required scratch changes (plus control and both 10a/10b variants) one at a time on isolated scratch copies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIG_SRC = REPO_ROOT / "src"

FOCUSED_TESTS = [
    "tests/test_wr_f2_saver_flag.py",
    "tests/test_f2_b1_stat_oserror_regression.py",
    "tests/test_contest_dated_cs4a.py",
    "tests/health/test_runner.py",
    "tests/health/test_contest_state.py",
]

MUTATIONS = (
    "0_control_unmodified",
    "1_drop_incident_key",
    "2_use_load_saver_state",
    "3_use_wall_clock_year",
    "4_warn_on_used_flag_too",
    "5_send_inside_existing_digest",
    "6_dispatch_mismatch_before_digest",
    "7_share_digest_status_file",
    "8_drop_recursion_error",
    "9_drop_oserror",
    "10a_put_guarded_used_block_before_not_earned",
    "10b_put_unguarded_flag_read_before_not_earned",
    "11_drop_require_message_id",
    "12_drop_default_receipt_fallback",
)


def apply_mutation(scratch_src: Path, name: str) -> None:
    cs_path = scratch_src / "bts" / "health" / "contest_state.py"
    rn_path = scratch_src / "bts" / "health" / "runner.py"
    cs_text = cs_path.read_text(encoding="utf-8")
    rn_text = rn_path.read_text(encoding="utf-8")

    if name == "0_control_unmodified":
        return
    elif name == "1_drop_incident_key":
        target = "                    incident_key=DATED_USED_MISMATCH,\n"
        assert target in cs_text
        cs_path.write_text(cs_text.replace(target, "", 1), encoding="utf-8")
    elif name == "2_use_load_saver_state":
        old = (
            "            from bts.saver_state import _read_saver_state\n\n"
            '            year = now.astimezone(ZoneInfo("America/New_York")).year\n'
            "            try:\n"
            "                flag = _read_saver_state(picks_dir, season=year, historical=True)"
        )
        new = (
            "            from bts.saver_state import load_saver_state\n\n"
            '            year = now.astimezone(ZoneInfo("America/New_York")).year\n'
            "            try:\n"
            "                flag = load_saver_state(picks_dir, season=year)"
        )
        assert old in cs_text
        cs_path.write_text(cs_text.replace(old, new, 1), encoding="utf-8")
    elif name == "3_use_wall_clock_year":
        old = '            year = now.astimezone(ZoneInfo("America/New_York")).year'
        new = '            year = datetime.now(ZoneInfo("America/New_York")).year'
        assert old in cs_text
        cs_path.write_text(cs_text.replace(old, new, 1), encoding="utf-8")
    elif name == "4_warn_on_used_flag_too":
        old = '            if flag.state in {"active", "not_earned"}:'
        new = '            if flag.state in {"active", "not_earned", "used"}:'
        assert old in cs_text
        cs_path.write_text(cs_text.replace(old, new, 1), encoding="utf-8")
    elif name == "5_send_inside_existing_digest":
        old = (
            "    if any(alert.incident_key == mismatch_key for alert in alerts):\n"
            "        digest_alerts = [alert for alert in alerts if alert.incident_key != mismatch_key]\n"
            "        digest_attention = [alert for alert in warn_attention if alert.incident_key != mismatch_key]\n"
            "        mismatch_attention = [alert for alert in warn_attention if alert.incident_key == mismatch_key]"
        )
        new = "    if False:\n        pass"
        assert old in rn_text
        rn_path.write_text(rn_text.replace(old, new, 1), encoding="utf-8")
    elif name == "6_dispatch_mismatch_before_digest":
        digest_call = (
            "    dispatch_dm_for_health_alerts(\n"
            "        digest_alerts,\n"
            "        dm_recipient,\n"
            "        warn_attention=digest_attention,\n"
            "        status_path=status_path,\n"
            "        # Dedup by the PROCESSED day, not the wall date: a post-midnight EOD\n"
            "        # for date D must not consume D+1's \"already sent today\" budget\n"
            "        # (round-2 review #5).\n"
            "        now_et_date=processed_date,\n"
            "    )\n"
        )
        mismatch_block = (
            "    if mismatch_attention:\n"
            '        suffix = ".dated_used_mismatch.json"\n'
            "        try:\n"
            "            mismatch_status = status_path.with_name(status_path.name + suffix)\n"
            "        except Exception:\n"
            "            mismatch_status = (\n"
            '                picks_dir.parent / "health_state" / ("health_dm_delivery_status.json" + suffix)\n'
            "            )\n"
            "        dispatch_dm_for_health_alerts(\n"
            "            [],\n"
            "            dm_recipient,\n"
            "            warn_attention=mismatch_attention,\n"
            "            status_path=mismatch_status,\n"
            "            now_et_date=processed_date,\n"
            "            require_message_id=True,\n"
            "        )\n"
        )
        assert digest_call + mismatch_block in rn_text
        rn_path.write_text(
            rn_text.replace(digest_call + mismatch_block, mismatch_block + digest_call, 1),
            encoding="utf-8",
        )
    elif name == "7_share_digest_status_file":
        old = "            status_path=mismatch_status,"
        new = "            status_path=status_path,"
        assert old in rn_text
        rn_path.write_text(rn_text.replace(old, new, 1), encoding="utf-8")
    elif name == "8_drop_recursion_error":
        old = "            except (OSError, ValueError, RecursionError):"
        new = "            except (OSError, ValueError):"
        assert old in cs_text
        cs_path.write_text(cs_text.replace(old, new, 1), encoding="utf-8")
    elif name == "9_drop_oserror":
        old = "            except (OSError, ValueError, RecursionError):"
        new = "            except (ValueError, RecursionError):"
        assert old in cs_text
        cs_path.write_text(cs_text.replace(old, new, 1), encoding="utf-8")
    elif name in ("10a_put_guarded_used_block_before_not_earned", "10b_put_unguarded_flag_read_before_not_earned"):
        not_earned_block = (
            '        if state.saver_state == "not_earned" and 10 <= state.streak <= 15:\n'
            "            if historical:\n"
            "                alerts.append(Alert(\n"
            '                    level="WARN", source=SAVER_SOURCE,\n'
            '                    message="The saver state in the dated observation contradicts its streak. The saver counted as spent.",\n'
            "                ))\n"
            "                return alerts\n"
            "            season = state.effective_date.year\n"
            '            base = (f"bts set-contest-streak --date {state.effective_date.isoformat()} "\n'
            '                    f"--streak {state.streak} --best-streak {state.best_streak}")\n'
            '            directory = f" --picks-dir {shlex.quote(str(picks_dir))}"\n'
            '            correction = (f"bts set-contest-streak --date {state.effective_date.isoformat()} "\n'
            '                          "--streak N --best-streak B --saver not_earned" + directory)\n'
            "            alerts.append(Alert(\n"
            '                level="WARN", source=SAVER_SOURCE,\n'
            '                message=(f"Streak Saver not_earned in dated observation {state.path}, season "\n'
            '                         f"{season}, contradicts streak {state.streak}; counting the saver as spent. "\n'
            '                         "Check MLB and run ONE correction to the observation. Earned and unused: "\n'
            '                         f"`{base} --saver active{directory}`. Already used: "\n'
            '                         f"`{base} --saver used{directory}`. Not earned: "\n'
            '                         f"`{correction}`, replacing N and B with the active and best streaks MLB "\n'
            '                         "shows. Changing the saver flag does not correct this observation."),\n'
            "            ))\n"
        )
        used_block = (
            '        if state.saver_state == "used":\n'
            "            from bts.saver_state import _read_saver_state\n\n"
            '            year = now.astimezone(ZoneInfo("America/New_York")).year\n'
            "            try:\n"
            "                flag = _read_saver_state(picks_dir, season=year, historical=True)\n"
            "            except (OSError, ValueError, RecursionError):\n"
            "                return alerts\n"
            '            if flag.state in {"active", "not_earned"}:\n'
            "                if historical:\n"
            "                    message = (\n"
            '                        "The dated observation recorded the saver as used. "\n'
            '                        f"At this check the flag for season {year} read {flag.state}."\n'
            "                    )\n"
            "                else:\n"
            "                    message = (\n"
            '                        f"Streak Saver on {state.effective_date.isoformat()} ET. "\n'
            '                        "The dated observation says the saver was used. "\n'
            '                        f"The flag for season {year} read {flag.state} at this check. "\n'
            '                        "A later automatic decision may count a saver as available. "\n'
            '                        "Verify the current season\'s saver and prior entries in MLB before any correction. "\n'
            '                        "Use the `bts season-contract status` command with the scheduler\'s picks directory "\n'
            '                        "as shown in checklist B8a. Follow B8a\'s saver reconciliation steps for any correction."\n'
            "                    )\n"
            "                alerts.append(Alert(\n"
            '                    level="WARN", source=SAVER_SOURCE, message=message,\n'
            "                    incident_key=DATED_USED_MISMATCH,\n"
            "                ))\n"
        )
        assert not_earned_block + used_block in cs_text
        if name == "10a_put_guarded_used_block_before_not_earned":
            cs_path.write_text(
                cs_text.replace(not_earned_block + used_block, used_block + not_earned_block, 1),
                encoding="utf-8",
            )
        else:
            unguarded_prefix = used_block.replace(
                '        if state.saver_state == "used":\n'
                "            from bts.saver_state import _read_saver_state\n\n"
                '            year = now.astimezone(ZoneInfo("America/New_York")).year\n'
                "            try:\n"
                "                flag = _read_saver_state(picks_dir, season=year, historical=True)\n"
                "            except (OSError, ValueError, RecursionError):\n"
                "                return alerts\n"
                '            if flag.state in {"active", "not_earned"}:\n',
                "        from bts.saver_state import _read_saver_state\n\n"
                '        year = now.astimezone(ZoneInfo("America/New_York")).year\n'
                "        try:\n"
                "            flag = _read_saver_state(picks_dir, season=year, historical=True)\n"
                "        except (OSError, ValueError, RecursionError):\n"
                "            return alerts\n"
                '        if state.saver_state == "used":\n'
                '            if flag.state in {"active", "not_earned"}:\n',
                1,
            )
            cs_path.write_text(
                cs_text.replace(not_earned_block + used_block, unguarded_prefix + not_earned_block, 1),
                encoding="utf-8",
            )
    elif name == "11_drop_require_message_id":
        old = "            require_message_id=True,\n"
        assert old in rn_text
        rn_path.write_text(rn_text.replace(old, "", 1), encoding="utf-8")
    elif name == "12_drop_default_receipt_fallback":
        old = (
            "        try:\n"
            "            mismatch_status = status_path.with_name(status_path.name + suffix)\n"
            "        except Exception:\n"
            "            mismatch_status = (\n"
            '                picks_dir.parent / "health_state" / ("health_dm_delivery_status.json" + suffix)\n'
            "            )"
        )
        new = "        mismatch_status = status_path.with_name(status_path.name + suffix)"
        assert old in rn_text
        rn_path.write_text(rn_text.replace(old, new, 1), encoding="utf-8")
    else:
        raise ValueError(f"Unknown mutation: {name}")


def run_pytest(env: dict[str, str], args: list[str], timeout_s: int = 25) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    tail = "\n".join(combined.splitlines()[-8:])
    return proc.returncode != 0, f"rc={proc.returncode}\n{tail}"


def main() -> int:
    rows = []
    with TemporaryDirectory(prefix="f2_r2_mut_") as tmp:
        tmp_root = Path(tmp)
        for name in MUTATIONS:
            mut_dir = tmp_root / name
            scratch_src = mut_dir / "src"
            shutil.copytree(ORIG_SRC / "bts", scratch_src / "bts")
            (scratch_src / "sitecustomize.py").write_text(
                f"import sys\n"
                f"from pathlib import Path\n"
                f"orig = Path({str(ORIG_SRC)!r}).resolve()\n"
                f"sys.path = [p for p in sys.path if Path(p or '.').resolve() != orig]\n"
                f"sys.meta_path = [f for f in sys.meta_path if 'editable' not in type(f).__name__.lower()]\n"
                f"sys.path.insert(0, {str(scratch_src.resolve())!r})\n",
                encoding="utf-8",
            )
            apply_mutation(scratch_src, name)

            env = {**os.environ, "PYTHONPATH": str(scratch_src.resolve()), "TZ": "America/New_York"}
            chk = subprocess.run(
                [sys.executable, "-c", "import bts; print(bts.__file__)"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=10, check=True,
            )
            bts_file = chk.stdout.strip()
            assert str(scratch_src.resolve()) in str(Path(bts_file).resolve())

            if name == "2_use_load_saver_state":
                # Check fast CS4a spy kill AND standalone test_wr_f2_saver_flag.py FlagReadTimeout kill!
                cs4a_killed, cs4a_detail = run_pytest(env, ["-x", "tests/test_contest_dated_cs4a.py", "-k", "without_flag_read"], timeout_s=15)
                f2_killed, f2_detail = run_pytest(env, ["-x", "tests/test_wr_f2_saver_flag.py", "-k", "fifo"], timeout_s=22)
                killed = cs4a_killed and f2_killed
                detail = f"cs4a_killed={cs4a_killed} ({cs4a_detail.splitlines()[-1]}); f2_fifo_killed={f2_killed} ({f2_detail.splitlines()[-1]})"
            else:
                killed, detail = run_pytest(env, ["-x", *FOCUSED_TESTS], timeout_s=25)

            status = "KILLED" if killed else "UNNOTICED"
            print(f"[{name}] -> {status}")
            for ln in detail.splitlines():
                print(f"    {ln}")
            rows.append({"mutation": name, "killed": killed, "detail": detail})

    print("\nSUMMARY_JSON:")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
