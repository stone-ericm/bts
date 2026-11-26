"""Unified pytest runner for Item 10 (base controls, 2027/2028 clock harness, and 16 scratch mutations).

Designed to be invoked inside ONE `~/bts-runs/yield_run.sh` wrapper session so all
pytest child runs execute sequentially one at a time without overlapping locks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
PY = str(REPO / ".venv/bin/python")

TOUCHED_AND_ADDED = [
    "tests/health/test_contest_state.py",
    "tests/test_contest_dated_cs4a.py",
    "tests/test_contest_state.py",
    "tests/test_contest_state_cs1_review.py",
    "tests/test_contest_state_decoder_limits.py",
    "tests/test_contest_state_defensive_reads.py",
    "tests/test_saver_cs3.py",
    "tests/test_saver_cs3_review.py",
    "tests/test_season_contract_cs4b1.py",
    "tests/test_season_contract_cs4b1_edges.py",
    "tests/test_season_contract_cs4b2.py",
    "tests/test_season_contract_cs4b2_edges.py",
    ".codex-review/cs4b2-review/adopt/test_cs4b2_review_regressions.py",
]


def _base_env(src_override: Path | None = None, fake_clock: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = tempfile.gettempdir()
    parts = []
    if src_override is not None:
        parts.append(str(src_override.resolve()))
    parts.append(str(WORK.resolve()))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if fake_clock is not None:
        env["FAKE_WALL_CLOCK_ISO"] = fake_clock
    else:
        env.pop("FAKE_WALL_CLOCK_ISO", None)
    return env


def _verify_import_path(src_dir: Path) -> None:
    env = _base_env(src_override=src_dir)
    out = subprocess.check_output(
        [PY, "-c", "import bts, pathlib; print(pathlib.Path(bts.__file__).resolve())"],
        cwd=str(REPO),
        env=env,
        text=True,
    ).strip()
    expected_prefix = str(src_dir.resolve())
    assert out.startswith(expected_prefix), f"PYTHONPATH override failed: {out} vs {expected_prefix}"


def _run_pytest(
    targets: list[str],
    *,
    src_override: Path | None = None,
    fake_clock: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    env = _base_env(src_override=src_override, fake_clock=fake_clock)
    cmd = [PY, "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    if fake_clock is not None:
        cmd.extend(["-p", "clock_plugin"])
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(targets)
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, check=False)
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    tail = combined.splitlines()[-3:] if combined else []
    return {
        "returncode": proc.returncode,
        "summary_lines": tail,
        "output": combined,
    }


def main() -> None:
    report: dict = {}

    # Verify PYTHONPATH cleanly overrides editable .pth
    _verify_import_path(WORK / "base/src")

    # Family A: adoptable regression suite on HEAD, and on base (must fail on base!)
    adopt_target = [".codex-review/cs4b2-review/adopt/test_cs4b2_review_regressions.py"]
    res_adopt_head = _run_pytest(adopt_target)
    assert res_adopt_head["returncode"] == 0, res_adopt_head["output"]
    res_adopt_base = _run_pytest(adopt_target, src_override=WORK / "base/src")
    assert res_adopt_base["returncode"] != 0, "adopt tests unexpectedly passed on base!"
    report["family_a_adopt"] = {
        "head_rc": res_adopt_head["returncode"],
        "head_summary": res_adopt_head["summary_lines"],
        "base_rc": res_adopt_base["returncode"],
        "base_summary": res_adopt_base["summary_lines"],
    }
    print("Family A (adopt):", report["family_a_adopt"]["head_summary"], flush=True)

    # Family B: cs4b2 and cs4b2_edges against base/src (0bbfc8d4) — classify passed vs failed
    res_cs4b2_base = _run_pytest(
        ["tests/test_season_contract_cs4b2.py", "tests/test_season_contract_cs4b2_edges.py"],
        src_override=WORK / "base/src",
        extra_args=["-ra", "--tb=no"],
    )
    passed_on_base_edges = _run_pytest(
        ["tests/test_season_contract_cs4b2_edges.py"],
        src_override=WORK / "base/src",
        extra_args=["-v", "--tb=no"],
    )
    edges_passed_nodes = [
        line.split(" PASSED")[0].strip()
        for line in passed_on_base_edges["output"].splitlines()
        if " PASSED" in line
    ]
    edges_failed_nodes = [
        line.split(" FAILED")[0].strip()
        for line in passed_on_base_edges["output"].splitlines()
        if " FAILED" in line
    ]
    report["family_b_base_run"] = {
        "combined_rc": res_cs4b2_base["returncode"],
        "combined_summary": res_cs4b2_base["summary_lines"][-2:],
        "edges_passed_nodes": edges_passed_nodes,
        "edges_failed_nodes": edges_failed_nodes,
    }
    print("Family B (base):", report["family_b_base_run"]["combined_summary"], flush=True)

    # Family C: Every added or touched file under clocks in 2027 and 2028
    report["family_c_clocks"] = {}
    for label, iso in [
        ("clock_2027", "2027-07-15T12:00:00-04:00"),
        ("clock_2028", "2028-07-15T12:00:00-04:00"),
    ]:
        rc_clock = _run_pytest(TOUCHED_AND_ADDED, fake_clock=iso)
        report["family_c_clocks"][label] = {
            "returncode": rc_clock["returncode"],
            "summary": rc_clock["summary_lines"],
        }
        print(f"Family C ({label}):", rc_clock["summary_lines"], flush=True)
        assert rc_clock["returncode"] == 0, rc_clock["output"]

    # Family D: Scratch mutations (NEVER touching tracked src/)
    mut_dir = WORK / "mutated_src"
    mutations = [
        (
            "M1_remove_refusal_call",
            "bts/contest_state.py",
            "    if require_contest_state and not (",
            "    if False and require_contest_state and not (",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M2_invert_dated_bypass",
            "bts/contest_state.py",
            "            contest is not None and dated_observation_in_effect(contest, now_dt)):",
            "            False):",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3a_drop_reason1_corrupt",
            "bts/season_contract.py",
            '    if state.state == "corrupt":',
            "    if False:",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3b_drop_reason2_legacy",
            "bts/season_contract.py",
            '    elif selected is not None and selected.path.name == "contest_streak.manual.json":',
            "    elif False:",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3c_drop_reason3_inactive",
            "bts/season_contract.py",
            '    elif state.state in {"absent", "other_season", "revoked"}:',
            "    elif False:",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3d_drop_reason4_none",
            "bts/season_contract.py",
            "    elif selected is None:\n        kind = 4\n        reason = \"Live decision refused: there is no automatic observation for this activation.\"",
            "    elif selected is None:\n        return None",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3e_drop_reason5_schema",
            "bts/season_contract.py",
            "        if selected.schema_version != AUTO_SCHEMA:",
            "        if False:",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3f_drop_reason5_season",
            "bts/season_contract.py",
            "        elif selected.season != year:",
            "        elif False:",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M3g_drop_reason5_id",
            "bts/season_contract.py",
            '        elif selected.activation_id != state.snapshot["activation_id"]:',
            "        elif False:",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M4a_skip_health_append_missing",
            "bts/health/contest_state.py",
            "        append_refusal()\n        return alerts",
            "        return alerts",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M4b_skip_health_append_final",
            "bts/health/contest_state.py",
            "    append_refusal()\n    return alerts",
            "    return alerts",
            ["tests/test_season_contract_cs4b2.py"],
            True,
        ),
        (
            "M5_revoke_continue_after_failed_manual",
            "bts/cli.py",
            '        click.echo(f"This invocation wrote the revoked switch; manual file was not set aside: {exc}; "\n                   "saver flag was not attempted", err=True)\n        raise SystemExit(1)',
            '        manual = f"failed: {exc}"',
            ["tests/test_season_contract_cs4b2.py", "tests/test_season_contract_cs4b2_edges.py"],
            True,
        ),
        (
            "M6_revoke_rename_replaces_target",
            "bts/cli.py",
            '        try:\n            target.lstat()\n        except FileNotFoundError:\n            pass\n        else:\n            raise FileExistsError(f"target already exists: {target}")',
            "        pass",
            ["tests/test_season_contract_cs4b2.py", "tests/test_season_contract_cs4b2_edges.py"],
            True,
        ),
        # Adversarial false-green probes!
        (
            "M7_drop_corrupt_check_num_and_text",
            "bts/season_contract.py",
            '        reason = f"Live decision refused: corrupt season switch, check {state.check}: {state.error}."',
            '        reason = "Live decision refused: corrupt season switch."',
            ["tests/test_season_contract_cs4b2.py", "tests/test_season_contract_cs4b2_edges.py"],
            False,  # Check whether existing tests catch it or stay GREEN!
        ),
        (
            "M8_drop_other_season_file_year",
            "bts/season_contract.py",
            '        detail = f" (file season {state.snapshot[\'season\']})" if state.state == "other_season" else ""',
            '        detail = ""',
            ["tests/test_season_contract_cs4b2.py", "tests/test_season_contract_cs4b2_edges.py"],
            False,
        ),
        (
            "M9_delete_health_missing_source_date",
            "bts/health/contest_state.py",
            "        if state.source_date is None:",
            "        if False:",
            ["tests/health/test_contest_state.py"],
            False,
        ),
    ]

    mut_results = {}
    for name, rel_file, old_s, new_s, suite_targets, fast_fail in mutations:
        if mut_dir.exists():
            shutil.rmtree(mut_dir)
        shutil.copytree(REPO / "src", mut_dir)
        _verify_import_path(mut_dir)
        target_file = mut_dir / rel_file
        src_text = target_file.read_text(encoding="utf-8")
        assert old_s in src_text, f"target snippet not found for {name}"
        target_file.write_text(src_text.replace(old_s, new_s, 1), encoding="utf-8")

        extra = ["-x", "--tb=no"] if fast_fail else ["--tb=no"]
        r_author = _run_pytest(suite_targets, src_override=mut_dir, extra_args=extra)
        entry = {
            "author_tests_killed": r_author["returncode"] != 0,
            "author_summary": r_author["summary_lines"][-1] if r_author["summary_lines"] else "",
        }
        if not fast_fail:
            r_adopt = _run_pytest(adopt_target, src_override=mut_dir, extra_args=["-x", "--tb=no"])
            entry["adopt_tests_killed"] = r_adopt["returncode"] != 0
            entry["adopt_summary"] = r_adopt["summary_lines"][-1] if r_adopt["summary_lines"] else ""
        mut_results[name] = entry
        print(f"Mutation {name}: {entry}", flush=True)

    if mut_dir.exists():
        shutil.rmtree(mut_dir)

    report["family_d_mutations"] = mut_results
    out_json = WORK / "item10_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ALL ITEM 10 FAMILIES COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
