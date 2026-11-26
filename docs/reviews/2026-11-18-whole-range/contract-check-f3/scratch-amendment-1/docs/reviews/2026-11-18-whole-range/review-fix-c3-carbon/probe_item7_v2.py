"""Item 7 runner v2 using wallclock_plugin_v2."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET_XML

import probe_item7_tests_and_mutants as v1

REVIEW_DIR = v1.REVIEW_DIR
REPO_ROOT = v1.REPO_ROOT
VENV_PYTHON = v1.VENV_PYTHON
NEW_TEST_FILE = v1.NEW_TEST_FILE
MUTANTS = v1.MUTANTS


def _run_one_pytest(
    label: str,
    src_dir: Path,
    test_files: list[str],
    xml_path: Path,
    *,
    wall_iso: str | None = None,
) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TZ"] = "America/New_York"
    env["PYTHONPATH"] = f"{src_dir}:{REVIEW_DIR}"
    env["BTS_ASSERT_SRC_PREFIX"] = str(src_dir)
    if wall_iso:
        env["BTS_PINNED_WALL_ISO"] = wall_iso
    else:
        env.pop("BTS_PINNED_WALL_ISO", None)

    cmd = [
        "/usr/bin/nice",
        "-n",
        "19",
        VENV_PYTHON,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "wallclock_plugin_v2",
        "-q",
        f"--junitxml={xml_path}",
        *test_files,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
    if not xml_path.exists():
        raise RuntimeError(f"pytest {label} exited {proc.returncode} without xml:\n{proc.stdout}\n{proc.stderr}")

    tree = ET_XML.parse(xml_path)
    passed: list[str] = []
    failed: list[dict] = []
    for tc in tree.findall(".//testcase"):
        node = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
        fail_el = tc.find("failure")
        err_el = tc.find("error")
        if fail_el is not None:
            failed.append({"node": node, "kind": "failure", "message": (fail_el.attrib.get("message") or "").splitlines()[0][:220]})
        elif err_el is not None:
            failed.append({"node": node, "kind": "error", "message": (err_el.attrib.get("message") or "").splitlines()[0][:220]})
        elif tc.find("skipped") is None:
            passed.append(node)

    return {
        "label": label,
        "returncode": proc.returncode,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "passed": passed,
        "failed": failed,
    }


def main() -> int:
    report: dict = {"base_red": {}, "wall_clocks": {}, "mutants_new_file_only": {}, "survivors_against_streak_suite": {}}

    with TemporaryDirectory(dir=REVIEW_DIR, prefix="_item7_v2_") as scratch_str:
        scratch = Path(scratch_str)

        # 1. Base RED run on c2039541
        base_dir = scratch / "base_c2039541"
        base_dir.mkdir()
        with subprocess.Popen(["git", "archive", "c2039541", "src"], cwd=REPO_ROOT, stdout=subprocess.PIPE) as arch:
            assert arch.stdout is not None
            with tarfile.open(fileobj=arch.stdout, mode="r|") as tf:
                tf.extractall(base_dir, filter="data")
            assert arch.wait() == 0

        report["base_red"] = _run_one_pytest(
            "base_c2039541",
            base_dir / "src",
            [NEW_TEST_FILE],
            scratch / "base.xml",
        )

        # 2. Wall clocks in 2027 and 2028
        for wc_label, wc_iso in [
            ("wall_2027_midseason", "2027-07-15T14:00:00-04:00"),
            ("wall_2027_nye", "2027-12-31T23:55:00-05:00"),
            ("wall_2028_new_year", "2028-01-01T00:05:00-05:00"),
            ("wall_2028_midseason", "2028-07-15T14:00:00-04:00"),
        ]:
            res = _run_one_pytest(
                wc_label,
                REPO_ROOT / "src",
                [
                    NEW_TEST_FILE,
                    "tests/test_lightgbm_optional.py",
                    "tests/test_streak_replay.py",
                    "tests/test_streak_applied_dates.py",
                    "tests/test_streak_season_cs2.py",
                    "tests/test_streak_season_cs2_review.py",
                    "tests/test_streak_season_cs2b_review.py",
                ],
                scratch / f"{wc_label}.xml",
                wall_iso=wc_iso,
            )
            report["wall_clocks"][wc_label] = {
                "returncode": res["returncode"],
                "passed_count": res["passed_count"],
                "failed_count": res["failed_count"],
                "failed": res["failed"],
            }

        # 3. Mutants against tests/test_wr_c3_reconcile.py
        for m_name, rel_file, target, replacement in MUTANTS:
            m_src = scratch / f"mut_{m_name}" / "src"
            shutil.copytree(REPO_ROOT / "src", m_src)
            fpath = m_src / rel_file
            orig = fpath.read_text()
            cnt = orig.count(target)
            assert cnt == 1, f"Mutant {m_name}: expected count 1 in {rel_file}, got {cnt}"
            fpath.write_text(orig.replace(target, replacement, 1))

            m_res = _run_one_pytest(
                m_name,
                m_src,
                [NEW_TEST_FILE],
                scratch / f"mut_{m_name}.xml",
            )
            killed = m_res["failed_count"] > 0
            report["mutants_new_file_only"][m_name] = {
                "status": "KILLED" if killed else "SURVIVED",
                "failed_count": m_res["failed_count"],
                "passed_count": m_res["passed_count"],
                "killing_tests": [f["node"] for f in m_res["failed"][:8]],
            }
            if not killed:
                s_res = _run_one_pytest(
                    f"{m_name}_streak_suite",
                    m_src,
                    ["tests/test_streak_season_cs2b_review.py"],
                    scratch / f"mut_{m_name}_cs2b.xml",
                )
                report["survivors_against_streak_suite"][m_name] = {
                    "status": "KILLED" if s_res["failed_count"] > 0 else "SURVIVED",
                    "failed_count": s_res["failed_count"],
                    "killing_tests": [f["node"] for f in s_res["failed"][:5]],
                }

    out_path = REVIEW_DIR / "item7_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "base_red": {"passed": report["base_red"]["passed_count"], "failed": report["base_red"]["failed_count"], "passed_nodes": report["base_red"]["passed"]},
        "wall_clocks": report["wall_clocks"],
        "mutants_new_file_only": report["mutants_new_file_only"],
        "survivors_against_streak_suite": report["survivors_against_streak_suite"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
