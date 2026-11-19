"""Item 6 probe:
1. Focused one-process pytest run on HEAD (aa61bb43): tests/test_wr_c3_*.py + streak, reconcile,
   scoring-lock, test_lightgbm_optional.py, test_web_saver.py.
2. RED runs of all new tests (tests/test_wr_c3_*.py) on R1 (c70effe9) and Base (c2039541),
   recording which fail and why.
3. Process wall-clock runs in 2027 (midseason + NYE) and 2028 (New Year + midseason).
4. Scratch-copy mutation sweep (the 5 requested mutants + 2 adversarial mutants), run one at a time
   under `/usr/bin/nice -n 19`.
"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET_XML


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENV_PY = str(REPO / ".venv/bin/python")

NEW_TESTS = [
    "tests/test_wr_c3_corpus.py",
    "tests/test_wr_c3_delivery.py",
    "tests/test_wr_c3_reconcile.py",
]

STREAK_RECONCILE_LOCK_TESTS = [
    "tests/test_reconcile_cutoff.py",
    "tests/test_reconcile_receipt.py",
    "tests/test_scoring_lock.py",
    "tests/test_streak_applied_dates.py",
    "tests/test_streak_replay.py",
    "tests/test_streak_season_cs2.py",
    "tests/test_streak_season_cs2_review.py",
    "tests/test_streak_season_cs2b_review.py",
    "tests/test_web_streak_subtitle.py",
]

FOCUSED_ONE_PROCESS_TESTS = [
    *NEW_TESTS,
    *STREAK_RECONCILE_LOCK_TESTS,
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

WALL_CLOCK_TESTS = [
    *NEW_TESTS,
    *STREAK_RECONCILE_LOCK_TESTS,
    "tests/test_lightgbm_optional.py",
]

# The 5 explicitly requested mutations + 2 adversarial mutations
MUTANTS = [
    (
        "M1_drop_scheduler_lock_check",
        "bts/picks.py",
        '            if state["pick_locked"]:\n                return "scheduler state marks the pick locked"',
        '            if False and state["pick_locked"]:\n                return "scheduler state marks the pick locked"',
    ),
    (
        "M2_missing_chosen_on_treated_as_preview",
        "bts/picks.py",
        '    if decision is None and (daily.chosen_on is None or daily.chosen_on["required"]):\n        return "no decision or preview stamp"',
        '    if decision is None and (daily.chosen_on is not None and daily.chosen_on["required"]):\n        return "no decision or preview stamp"',
    ),
    (
        "M3_non_scoreable_decision_overrides_delivery_signal",
        "bts/picks.py",
        '    if delivery_signal:\n        if decision is not None:\n            return "skip record contradicts a delivery"',
        '    if delivery_signal and decision is None:\n        if decision is not None:\n            return "skip record contradicts a delivery"',
    ),
    (
        "M4_validate_terminal_picks_strictly_again",
        "bts/picks.py",
        '            if data is not None:\n                result = data.get("result")\n                # Terminal outcomes predate delivery and scheduler metadata. Use\n                # the legacy hit decoder and let a miss or void stand on its own.\n                if result == "hit":\n                    daily = _daily_pick_from_data(data)\n                    streak, saver = _apply_streak_day(\n                        streak, saver, True, streak_increment_for_resolved_hit(daily)\n                    )\n                    continue\n                if result == "miss":\n                    streak, saver = _apply_streak_day(streak, saver, False, 0)\n                    continue\n                if result == "void":\n                    continue\n                daily = _validate_reconcile_pick(data, day)',
        '            if data is not None:\n                daily = _validate_reconcile_pick(data, day)\n                if daily.result in _TERMINAL_RESULTS:\n                    if daily.result != "void":\n                        streak, saver = _apply_streak_day(\n                            streak, saver, daily.result == "hit",\n                            streak_increment_for_resolved_hit(daily),\n                        )\n                    continue',
    ),
    (
        "M5_skip_when_decision_unreadable",
        "bts/picks.py",
        '        except _REPLAY_READ_ERRORS:\n            return None, (day, "unreadable or invalid decision record")',
        '        except _REPLAY_READ_ERRORS:\n            continue',
    ),
    (
        "M6_drop_uri_and_delivered_at_type_guard",
        "bts/picks.py",
        '    if any(value is not None and not isinstance(value, str) for value in (\n        daily.bluesky_uri, daily.delivered_at,\n    )):\n        return "unreadable or invalid pick file"',
        '    if False:\n        return "unreadable or invalid pick file"',
    ),
    (
        "M7_nonatomic_write_streak_when_unmocked",
        "bts/picks.py",
        'def _write_streak(path, streak, saver, applied, *, season, legacy=None):\n    data = {\n        "streak": streak,\n        "saver_available": saver,\n        "updated": datetime.now(timezone.utc).isoformat(),\n        "applied_dates": applied,\n    }\n    if season is not None:\n        data["season"] = season\n    elif legacy is not None:\n        data["season"] = legacy["season"]\n    atomic_write_text(path, json.dumps(data))',
        '_ORIG_ATOMIC = atomic_write_text\n\n\ndef _write_streak(path, streak, saver, applied, *, season, legacy=None):\n    data = {\n        "streak": streak,\n        "saver_available": saver,\n        "updated": datetime.now(timezone.utc).isoformat(),\n        "applied_dates": applied,\n    }\n    if season is not None:\n        data["season"] = season\n    elif legacy is not None:\n        data["season"] = legacy["season"]\n    if atomic_write_text is _ORIG_ATOMIC:\n        Path(path).write_text(json.dumps(data))\n    else:\n        atomic_write_text(path, json.dumps(data))',
    ),
]


def _archive_src(rev: str, dest: Path) -> Path:
    raw = subprocess.check_output(["git", "archive", rev, "src"], cwd=REPO, timeout=20)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(raw)) as tf:
        tf.extractall(dest, filter="data")
    return dest / "src"


def _run_pytest(
    label: str,
    src_dir: Path,
    test_files: list[str],
    xml_path: Path,
    log_path: Path,
    *,
    wall_iso: str | None = None,
    timeout: int = 90,
) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TZ"] = "America/New_York"
    env["MPLCONFIGDIR"] = "/tmp"
    env["PYTHONPATH"] = f"{src_dir}:{HERE}"
    env["BTS_ASSERT_SRC_PREFIX"] = str(src_dir)
    if wall_iso:
        env["BTS_PINNED_WALL_ISO"] = wall_iso
    else:
        env.pop("BTS_PINNED_WALL_ISO", None)

    cmd = [
        "/usr/bin/nice", "-n", "19",
        VENV_PY, "-m", "pytest",
        "-p", "no:cacheprovider",
        "-p", "r2_pytest_plugin",
        "-q",
        f"--junitxml={xml_path}",
        *test_files,
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    log_path.write_text(f"$ {' '.join(cmd)}\nexit={proc.returncode}\n\n{proc.stdout}\n{proc.stderr}")
    if not xml_path.exists():
        raise RuntimeError(f"{label} produced no junitxml (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    tree = ET_XML.parse(xml_path)
    passed = []
    skipped = []
    failed = []
    for tc in tree.findall(".//testcase"):
        node = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
        f_el = tc.find("failure")
        e_el = tc.find("error")
        s_el = tc.find("skipped")
        if f_el is not None:
            msg = (f_el.attrib.get("message") or f_el.text or "").splitlines()[0][:200]
            failed.append({"node": node, "kind": "failure", "msg": msg})
        elif e_el is not None:
            msg = (e_el.attrib.get("message") or e_el.text or "").splitlines()[0][:200]
            failed.append({"node": node, "kind": "error", "msg": msg})
        elif s_el is not None:
            skipped.append(node)
        else:
            passed.append(node)

    return {
        "label": label,
        "returncode": proc.returncode,
        "passed_count": len(passed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "failed": failed,
        "passed": passed,
    }


def _summarize_failures_by_file(failed: list[dict]) -> dict:
    by_cls = Counter()
    samples = {}
    for item in failed:
        cls = item["node"].split("::")[0]
        by_cls[cls] += 1
        samples.setdefault(cls, []).append(f"{item['node'].split('::')[-1]}: {item['msg']}")
    return {"counts_by_module": dict(by_cls), "sample_reasons": {k: v[:5] for k, v in samples.items()}}


def main() -> int:
    logs_dir = HERE / "pytest_logs"
    logs_dir.mkdir(exist_ok=True)
    report: dict = {}

    with TemporaryDirectory(dir=HERE, prefix="_item6_") as td:
        scratch = Path(td)

        # 1. Focused one-process run on HEAD (aa61bb43)
        focused = _run_pytest(
            "focused_head_aa61bb43",
            REPO / "src",
            FOCUSED_ONE_PROCESS_TESTS,
            scratch / "focused.xml",
            logs_dir / "focused_head.log",
            timeout=120,
        )
        report["focused_one_process"] = {
            "returncode": focused["returncode"],
            "passed": focused["passed_count"],
            "skipped": focused["skipped_count"],
            "failed": focused["failed_count"],
            "failures": focused["failed"],
        }

        # 2. RED on R1 (c70effe9) and Base (c2039541)
        r1_src = _archive_src("c70effe9", scratch / "r1_c70effe9")
        r1_red = _run_pytest(
            "red_on_r1_c70effe9",
            r1_src,
            NEW_TESTS,
            scratch / "r1_red.xml",
            logs_dir / "red_r1_c70effe9.log",
            timeout=120,
        )
        report["red_on_c70effe9"] = {
            "passed": r1_red["passed_count"],
            "failed": r1_red["failed_count"],
            "summary": _summarize_failures_by_file(r1_red["failed"]),
            "all_failed": r1_red["failed"],
        }

        base_src = _archive_src("c2039541", scratch / "base_c2039541")
        base_red = _run_pytest(
            "red_on_base_c2039541",
            base_src,
            NEW_TESTS,
            scratch / "base_red.xml",
            logs_dir / "red_base_c2039541.log",
            timeout=120,
        )
        report["red_on_c2039541"] = {
            "passed": base_red["passed_count"],
            "failed": base_red["failed_count"],
            "summary": _summarize_failures_by_file(base_red["failed"]),
            "all_failed": base_red["failed"],
        }

        # 3. Wall clocks in 2027 and 2028
        report["wall_clocks"] = {}
        for wc_label, wc_iso in [
            ("2027_midseason", "2027-07-15T14:00:00-04:00"),
            ("2027_nye", "2027-12-31T23:55:00-05:00"),
            ("2028_new_year", "2028-01-01T00:05:00-05:00"),
            ("2028_midseason", "2028-07-15T14:00:00-04:00"),
        ]:
            wc = _run_pytest(
                f"wall_{wc_label}",
                REPO / "src",
                WALL_CLOCK_TESTS,
                scratch / f"wc_{wc_label}.xml",
                logs_dir / f"wall_{wc_label}.log",
                wall_iso=wc_iso,
                timeout=120,
            )
            report["wall_clocks"][wc_label] = {
                "passed": wc["passed_count"],
                "skipped": wc["skipped_count"],
                "failed": wc["failed_count"],
                "failures": wc["failed"],
            }

        # 4. Scratch-copy mutation sweep
        report["mutants"] = {}
        for m_name, rel_file, target, replacement in MUTANTS:
            m_src = scratch / f"mut_{m_name}" / "src"
            shutil.copytree(REPO / "src", m_src)
            fpath = m_src / rel_file
            orig = fpath.read_text()
            cnt = orig.count(target)
            assert cnt == 1, f"Mutant {m_name}: target count={cnt} in {rel_file}"
            fpath.write_text(orig.replace(target, replacement, 1))

            m_new = _run_pytest(
                f"mut_{m_name}_new_tests",
                m_src,
                NEW_TESTS,
                scratch / f"mut_{m_name}_new.xml",
                logs_dir / f"mut_{m_name}_new.log",
                timeout=120,
            )
            entry = {
                "new_tests_status": "KILLED" if m_new["failed_count"] > 0 else "SURVIVED",
                "new_tests_failed_count": m_new["failed_count"],
                "new_tests_killing_samples": [x["node"] for x in m_new["failed"][:6]],
            }
            if m_new["failed_count"] == 0:
                m_all = _run_pytest(
                    f"mut_{m_name}_streak_suite",
                    m_src,
                    STREAK_RECONCILE_LOCK_TESTS,
                    scratch / f"mut_{m_name}_streak.xml",
                    logs_dir / f"mut_{m_name}_streak.log",
                    timeout=120,
                )
                entry["streak_suite_status"] = "KILLED" if m_all["failed_count"] > 0 else "SURVIVED"
                entry["streak_suite_failed_count"] = m_all["failed_count"]
                entry["streak_suite_killing_samples"] = [x["node"] for x in m_all["failed"][:6]]
            report["mutants"][m_name] = entry

    (HERE / "item6_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("=== FOCUSED ONE-PROCESS ===")
    print(json.dumps(report["focused_one_process"], indent=2))
    print("=== RED ON c70effe9 ===")
    print(json.dumps({k: v for k, v in report["red_on_c70effe9"].items() if k != "all_failed"}, indent=2))
    print("=== RED ON c2039541 ===")
    print(json.dumps({k: v for k, v in report["red_on_c2039541"].items() if k != "all_failed"}, indent=2))
    print("=== WALL CLOCKS ===")
    print(json.dumps(report["wall_clocks"], indent=2))
    print("=== MUTANTS ===")
    print(json.dumps(report["mutants"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
