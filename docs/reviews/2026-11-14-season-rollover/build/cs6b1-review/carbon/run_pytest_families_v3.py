"""Self-contained sequential pytest runner v3 for CS6b-1 review Item 7."""
from __future__ import annotations

import collections
import datetime as dt
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE_SRC = HERE / "base_tree" / "src"
BRANCH_SRC = REPO / "src"
AUTHOR_TEST = "tests/test_morning_state_cs6b1.py"
ADOPT_TEST = str(HERE / "adopt" / "test_morning_state_cs6b1_adopt.py")
REAL_DT = dt.datetime
REAL_DATE = dt.date
ET = ZoneInfo("America/New_York")


def install_network_guard() -> None:
    def audit(event: str, args: tuple) -> None:
        if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            host = args[0] if args else None
            if host not in {"127.0.0.1", "localhost", "::1", None}:
                raise AssertionError(f"forbidden DNS lookup: {event} {args!r}")
        elif event in {"socket.connect", "socket.connect_ex"}:
            addr = args[1] if len(args) > 1 else None
            host = addr[0] if isinstance(addr, tuple) and addr else None
            if host not in {"127.0.0.1", "::1"}:
                raise AssertionError(f"forbidden outbound connect: {event} {args!r}")

    sys.addaudithook(audit)


def child_pytest(payload_json: str) -> None:
    cfg = json.loads(payload_json)
    src = Path(cfg["src"])
    sys.path.insert(0, str(src))
    if cfg.get("guard", True):
        install_network_guard()

    hygiene = None
    if cfg.get("check_hygiene"):
        sys.path.insert(0, str(REPO))
        mod_path = cfg["selections"][0]
        if mod_path.endswith("test_morning_state_cs6b1.py"):
            importlib.import_module("tests.test_morning_state_cs6b1")
        else:
            spec = importlib.util.spec_from_file_location("adopt_mod", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        hygiene = {
            "predict_imported_at_module_level": "bts.model.predict" in sys.modules,
            "lightgbm_imported_at_module_level": "lightgbm" in sys.modules,
        }

    import pandas  # noqa: F401
    import pytest

    if cfg.get("clock_iso"):
        fixed = REAL_DT.fromisoformat(cfg["clock_iso"])

        class WallClock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

            @classmethod
            def today(cls):
                return cls.now()

            @classmethod
            def utcnow(cls):
                return fixed.astimezone(dt.timezone.utc).replace(tzinfo=None)

        class WallDate(REAL_DATE):
            @classmethod
            def today(cls):
                return fixed.astimezone(ET).date()

        dt.datetime = WallClock
        dt.date = WallDate

    import bts.scheduler as sched_mod
    assert Path(sched_mod.__file__).resolve().is_relative_to(src.resolve()), sched_mod.__file__

    snap_before = {
        "env": dict(os.environ),
        "datetime": dt.datetime,
        "date": dt.date,
        "time_sleep": time.sleep,
        "urlopen": urllib.request.urlopen,
        "root_handlers": len(logging.getLogger().handlers),
    }

    records: dict[str, dict] = {}

    class Recorder:
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.failed and report.when != "call"):
                records[report.nodeid] = {
                    "outcome": report.outcome,
                    "when": report.when,
                    "failure": str(report.longrepr)[:600] if report.failed else None,
                }

    basetemp = HERE / "pytest_tmp" / cfg["label"]
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    if basetemp.exists():
        shutil.rmtree(basetemp)
    rc = pytest.main(
        ["-p", "no:cacheprovider", "-q", "--tb=short", "--basetemp", str(basetemp), *cfg["selections"]],
        plugins=[Recorder()],
    )

    snap_after = {
        "env": dict(os.environ),
        "datetime": dt.datetime,
        "date": dt.date,
        "time_sleep": time.sleep,
        "urlopen": urllib.request.urlopen,
        "root_handlers": len(logging.getLogger().handlers),
    }
    env_diff = {
        k: (snap_before["env"].get(k), snap_after["env"].get(k))
        for k in set(snap_before["env"]) | set(snap_after["env"])
        if k != "PYTEST_CURRENT_TEST" and snap_before["env"].get(k) != snap_after["env"].get(k)
    }
    state_leak = {
        "env_diff": env_diff,
        "datetime_changed": snap_before["datetime"] is not snap_after["datetime"],
        "date_changed": snap_before["date"] is not snap_after["date"],
        "sleep_changed": snap_before["time_sleep"] is not snap_after["time_sleep"],
        "urlopen_changed": snap_before["urlopen"] is not snap_after["urlopen"],
        "root_handlers_delta": snap_after["root_handlers"] - snap_before["root_handlers"],
    }

    counts = dict(collections.Counter(r["outcome"] for r in records.values()))
    out = {
        "label": cfg["label"],
        "rc": int(rc),
        "counts": counts,
        "hygiene": hygiene,
        "state_leak": state_leak,
        "passed_ids": sorted(k for k, v in records.items() if v["outcome"] == "passed"),
        "failed_ids": sorted(k for k, v in records.items() if v["outcome"] == "failed"),
        "failures": {k: v["failure"] for k, v in records.items() if v["outcome"] == "failed"},
    }
    Path(cfg["out_file"]).write_text(json.dumps(out, indent=2))


def run_case(label: str, src: Path, selections: list[str], *, clock_iso: str | None = None,
             check_hygiene: bool = False, guard: bool = True) -> dict:
    out_dir = HERE / "pytest_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{label}.json"
    payload = json.dumps({
        "label": label,
        "src": str(src),
        "selections": selections,
        "clock_iso": clock_iso,
        "check_hygiene": check_hygiene,
        "guard": guard,
        "out_file": str(out_file),
    })
    subprocess.run(
        [sys.executable, __file__, "child", payload],
        cwd=REPO,
        check=True,
        env=os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    res = json.loads(out_file.read_text())
    print(f"[{label}] rc={res['rc']} counts={res['counts']}", flush=True)
    return res


def make_mutant(root: Path, name: str, old: str, new: str, *, apply_b1_fix: bool = False) -> Path:
    target = root / name / "src"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(BRANCH_SRC, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sched = target / "bts" / "scheduler.py"
    text = sched.read_text(encoding="utf-8")
    if apply_b1_fix:
        text = text.replace(
            "from bts.contest_state import ContestStateError, automatic_reading_record, load_decision_streak_state",
            "from bts.contest_state import ContestStateError, _UNREADABLE, automatic_reading_record, load_decision_streak_state",
            1,
        ).replace(
            "    except (OSError, ValueError):",
            "    except _UNREADABLE:",
            1,
        ).replace(
            "    except (OSError, ValueError, TypeError, KeyError, AttributeError):",
            "    except (*_UNREADABLE, TypeError, KeyError, AttributeError):",
            1,
        )
    assert text.count(old) == 1, f"mutant {name}: expected 1 match, got {text.count(old)}"
    sched.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child_pytest(sys.argv[2])
        return

    changed_tests = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", "1bf518a2", "HEAD", "--", "tests/"],
        text=True,
    ).splitlines()
    assert changed_tests == ["tests/test_morning_state_cs6b1.py"], changed_tests

    base_res = run_case("01_base_author", BASE_SRC, [AUTHOR_TEST])
    assert base_res["counts"] == {"failed": 41, "passed": 20}, base_res["counts"]

    branch_res = run_case("02_branch_author", BRANCH_SRC, [AUTHOR_TEST], check_hygiene=True)
    assert branch_res["rc"] == 0 and branch_res["counts"] == {"passed": 61}
    assert branch_res["hygiene"] == {
        "predict_imported_at_module_level": False,
        "lightgbm_imported_at_module_level": False,
    }
    assert branch_res["state_leak"] == {
        "env_diff": {},
        "datetime_changed": False,
        "date_changed": False,
        "sleep_changed": False,
        "urlopen_changed": False,
        "root_handlers_delta": 0,
    }, branch_res["state_leak"]

    adopt_unpatched = run_case("03_branch_adopt_unpatched", BRANCH_SRC, [ADOPT_TEST], check_hygiene=True)
    assert adopt_unpatched["hygiene"] == {
        "predict_imported_at_module_level": False,
        "lightgbm_imported_at_module_level": False,
    }
    assert adopt_unpatched["state_leak"] == {
        "env_diff": {},
        "datetime_changed": False,
        "date_changed": False,
        "sleep_changed": False,
        "urlopen_changed": False,
        "root_handlers_delta": 0,
    }, adopt_unpatched["state_leak"]
    assert len(adopt_unpatched["failed_ids"]) == 2 and all(
        "test_b1_deeply_nested_unreadable_decision_or_pick_counts_as_absent" in fid
        for fid in adopt_unpatched["failed_ids"]
    ), adopt_unpatched["failed_ids"]

    mut_root = HERE / "scratch_mutants"
    if mut_root.exists():
        shutil.rmtree(mut_root)
    patched_src = make_mutant(
        mut_root, "00_patched_b1",
        "MORNING_CHECK_MIN_LEAD_MIN = 10",
        "MORNING_CHECK_MIN_LEAD_MIN = 10",
        apply_b1_fix=True,
    )
    adopt_patched = run_case("04_patched_author_plus_adopt", patched_src, [AUTHOR_TEST, ADOPT_TEST])
    assert adopt_patched["rc"] == 0 and adopt_patched["failed_ids"] == [], adopt_patched

    mutant_specs = [
        ("M01_remove_call",
         "                _morning_state_check(config, picks_dir, date, now, first_check_et)",
         "                pass"),
        ("M02a_drop_delivery_gate",
         '        if (delivery_mode in {"dm", "public"} and _contest_state_required(config)',
         "        if (True and _contest_state_required(config)"),
        ("M02b_drop_required_gate",
         '        if (delivery_mode in {"dm", "public"} and _contest_state_required(config)',
         '        if (delivery_mode in {"dm", "public"} and True'),
        ("M02c_drop_prior_lock_gate",
         "                and (previous_state is None or not previous_state.pick_locked)):",
         "                and True):"),
        ("M03a_lead_10_to_9",
         "MORNING_CHECK_MIN_LEAD_MIN = 10",
         "MORNING_CHECK_MIN_LEAD_MIN = 9"),
        ("M03b_lt_to_le",
         "            elif first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):",
         "            elif first_check_et - now <= timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):"),
        ("M04_drop_same_date",
         "    if date != now.date().isoformat():\n        return",
         "    if False:\n        return"),
        ("M05a_drop_scoreable_return",
         "        if is_scoreable_commit(date, picks_dir, None):\n            return",
         "        if False:\n            return"),
        ("M05b_drop_delivered_return",
         "    if daily is not None and (pick_was_delivered(daily) or daily.delivery_attempted):",
         "    if daily is not None and (False or daily.delivery_attempted):"),
        ("M05c_drop_attempted_return",
         "    if daily is not None and (pick_was_delivered(daily) or daily.delivery_attempted):",
         "    if daily is not None and (pick_was_delivered(daily) or False):"),
        ("M06_refusal_in_dm",
         '                f"a lineup check remains. First planned lineup check at {first_check_et:%H:%M} ET."',
         '                f"a lineup check remains. First planned lineup check at {first_check_et:%H:%M} ET. {error}"'),
        ("M07_shared_status_row2",
         '                status_path=health_dir / "contest_morning_check_dm_delivery_status.json",',
         '                status_path=health_dir / "health_dm_delivery_status.json",'),
        ("M08_wrong_key_row3",
         '            warn_attention=[Alert("WARN", "contest_state", message, incident_key="contest_state:stale_reading")],',
         '            warn_attention=[Alert("WARN", "contest_state", message, incident_key="contest_state:morning_stale")],'),
        ("M09a_drop_now_et_date_row2",
         '                status_path=health_dir / "contest_morning_check_dm_delivery_status.json",\n                now_et_date=now.date(),',
         '                status_path=health_dir / "contest_morning_check_dm_delivery_status.json",'),
        ("M09b_drop_now_et_date_row3",
         '            status_path=health_dir / "contest_stale_reading_dm_delivery_status.json",\n            now_et_date=now.date(),',
         '            status_path=health_dir / "contest_stale_reading_dm_delivery_status.json",'),
        ("M10a_narrow_base_exc_report",
         'def _report_morning_check_failure(error: BaseException) -> None:\n    try:\n        print(f"  Morning state check failed: {ascii(str(error))[:200]}", file=sys.stderr)\n    except BaseException:',
         'def _report_morning_check_failure(error: BaseException) -> None:\n    try:\n        print(f"  Morning state check failed: {ascii(str(error))[:200]}", file=sys.stderr)\n    except Exception:'),
        ("M10b_narrow_base_exc_print",
         'def _print_morning_check(message: str) -> None:\n    try:\n        print(message, file=sys.stderr)\n    except BaseException as error:',
         'def _print_morning_check(message: str) -> None:\n    try:\n        print(message, file=sys.stderr)\n    except Exception as error:'),
        ("M10c_narrow_base_exc_runday",
         "            else:\n                _morning_state_check(config, picks_dir, date, now, first_check_et)\n    except BaseException as error:\n        _report_morning_check_failure(error)",
         "            else:\n                _morning_state_check(config, picks_dir, date, now, first_check_et)\n    except Exception as error:\n        _report_morning_check_failure(error)"),
        ("M11_helper_writes_scheduler_state",
         '    health_dir = picks_dir.parent / "health_state"',
         '    save_state(load_state(date, picks_dir) or SchedulerState(date, now.isoformat(), [], [], [], False, None, None, None), picks_dir)\n    health_dir = picks_dir.parent / "health_state"'),
        ("M12a_read_twice_before_try",
         "    try:\n        state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)",
         "    try:\n        load_decision_streak_state(picks_dir, require_contest_state=True, now=now)\n    except ContestStateError:\n        pass\n    try:\n        state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)"),
        ("M12b_read_twice_after_try",
         "    record = automatic_reading_record(state, now)",
         "    state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)\n    record = automatic_reading_record(state, now)"),
    ]

    mutant_summary = {}
    for mname, old, new in mutant_specs:
        msrc = make_mutant(mut_root, mname, old, new)
        mres = run_case(f"mut_{mname}", msrc, [AUTHOR_TEST])
        killed_by_author = mres["rc"] != 0
        killed_by_adopt = None
        if not killed_by_author:
            msrc_b1 = make_mutant(mut_root, f"{mname}_b1", old, new, apply_b1_fix=True)
            ares = run_case(f"mut_adopt_{mname}", msrc_b1, [ADOPT_TEST])
            killed_by_adopt = ares["rc"] != 0
            assert killed_by_adopt, f"adopt failed to kill {mname}!"
        if mname == "M10c_narrow_base_exc_runday":
            assert sum(mres["counts"].values()) == 61, mres["counts"]
        mutant_summary[mname] = {
            "killed_by_author": killed_by_author,
            "author_failed_count": len(mres["failed_ids"]),
            "author_total_ran": sum(mres["counts"].values()),
            "killed_by_adopt": killed_by_adopt,
        }

    clock_summary = {}
    for clabel, ciso in (
        ("clock_2027_mid", "2027-07-15T12:00:00-04:00"),
        ("clock_2027_nye", "2027-12-31T23:59:59-05:00"),
        ("clock_2028_leap", "2028-02-29T12:00:00-05:00"),
        ("clock_2028_mid", "2028-07-15T12:00:00-04:00"),
    ):
        cres_author = run_case(f"{clabel}_author", BRANCH_SRC, [AUTHOR_TEST], clock_iso=ciso)
        assert cres_author["rc"] == 0 and cres_author["counts"] == {"passed": 61}
        cres_both = run_case(f"{clabel}_patched_both", patched_src, [AUTHOR_TEST, ADOPT_TEST], clock_iso=ciso)
        assert cres_both["rc"] == 0 and cres_both["failed_ids"] == []
        clock_summary[clabel] = {
            "author_branch": cres_author["counts"],
            "patched_author_plus_adopt": cres_both["counts"],
        }

    # Per brief.md line 5 ("do not run tests/test_web_saver.py ... under the guard"), disable DNS/connect guard here
    combined_files = [
        AUTHOR_TEST,
        "tests/test_lightgbm_optional.py",
        "tests/test_web_saver.py",
        "tests/test_saved_pick_paths_cs5a.py",
        "tests/test_stale_delivery_cs4c.py",
        "tests/test_stale_delivery_cs4c_carbon.py",
    ]
    comb_res = run_case("05_combined_one_process", BRANCH_SRC, combined_files, guard=False)
    assert comb_res["failed_ids"] in (
        [],
        ["tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring"],
    ), comb_res["failed_ids"]

    summary = {
        "changed_test_files": changed_tests,
        "base_author": {
            "counts": base_res["counts"],
            "passed_ids": base_res["passed_ids"],
        },
        "branch_author": {
            "counts": branch_res["counts"],
            "hygiene": branch_res["hygiene"],
            "state_leak": branch_res["state_leak"],
        },
        "adopt_unpatched": {
            "counts": adopt_unpatched["counts"],
            "failed_ids": adopt_unpatched["failed_ids"],
            "hygiene": adopt_unpatched["hygiene"],
            "state_leak": adopt_unpatched["state_leak"],
        },
        "adopt_patched_with_author": adopt_patched["counts"],
        "mutants": mutant_summary,
        "clocks": clock_summary,
        "combined_one_process": {
            "counts": comb_res["counts"],
            "failed_ids": comb_res["failed_ids"],
            "failures": comb_res["failures"],
        },
    }
    (HERE / "pytest_families_summary.json").write_text(json.dumps(summary, indent=2))
    print("[ALL PYTEST FAMILIES COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
