"""Run the changed assertions against historical and scratch scheduler code."""

import local_guard

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from schedule_round2 import HERE, ROOT, archive


TEST = "tests/test_live_gate_cs6a.py"
RESOURCES = "test_schedule_refuses_before_resource_access"
ORDER = "test_default_date_is_resolved_before_the_configuration_is_loaded"
WARNING = "test_dry_run_warning_case_also_keeps_the_start_date"
REFUSAL = "test_real_run_refusal_resolves_the_date_first_and_does_nothing_else"
EXPLICIT = "test_explicit_date_never_asks_for_today"
OLD_ORDER = "test_dry_run_default_date_is_resolved_before_the_configuration_is_loaded"
OLD_REFUSAL = "test_real_run_refusal_still_computes_no_default_date"
NODES = (RESOURCES, ORDER, WARNING, REFUSAL, EXPLICIT)
HISTORICAL = (RESOURCES, OLD_ORDER, OLD_REFUSAL)
DAY = (
    "    # Resolve the default date FIRST, before the configuration is loaded, as this\n"
    "    # command always did: a launch whose load straddles midnight ET must schedule\n"
    "    # the day it started on (the old day's late work), not the next one.\n"
    "    if date is None:\n"
    "        date = _today_et()\n\n"
)
DAY_ONLY = "    if date is None:\n        date = _today_et()\n\n"
LOAD = "    config = load_config(Path(config_path))\n"
WORKER = "    run_day(date=date, config=config, dry_run=dry_run)\n"
GATE = '            click.echo(f"bts schedule: refusing to start: {problem}", err=True)\n'
REAL_ORDER = {ORDER + "/required/real", ORDER + "/private/real"}
DRY_ORDER = {ORDER + "/required/dry", ORDER + "/private/dry"}
ORDER_FAILURES = REAL_ORDER | DRY_ORDER | {WARNING, REFUSAL}


def select_tests(item):
    name = item.originalname
    if name in {ORDER, OLD_ORDER}:
        params = item.callspec.params
        body = "required" if "contest_state_required" in params["body"] else (
            "warned" if 'pick_delivery = "dm"' in params["body"] else "private"
        )
        mode = "dry" if params.get("dry", True) else "real"
        return f"{name}/{body}/{mode}"
    return name


def child(src, temp, expected_failures, expected_count, *nodes):
    sys.path[:0] = [src, str(ROOT)]
    from bts import cli
    import pytest

    assert Path(cli.__file__).is_relative_to(Path(src)), cli.__file__

    class Checks:
        def __init__(self):
            self.labels = {}
            self.passed = set()
            self.failed = set()
            self.other = []

        def pytest_collection_modifyitems(self, items):
            self.labels = {item.nodeid: select_tests(item) for item in items}

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed.add(self.labels[report.nodeid])
            elif report.when == "call" and report.failed:
                self.failed.add(self.labels[report.nodeid])
            elif not report.passed:
                self.other.append((report.nodeid, report.when, report.outcome))

    checks = Checks()
    result = pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider", "--tb=short",
                          "--basetemp", temp, *nodes], plugins=[checks])
    assert not local_guard.network, local_guard.network
    assert not checks.other, checks.other
    expected_status = pytest.ExitCode.TESTS_FAILED if expected_failures else pytest.ExitCode.OK
    assert result == expected_status, result
    assert checks.failed == expected_failures, (checks.failed, expected_failures)
    assert len(checks.passed | checks.failed) == expected_count, (checks.passed, checks.failed)
    for label in sorted(checks.passed):
        print("PASS", label)
    for label in sorted(checks.failed):
        print("FAIL", label)


def run(label, src, scratch, expected, count=8, nodes=NODES, tests=TEST):
    print(f"=== {label} ===", flush=True)
    args = [sys.executable, __file__, "child", str(src), str(scratch / (label + "-tmp")),
            ",".join(sorted(expected)), str(count), *(f"{tests}::{node}" for node in nodes)]
    subprocess.run(args, cwd=ROOT, check=True, timeout=100)


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def mutate_source(source, *edits):
    start = source.index("def schedule(")
    end = source.index("\n@cli.command", start)
    change = source[start:end]
    for old, new in edits:
        change = replace_once(change, old, new)
    return source[:start] + change + source[end:]


def main():
    with TemporaryDirectory(prefix="mutation-runs-", dir=HERE) as folder:
        scratch = Path(folder)
        head = ROOT / "src"
        prior = archive("87c118f6", scratch / "prior")
        run("head", head, scratch, set())
        run("prior-with-current-tests", prior, scratch, REAL_ORDER | {REFUSAL})

        historical = scratch / "test_historical_live_gate_cs6a.py"
        historical.write_bytes(subprocess.check_output(["git", "show", f"87c118f6:{TEST}"], cwd=ROOT))
        run("prior-with-prior-tests", prior, scratch, set(), 4, HISTORICAL, historical)
        run("head-with-prior-tests", head, scratch, {RESOURCES, OLD_REFUSAL}, 4, HISTORICAL, historical)

        original = (head / "bts" / "cli.py").read_text()
        mutants = (
            ("date-after-load", ((DAY, ""), (LOAD, LOAD + DAY_ONLY)), ORDER_FAILURES, NODES, 8),
            ("date-after-gate", ((DAY, ""), (WORKER, DAY_ONLY + WORKER)), ORDER_FAILURES, NODES, 8),
            ("date-only-for-dry-run", ((DAY, DAY_ONLY.replace("if date", "if dry_run and date")),),
             REAL_ORDER | {REFUSAL}, NODES, 8),
            ("overwrite-explicit-date", ((DAY, "    date = _today_et()\n\n"),), {EXPLICIT}, (EXPLICIT,), 1),
            ("load-state-on-refusal", ((GATE, "            from bts.scheduler import load_state\n"
                                             "            load_state(date, Path(config_path).parent)\n" + GATE),),
             {RESOURCES}, (RESOURCES,), 1),
        )
        for label, edits, failures, nodes, count in mutants:
            src = archive("HEAD", scratch / label)
            (src / "bts" / "cli.py").write_text(mutate_source(original, *edits))
            run(label, src, scratch, failures, count, nodes)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        _, _, src, temp, expected, count, *nodes = sys.argv
        child(src, temp, set(expected.split(",")) if expected else set(), int(count), *nodes)
    else:
        main()
