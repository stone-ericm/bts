"""Repeat applicable round-one mutations and probe amendment 4 on independent source copies."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from run_selection import FOCUSED, HERE, ROOT, W15


sys.path.insert(0, str(HERE.parent))
import probe_mutations as old


PRODUCER, READER, HEALTH = old.PRODUCER, old.READER, old.HEALTH
change, replacement, span = old.change, old.replacement, old.span
PRIOR = dict(old.MUTATIONS)
R1 = "tests/test_wr_w15b_r1.py"
HEALTH_TESTS = "tests/test_wr_w15_health.py"
W15B = "tests/test_wr_w15b.py"


def escaped(value):
    return f'(str({value}).replace(chr(92), chr(92) * 2).replace(chr(96), chr(92) + "x60"))'


def restore_evidence_location(text):
    return span(text, '        evidence = (f"That run recorded an evidence copy named',
                "\n\n    rescanned =",
                '        copy = Path(row["cache_path"]).parent / "rejected" / row["preserved"]\n'
                '        evidence = f"That run recorded an evidence copy at {copy}."')


OBSOLETE = {
    "drop-backticks-in-display",
    "skip-display-status-log", "skip-display-evidence-log",
    "skip-display-stderr-evidence-failure", "skip-display-stderr-status-failure", "skip-display-stderr-drop",
}
REPLACED = {
    "omit-date-in-writer-and-reader": ("omit-date-in-writer-and-reader", [
        *PRIOR["omit-date-in-writer-and-reader"][:2],
        (HEALTH, lambda text: change(text,
                 'day = next((day for day in days if fnmatchcase(path.name, f"{day}.*.json")), None)',
                 'day = str(today) if any(fnmatchcase(path.name, pattern) for pattern in '
                 '("invalid_entries.*.json", "unreadable_cache.*.json")) else None')),
    ]),
    "drop-yesterday": replacement("drop-yesterday", HEALTH, "(str(today), str(today - timedelta(days=1)))", "(str(today),)"),
    "restore-evidence-backticks": replacement(
        "restore-evidence-backticks", HEALTH, "{row['preserved']} ", "{chr(96)}{row['preserved']}{chr(96)} "),
    "escape-record-cache-path": replacement(
        "escape-record-cache-path", PRODUCER, "cache_path=str(cache)", f"cache_path={escaped('cache')}"),
    "escape-path-in-producer-identity-only": replacement(
        "escape-path-in-producer-identity-only", PRODUCER, "dict(observation, outcomes=outcomes)",
        f"dict(observation, cache_path={escaped('cache')}, outcomes=outcomes)"),
    "escape-bad-name-in-incident-key": replacement(
        "escape-bad-name-in-incident-key", HEALTH, "identities.append(path.name)", f"identities.append({escaped('path.name')})"),
    "restore-stderr-replace-errors": replacement(
        "restore-stderr-replace-errors", PRODUCER, 'getattr(stream, "errors", None) or "backslashreplace"', 'errors="replace"'),
}
NEW = [
    replacement("raw-status-directory-log", HEALTH, "log_unreadable(FOLDER)", "log_unreadable(status_dir)"),
    replacement("raw-status-entry-log", HEALTH, "log_unreadable(entry_label(path.name, day))", "log_unreadable(path)"),
    ("raw-evidence-path-log", [(HEALTH, restore_evidence_location)]),
    replacement("raw-preservation-failure-path", PRODUCER, "cache evidence ({type(error).__name__})",
                "cache evidence at {target} ({type(error).__name__})"),
    replacement("raw-publication-failure-path", PRODUCER, "cache status record {target.name}",
                "cache status record {target}"),
    replacement("raw-drop-cache-path", PRODUCER, "cache: dropped", "cache {cache}: dropped"),
    replacement("any-name-as-log-record", HEALTH, "if RECORD_NAME.fullmatch(name):", "if True:"),
    replacement("any-status-record-basename", READER, 'or path.name != f"{day}.{reason}.{identity[:12]}.json"):',
                "or False):"),
    replacement("any-evidence-basename", READER, "not isinstance(copy, str) or not EVIDENCE.fullmatch(copy)",
                "not isinstance(copy, str)"),
    replacement("raw-recorded-at", HEALTH, 'datetime.fromisoformat(row["recorded_at"]).isoformat()', 'row["recorded_at"]'),
    replacement("no-errors-fallback", PRODUCER, 'getattr(stream, "errors", None)', "stream.errors"),
    replacement("escape-directory-name-in-incident-key", HEALTH, "[status_dir.name]", f"[{escaped('status_dir.name')}]"),
]
MUTATIONS = [REPLACED.get(name, (name, changes)) for name, changes in old.MUTATIONS if name not in OBSOLETE] + NEW
TARGETS = {
    "restore-evidence-backticks": f"{HEALTH_TESTS}::test_operator_text_describes_the_observation_and_recovery_route",
    "escape-record-cache-path": f"{R1}::test_producer_dropped_line_names_no_path",
    "escape-bad-name-in-incident-key": f"{R1}::test_badly_named_status_entry_supplies_no_journal_evidence",
    "restore-stderr-replace-errors": f"{W15B}::test_ascii_stderr_keeps_backslash_spelling_of_a_non_ascii_path",
    "raw-status-directory-log": f"{R1}::test_unreadable_status_directory_names_no_path",
    "raw-status-entry-log": f"{R1}::test_badly_named_status_entry_supplies_no_journal_evidence",
    "raw-evidence-path-log": f"{R1}::test_record_cache_path_is_not_displayed",
    "raw-preservation-failure-path": f"{R1}::test_producer_failure_lines_name_no_path",
    "raw-publication-failure-path": f"{R1}::test_producer_failure_lines_name_no_path",
    "raw-drop-cache-path": f"{R1}::test_producer_dropped_line_names_no_path",
    "any-name-as-log-record": f"{R1}::test_badly_named_status_entry_supplies_no_journal_evidence",
    "any-status-record-basename": f"{HEALTH_TESTS}::test_malformed_fields_warn",
    "any-evidence-basename": f"{HEALTH_TESTS}::test_malformed_fields_warn",
    "raw-recorded-at": f"{R1}::test_timestamp_is_rendered_from_its_parsed_value",
    "no-errors-fallback": f"{R1}::test_stream_without_errors_attribute_still_gets_the_line",
}


def historical_targets():
    log = (HERE.parent / "mutations.log").read_text()
    return dict(re.findall(r"^KILLED (\S+) FAILED (tests/\S+)", log, re.MULTILINE))


def run(name, stage, source, selection, label):
    log = HERE / "mutation-results" / f"{name}.{label}.log"
    command = [sys.executable, str(HERE / "run_selection.py"), str(source),
               "-q", "-x", "-o", "addopts=", "--tb=short", "-p", "no:cacheprovider",
               "--basetemp", str(stage / label), *selection]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(HERE.parent / "offline"), str(source))),
               BTS_W15_REAL_ARTIFACTS=("1" if label == "combined" else "0"))
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, timeout=90,
                                   stdout=stream, stderr=subprocess.STDOUT)
    message = log.read_text()
    if completed.returncode == 0:
        return None
    if completed.returncode == 1:
        first = next((line for line in message.splitlines() if line.startswith("FAILED tests/")), None)
        if first:
            return first
    # Old node IDs occasionally disappear between rounds. Retry against the current suite.
    if label == "targeted" and completed.returncode in (4, 5):
        print("Updated test selection for", name, flush=True)
        return None
    raise RuntimeError(f"{name} {label}: pytest exit {completed.returncode}\n{message[-5000:]}")


def main():
    (HERE / "mutation-results").mkdir(exist_ok=True)
    targets = historical_targets() | TARGETS
    survivors = []
    for name, edits in MUTATIONS:
        with TemporaryDirectory(dir=HERE, prefix="mutant-") as work:
            stage = Path(work)
            source = stage / "src"
            shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
            for filename, mutate in edits:
                path = source / filename
                updated = mutate(path.read_text())
                compile(updated, str(path), "exec")
                path.write_text(updated)
            failure = None
            if name in targets:
                failure = run(name, stage, source, [targets[name]], "targeted")
            if failure is None:
                failure = run(name, stage, source, W15, "w15")
            if failure is None:
                failure = run(name, stage, source, FOCUSED, "combined")
            if failure:
                print("KILLED", name, failure, flush=True)
            else:
                survivors.append(name)
                print("SURVIVED", name, "passed full selection", flush=True)
    print("TOTAL", len(MUTATIONS), "KILLED", len(MUTATIONS) - len(survivors),
          "SURVIVED", len(survivors), repr(survivors), flush=True)
    print("Removed display helper mutants replaced by raw-path mutations:", sorted(OBSOLETE), flush=True)


if __name__ == "__main__":
    main()
