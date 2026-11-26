"""Isolate each source mutation, then run W15 and confirm survivors against the full selection."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from run_focused import FOCUSED, HERE, ROOT, W15


PRODUCER = "bts/features/probable_pitcher_cache.py"
READER = "bts/probable_pitcher_status.py"
HEALTH = "bts/health/probable_pitcher_cache.py"

SAVE = """    if scanned and can_rewrite and not special:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))
            os.replace(temp, cache)
        finally:
            temp.unlink(missing_ok=True)
"""


def change(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def span(text, start, end, replacement):
    beginning = text.index(start)
    ending = text.index(end, beginning)
    return text[:beginning] + replacement + text[ending:]


def move_save(text, before_preserve):
    _, _, body = SAVE.partition("\n")
    helper = "def save(cache, lookup):\n" + "\n".join(line[4:] for line in body.splitlines()) + "\n\n\n"
    text = change(text, SAVE, "")
    text = change(text, "def build(cache, raw):", helper + "def build(cache, raw):")
    if before_preserve:
        return change(text, "    can_rewrite = True\n", "    if scanned and not special:\n        save(cache, lookup | scanned)\n    can_rewrite = True\n")
    text = change(text, "def report(cache, original, special, reason, scanned, outcomes, counts):",
                  "def report(cache, original, special, reason, scanned, outcomes, counts, final_lookup):")
    text = text.replace("scanned, outcomes, counts)", "scanned, outcomes, counts, lookup | scanned)")
    text = change(text, "    try:\n        publish(target, row)",
                  "    if scanned and not special and preserved_ok:\n        save(cache, final_lookup)\n    try:\n        publish(target, row)")
    return change(text, "    lookup.update(scanned)\n",
                  "    if scanned and not special and not unreadable and not invalid:\n        save(cache, lookup | scanned)\n    lookup.update(scanned)\n")


def reinsert_summary_path(text):
    text = change(text, "def summarize(today, identities, kinds):", "def summarize(today, identities, kinds, status_dir=None):")
    text = change(text, "    payload = json.dumps(sorted(identities)",
                  "    if status_dir is not None:\n"
                  '        text = text.replace("in this health run\'s log and in health_state/probable_pitcher_cache beside the picks directory", f"in {status_dir}")\n'
                  "    payload = json.dumps(sorted(identities)")
    text = change(text, "summarize(today, [status_dir.name], {STATUS_UNREADABLE})",
                  "summarize(today, [status_dir.name], {STATUS_UNREADABLE}, status_dir)")
    return change(text, "summarize(today, identities, kinds)] if identities",
                  "summarize(today, identities, kinds, status_dir)] if identities")


def revert_fd(text):
    return span(text, "    fd = os.open(path, flags)", "\n\n\ndef directory_exists",
                '    with os.fdopen(os.open(path, flags), "rb") as stream:\n'
                "        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):\n"
                "            raise NonRegularFile(path)\n"
                "        yield stream")


def revert_buffered_stderr(text):
    return span(text, "def warn(message):", "\n\n\ndef decode",
                'def warn(message):\n    try:\n        print(f"WARNING: {message}", file=sys.stderr)\n'
                "    except Exception:\n        pass")


def drop_backticks(text):
    return span(text, "    return str(value).replace(", "\n\n\nclass NonRegularFile",
                "    return str(value).replace(chr(92), chr(92) * 2).replace(chr(96), '')")


def replacement(name, module, old, new):
    return name, [(module, lambda text: change(text, old, new))]


def edits(*changes):
    def transform(text):
        for old, new in changes:
            text = change(text, old, new)
        return text
    return transform


MUTATIONS = [
    replacement("producer-accept-bool", PRODUCER, "type(value) is int", "isinstance(value, int)"),
    replacement("reader-accept-bool", READER, "elif (any(type(value) is not int", "elif (any(not isinstance(value, int)"),
    replacement("skip-raw-validation", PRODUCER, "if valid(entry):\n                scanned[game]", "if True:\n                scanned[game]"),
    ("replace-before-preserve", [(PRODUCER, lambda text: move_save(text, True))]),
    ("publish-after-replace", [(PRODUCER, lambda text: move_save(text, False))]),
    replacement("reuse-evidence-by-name", PRODUCER, "        try:\n            with open_regular(candidate)",
                '        if candidate.name.endswith(f".{original_sha[:12]}.json"):\n            return candidate.name\n'
                "        try:\n            with open_regular(candidate)"),
    replacement("omit-date", PRODUCER, 'f"{day}.{reason}.{identity[:12]}.json"', 'f"{reason}.{identity[:12]}.json"'),
    ("omit-date-in-writer-and-reader", [
        (PRODUCER, edits(('f"{day}.{reason}.{identity[:12]}.json"', 'f"{reason}.{identity[:12]}.json"'))),
        (READER, edits(('f"{day}.{reason}.{identity[:12]}.json"', 'f"{reason}.{identity[:12]}.json"'))),
        (HEALTH, edits(('(f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")',
                        '("invalid_entries.*.json", "unreadable_cache.*.json")'))),
    ]),
    replacement("hash-sample-only", PRODUCER, "dict(observation, outcomes=outcomes)", "dict(observation, outcomes=outcomes[:10])"),
    replacement("hash-id-sample-only", PRODUCER, "dict(observation, outcomes=outcomes)",
                'dict(observation, outcomes=[outcome[0] for outcome in outcomes[:10]] if reason == "invalid_entries" else [])'),
    replacement("drop-yesterday", HEALTH, '(f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")', '(f"{today}.*.json",)'),
    replacement("key-by-reason", HEALTH, "json.dumps(sorted(identities)", "json.dumps(sorted(kinds)"),
    replacement("key-by-label", HEALTH, 'incident_key=f"{SOURCE}:{today}:{digest}"', 'incident_key=f"{SOURCE}:{today}:{labels}"'),
    replacement("follow-status-symlink", READER, "with open_regular(path) as stream:", "with open_regular(path, follow_symlinks=True) as stream:"),
    replacement("skip-existing-row-comparison", PRODUCER,
                '    existing = read_record(path)\n    if any(existing[key] != value for key, value in row.items() if key != "recorded_at"):\n        raise FileExistsError(path)',
                "    read_record(path)"),
    replacement("raw-root-error-as-missing", PRODUCER, "except OSError:\n        return scanned, rejected, True",
                "except OSError:\n        return scanned, rejected, False"),
    replacement("omit-producer-size-limit", PRODUCER, '    if len(payload) > MAX_BYTES:\n        raise ValueError("status exceeds size limit")\n', ""),
    replacement("12-hex-observation-identity", HEALTH, 'identities.append(row["identity"])', 'identities.append(row["identity"][:12])'),
    ("restore-descriptor-leak", [(READER, revert_fd)]),
    ("restore-buffered-stderr", [(PRODUCER, revert_buffered_stderr)]),
    replacement("omit-detail-log", HEALTH, '        log.warning("%s", message(row))', "        pass"),
    replacement("restore-evidence-backticks", HEALTH, "{display(copy)}", "{chr(96)}{display(copy)}{chr(96)}"),
    ("restore-summary-path-without-fallback", [(HEALTH, reinsert_summary_path)]),
    replacement("skip-display-status-log", HEALTH, "display(path), ROUTE", "path, ROUTE"),
    replacement("skip-display-evidence-log", HEALTH, "{display(copy)}", "{copy}"),
    replacement("skip-display-stderr-evidence-failure", PRODUCER, "evidence at {display(target)}", "evidence at {target}"),
    replacement("skip-display-stderr-status-failure", PRODUCER, "status at {display(target)}", "status at {target}"),
    replacement("skip-display-stderr-drop", PRODUCER, "{display(cache)}: dropped", "{cache}: dropped"),
    ("drop-backticks-in-display", [(READER, drop_backticks)]),
    replacement("escape-record-cache-path", PRODUCER, "cache_path=str(cache)", "cache_path=display(cache)"),
    replacement("escape-path-in-producer-identity-only", PRODUCER, "digest(canonical(dict(observation, outcomes=outcomes)))",
                "digest(canonical(dict(observation, cache_path=display(cache), outcomes=outcomes)))"),
    replacement("escape-bad-name-in-incident-key", HEALTH, "identities.append(path.name)", "identities.append(display(path.name))"),
    replacement("restore-stderr-replace-errors", PRODUCER, 'stream.errors or "backslashreplace"', 'errors="replace"'),
]


def invoke(name, work, source, tests, label):
    log = HERE / "mutation-results" / f"{name}.{label}.log"
    args = [sys.executable, str(HERE / "run_focused.py"), str(source),
            "-q", "-x", "-o", "addopts=", "--tb=short", "-p", "no:cacheprovider",
            "--basetemp", str(work / label), *tests]
    options = dict(os.environ, BTS_W15_REAL_ARTIFACTS=("1" if label == "full" else "0"),
                   PYTHONPATH=os.pathsep.join((str(HERE / "offline"), str(source))))
    with log.open("w") as captured:
        completed = subprocess.run(args, cwd=ROOT, env=options, timeout=90, stdout=captured, stderr=subprocess.STDOUT)
    contents = log.read_text()
    if completed.returncode == 0:
        return None
    if completed.returncode == 1:
        first = next((line for line in contents.splitlines() if line.startswith("FAILED tests/")), None)
        if first:
            return first
    raise RuntimeError(f"{name} {label}: exit {completed.returncode}\n{contents[-4000:]}")


def main():
    outcomes = {"KILLED": [], "SURVIVED": []}
    (HERE / "mutation-results").mkdir(exist_ok=True)
    for name, changes in MUTATIONS:
        with TemporaryDirectory(dir=HERE, prefix=f"mut-{name}-") as work:
            work = Path(work)
            source = work / "src"
            shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
            for filename, apply in changes:
                path = source / filename
                new = apply(path.read_text())
                compile(new, str(path), "exec")
                path.write_text(new)
            failed = invoke(name, work, source, W15, "w15")
            if failed is None:
                failed = invoke(name, work, source, FOCUSED, "full")
            result = "KILLED" if failed else "SURVIVED"
            outcomes[result].append(name)
            print(result, name, failed or "passed full selection", flush=True)
    print({key: len(value) for key, value in outcomes.items()}, outcomes["SURVIVED"], flush=True)


if __name__ == "__main__":
    main()
