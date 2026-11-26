"""Run only touched tests, base controls and discriminating source mutants.

Invoke this script through yield_run.sh. Each pytest child loads our offline
audit plugin and independent clock; no tracked files are changed.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(OUT))
import review_support  # noqa: E402,F401

changed = subprocess.check_output(["git", "diff", "--name-only", "0bbfc8d4", "HEAD"], text=True).splitlines()
tests = [p for p in changed if p.startswith("tests/") and p != "tests/conftest.py"]
new = ["tests/test_season_contract_cs4b2.py", "tests/test_season_contract_cs4b2_edges.py"]
results = []


def run(label, source, year, paths, selection=None):
    env = dict(os.environ, REVIEW_SRC=str(source), REVIEW_YEAR=str(year),
               PYTHONPATH=str(OUT) + os.pathsep + str(ROOT),
               PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "review_support", *paths,
           "-q", "--tb=short", "-r", "fEs", "--basetemp", str(OUT / "test-work" / label)]
    if selection:
        cmd += ["-k", selection]
    log = OUT / (label + ".log")
    print("START", label, flush=True)
    with log.open("w") as stream:
        answer = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    tail = log.read_text().splitlines()[-4:]
    results.append(dict(label=label, exit=answer.returncode, tail=tail))
    (OUT / "test-results.json").write_text(json.dumps(results, indent=2))
    print("END", label, answer.returncode, tail, flush=True)


for year in [2027, 2028]:
    run(f"touched-{year}", ROOT / "src", year, tests)
run("base-new", OUT / "base/src", 2027, new)

mutations = {
    "no-refusal": ("contest_state.py", "if require_contest_state and not (", "if False and not (", "selection_matrix"),
    "invert-dated": ("contest_state.py", "contest is not None and dated_observation_in_effect(contest, now_dt)):", "contest is not None and not dated_observation_in_effect(contest, now_dt)):", "selection_matrix"),
    "drop-corrupt": ("season_contract.py", 'if state.state == "corrupt":\n        kind = 1', 'if False:\n        kind = 1', "selection_matrix"),
    "drop-manual": ("season_contract.py", 'elif selected is not None and selected.path.name == "contest_streak.manual.json":', 'elif False:', "selection_matrix"),
    "drop-switch": ("season_contract.py", 'elif state.state in {"absent", "other_season", "revoked"}:', 'elif False:', "selection_matrix"),
    "drop-missing": ("season_contract.py", 'elif selected is None:', 'elif False:', "selection_matrix"),
    "drop-schema": ("season_contract.py", 'if selected.schema_version != AUTO_SCHEMA:', 'if False:', "selection_matrix"),
    "drop-season": ("season_contract.py", 'elif selected.season != year:', 'elif False:', "selection_matrix"),
    "drop-id": ("season_contract.py", 'elif selected.activation_id != state.snapshot["activation_id"]:', 'elif False:', "selection_matrix"),
    "no-health": ("health/contest_state.py", '        append_refusal()\n', '        pass\n', "selection_matrix"),
    "continue-switch": ("cli.py", '        raise SystemExit(1)\n\n    stamp = utc.strftime', '        pass\n\n    stamp = utc.strftime', "revoke_failure_prefixes"),
    "continue-manual": ("cli.py", '        raise SystemExit(1)\n    try:\n        saver = set_aside', '        manual = "failed"\n    try:\n        saver = set_aside', "revoke_failure_prefixes"),
    "continue-saver": ("cli.py", '        raise SystemExit(1)\n\n    for line in switch_lines(SwitchState("revoked", data)):', '        saver = "failed"\n\n    for line in switch_lines(SwitchState("revoked", data)):', "revoke_failure_prefixes"),
    "replace-target": ("cli.py", '            raise FileExistsError(f"target already exists: {target}")', '            pass', "revoke_failure_prefixes"),
}
for label, (filename, old, new_text, selection) in mutations.items():
    source = OUT / "mutants" / label / "src"
    shutil.copytree(ROOT / "src/bts", source / "bts", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    path = source / "bts" / filename
    text = path.read_text()
    assert text.count(old) == 1, (label, text.count(old))
    path.write_text(text.replace(old, new_text))
    run("mutant-" + label, source, 2027, ["tests/test_season_contract_cs4b2.py"], selection)
print(json.dumps(results, indent=2), flush=True)
