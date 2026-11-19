"""Run six isolated regressions with the same two WR-B test files."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

review = Path(__file__).resolve().parent
original = review / "head-tree"
root = review.parents[1]
wrapper = Path.home() / "bts-runs" / "yield_run.sh"

mutants = {
    "omit": ("season_contract.py", "verdict(omit_selected_dated=True) if dated", "verdict() if dated"),
    "always-accepted": ("season_contract.py",
        'lines.append(f"automatic path alone: refused: {refusal.splitlines()[0]}")',
        'lines.append("automatic path alone: accepted (streak 0, recorded unavailable)")'),
    "bare-command": ("health/contest_state.py",
        'f"with `{dated_template(picks_dir, now)}`. Replace N, B and S with MLB\'s "',
        '"with bts set-contest-streak. Replace N, B and S with MLB\'s "'),
    "fix-it": ("cli.py", 'and decision.get("delivery_status") == "locked_unconfirmed"):',
        'and decision.get("delivery_status") == "disabled_unconfirmed"):'),
    "two-names": ("day_outcome.py",
        'legs = ("primary", "double_down") if decision.get("action") == "double" else ("primary",)',
        'legs = ("primary", "double_down")'),
    "non-object-error": ("cli.py", "if not isinstance(status, dict):\n        return {}",
        'if not isinstance(status, dict):\n        raise TypeError("expected object")'),
}

for label, (filename, old, new) in mutants.items():
    tree = review / f"mutant-{label}"
    tree.mkdir(exist_ok=True)
    for directory in ("src", "tests"):
        shutil.copytree(original / directory, tree / directory, dirs_exist_ok=True)
    shutil.copytree(original / "docs/ops", tree / "docs/ops", dirs_exist_ok=True)
    shutil.copy2(original / "pyproject.toml", tree)
    path = tree / "src/bts" / filename
    content = path.read_text()
    assert content.count(old) == 1, label
    path.write_text(content.replace(old, new, 1))
    log = review / f"mutant-{label}.log"
    environment = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1",
                       WRB_TREE=str(tree), BTS_YIELD_SECONDS="3")
    result = subprocess.run([
        str(wrapper), str(log), sys.executable, str(review / "pytest_offline.py"),
        "-p", "no:cacheprovider", "tests/test_wr_b_contract.py", "tests/test_wr_b_edges.py",
        "-q", "--tb=short",
    ], cwd=root, env=environment)
    print(f"{label}: {result.returncode}", flush=True)
    if result.returncode not in (0, 1):
        raise SystemExit(result.returncode)
