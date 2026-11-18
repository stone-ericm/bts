"""Replay the unmodified audits and test the new canary's discrimination."""

import ast
import contextlib
import io
import os
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

from probe_common import BASE, DATA, FEATURE, HERE, LOOKUP_PATH, ROOT, load_base, run, toy


def stage():
    staged = HERE / "audit-root"
    processed = staged / "data" / "processed"
    models = staged / "data" / "models"
    processed.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    for year in (2023, 2024, 2025):
        target = processed / f"pa_{year}.parquet"
        if not target.is_symlink():
            target.symlink_to(DATA / target.name)
    target = models / "probable_pitcher_lookup.json"
    if not target.is_symlink():
        target.symlink_to(LOOKUP_PATH)
    assert sorted(path.name for path in processed.iterdir()) == [
        "pa_2023.parquet", "pa_2024.parquet", "pa_2025.parquet"
    ]
    assert not (staged / "data" / "raw").exists()
    return staged


def execute(label, source, script, staged):
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=staged,
        env=dict(os.environ, PYTHONPATH=str(source / "src")),
        capture_output=True, text=True,
    )
    log = HERE / f"audit-{label}.txt"
    log.write_text(proc.stdout + proc.stderr)
    print(label, "process exit", proc.returncode, flush=True)
    return proc.returncode, proc.stdout


def check_canary(script):
    from bts.features import compute

    # Execute precisely the checked-in definitions, isolated from the expensive
    # top-level real-data audit. The separate runs above execute the entire file.
    syntax = ast.parse(script.read_text(), filename=str(script))
    syntax.body = [
        node for node in syntax.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    namespace = {}
    exec(compile(syntax, str(script), "exec"), namespace)
    canary = namespace["audit_bullpen_availability"]
    outputs = []
    for attempt in range(3):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            assert canary()
        outputs.append(log.getvalue())
    assert len(set(outputs)) == 1
    print("three identical branch canary passes", outputs[0].strip(), flush=True)

    real = compute.compute_all_features

    def never_warmed(frame):
        frame = frame.copy()
        frame.loc[frame["date"].lt(frame["date"].max()), "pitcher_id"] = 21
        return real(frame)

    with patch.object(compute, "compute_all_features", side_effect=never_warmed):
        assert not canary()
    print("canary rejects an unwarmed team", flush=True)

    base = load_base()

    def only_regress_on_multi_pa_date(frame):
        if frame["date"].value_counts().max() == 1:
            return real(frame)
        with patch.object(base, "_build_probable_pitcher_lookup",
                          compute._build_probable_pitcher_lookup):
            return base.compute_all_features(frame)

    dates = pd.date_range("2025-04-01", periods=11)
    frame, lookup = toy([(date, 51, 21, 22, int(i < 4)) for i, date in enumerate(dates[:10])]
                        + [(dates[-1], 51, 21, 21, 0)] * 2)
    relief = frame.copy()
    relief.loc[relief.index[-1], "pitcher_id"] = 22
    with patch.object(compute, "compute_all_features", side_effect=only_regress_on_multi_pa_date):
        assert canary()
        old_gap = run(compute, frame, lookup)[FEATURE].iloc[-2]
        filled_gap = run(compute, relief, lookup)[FEATURE].iloc[-2]
    assert pd.isna(old_gap) and filled_gap == 0.4
    print("canary passes a conditional defect replicated on a two-PA date:", old_gap, filled_gap,
          flush=True)


def main():
    staged = stage()
    script = ROOT / "scripts" / "leakage_audit.py"
    relative_script = script.relative_to(ROOT)
    clean_base, old = execute("base-pristine", BASE, BASE / relative_script, staged)
    patched_base, red = execute("base-with-canary", BASE, script, staged)
    branch, green = execute("branch", ROOT, script, staged)
    assert (clean_base, patched_base, branch) == (0, 1, 0)
    canary_heading = "\n--- AUDIT 5:"
    assert red.split(canary_heading)[0].rstrip() == old.rstrip()
    assert green.split(canary_heading)[0].rstrip() == old.rstrip()
    assert "all starter: [nan] vs 0.4 → FAIL" in red
    assert "all relief: [0.4] vs 0.4 → PASS" in red
    assert "all starter: [0.4] vs 0.4 → PASS" in green
    assert "all relief: [0.4] vs 0.4 → PASS" in green
    print("existing audit stdout identical across three executions", flush=True)
    check_canary(script)


if __name__ == "__main__":
    main()
