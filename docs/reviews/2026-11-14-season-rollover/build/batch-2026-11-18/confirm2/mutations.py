"""Run F7 red/green and eight independent mutations against current tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from review_runner import OLD, REVIEW, ROOT, environment

FILE = "tests/experiment/test_artifacts.py"
IMPLEMENTATION = "src/bts/experiment/artifacts.py"
NEW_TESTS = [f"{FILE}::{name}" for name in (
    "test_live_export_refuses_a_profile_shorter_than_its_scored_predictions",
    "test_live_export_counts_only_scored_predictions",
)]
COUNT = '''    scored_rows = {
        "production": _scored_prediction_rows(production_predictions, top_n),
        "candidate": _scored_prediction_rows(candidate_predictions, top_n),
    }
'''
CHECK = '''    for variant_name, variant_profiles in (("production", production_profiles),
                                           ("candidate", candidate_profiles)):
        if len(variant_profiles) != scored_rows[variant_name]:
            raise ValueError(
                f"live {variant_name} profile for {date} has {len(variant_profiles)} rows "
                f"for {scored_rows[variant_name]} scored prediction rows"
            )
'''
PROFILE_COUNT = '''    scored_rows = {
        "production": len(production_profiles),
        "candidate": len(candidate_profiles),
    }
'''
MUTATIONS = {
    "drop-length-check": [(CHECK, "")],
    "converted-count": [(COUNT, ""), (CHECK, PROFILE_COUNT + CHECK)],
    "count-all-predictions": [('int(predictions["p_game_hit"].notna().sum())', "len(predictions)")],
    "uncapped-count": [('min(int(top_n), int(predictions["p_game_hit"].notna().sum()))',
                        'int(predictions["p_game_hit"].notna().sum())')],
    "minimum-variant-count": [("effective_top_n = max(scored_rows.values())",
                               "effective_top_n = min(scored_rows.values())")],
    "allow-bool": [("type(effective) is int and type(requested) is int",
                    "isinstance(effective, int) and type(requested) is int")],
    "allow-negative": [("0 < effective <= requested", "effective != 0 and effective <= requested")],
    "allow-historical-kind": [
        ('run_kind in {"live_forward_preoutcome", "live_forward_resolved"} and "effective_top_n" in manifest',
         '"effective_top_n" in manifest'),
    ],
}


def run_tests(mode, tree, scratch):
    import bts.experiment
    from sitecustomize import check

    check()
    if mode in MUTATIONS:
        bts.experiment.__path__.insert(0, str(scratch / mode))
        expected = scratch / mode / "artifacts.py"
    else:
        expected = tree / IMPLEMENTATION
    from bts.experiment import artifacts
    assert Path(artifacts.__file__) == expected, artifacts.__file__
    import pytest

    selection = [FILE] if mode in MUTATIONS else NEW_TESTS
    return pytest.main(["-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short",
                        "--basetemp", str(scratch / f"pytest-{mode}"), *selection])


def main():
    if len(sys.argv) > 1:
        _, mode, tree, scratch = sys.argv
        return run_tests(mode, Path(tree), Path(scratch))
    scratch = Path(tempfile.mkdtemp(prefix="f7-test-", dir=REVIEW))
    original = (ROOT / IMPLEMENTATION).read_text()
    for mode, replacements in MUTATIONS.items():
        mutant = original
        for expected, replacement in replacements:
            assert mutant.count(expected) == 1, (mode, expected)
            mutant = mutant.replace(expected, replacement)
        target = scratch / mode / "artifacts.py"
        target.parent.mkdir()
        target.write_text(mutant)
    results = {}
    for mode in ("red", "green", *MUTATIONS):
        tree = OLD if mode == "red" else ROOT
        print(f"\n=== {mode} ===", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, mode, str(tree), str(scratch)], cwd=ROOT, env=environment(tree),
        )
        results[mode] = result.returncode
        print(f"{mode}: pytest exit {result.returncode}", flush=True)
    (REVIEW / "mutations.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    return int(results["red"] != 1 or results["green"] != 0 or any(rc not in (0, 1) for rc in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
