"""Check existing assertions against single scratch edits of 428cfa93."""
from ast import Assign, Name, literal_eval, parse
from pathlib import Path
from shutil import copy2
from subprocess import run
from tempfile import TemporaryDirectory
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORKFLOW = Path(".github/workflows/deploy.yml")
TEST = Path("tests/test_deploy_workflow.py")
MARK = '''                    if [[ $name == "$scheduler" ]]; then
                        fs 10 mutation mark "$lock_dir/legacy.$RUN_ID.$RUN_ATTEMPT" >/dev/null
                        step_status legacy-units "$?"
                    fi
'''
STOP = '''                    system_unit 20 mutation stop "$name" >/dev/null
                    step_status legacy-units "$?"
'''
MUTANTS = "d3b_r1_phase_two_legacy_cleanup or d3b_r1_mark_failure or d3b_r1_a_passing_deploy"
LEGACY_CLAUSES = "d3b_r1_each_legacy_stop_clause"
OLD = next(literal_eval(item.value) for item in parse((HERE.parent / "mutations.py").read_text()).body
           if isinstance(item, Assign) and any(isinstance(t, Name) and t.id == "CASES" for t in item.targets))
ADAPTED = {
    "forget-legacy-group": (
        '[[ -z $legacy_group ]] || fs 20 normal cgroup /sys/fs/cgroup "$legacy_group" || system_proof=1',
        ':', "d3b_recorded_legacy_group",
    ),
    "report-outside-prove": (
        '                prove report\n                exit 1',
        "                prove report\n                log 'Scheduler stopped: verified.'\n                exit 1",
        "only_the_proof_function",
    ),
    "follow-redirect": (
        'curl --disable --globoff --path-as-is --noproxy',
        'curl --disable --globoff --path-as-is -L --noproxy', "round2_failure_post_never_follows",
    ),
    "extra-report-time": (
        'timeout -k 1 1 /bin/bash -c', 'timeout -k 5 1 /bin/bash -c',
        "d3b_r1_final_lines_are_one_bounded_print or d3b_containment_windows or d3b_recovery_reproves_stop_and_marker",
    ),
}
CASES = [(label, *ADAPTED.get(label, (old, new, selection))) for label, old, new, selection in OLD]
CASES += [
    ("skip-mark-in-first-sample", "no_fragment && ! legacy_cleaned; then",
     "no_fragment; then", f"{MUTANTS} or d3b_legacy_absent_throughout"),
    ("skip-mark-in-proof", "if ! legacy_cleaned && post_group=", "if post_group=",
     f"{MUTANTS} or d3b_r1_a_cgroup_reported"),
    ("write-mark-after-stop", MARK, "", MUTANTS),
    ("drop-path-as-is", " --path-as-is", "", "d3b_r1_failure_post_sends_the_path_as_written"),
    ("allow-percent-escapes", 'if "%" in value or any(part', 'if any(part', "d3b_r1_page_url"),
    ("drop-loaded-requirement", 'elif [[ $legacy == loaded ]] && (( legacy_group_status',
     'elif (( legacy_group_status', "d3b_r1_not_found_with_a_fragment or d3b_unproved_legacy_absence"),
    ("ignore-legacy-previous-query-failure", 'elif (( legacy_group_status == 0 )) && [[ -n $legacy_group',
     'elif [[ -n $legacy_group', LEGACY_CLAUSES),
    ("ignore-legacy-new-group", '[[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
     '                                        fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1',
     ":", LEGACY_CLAUSES),
    ("ignore-legacy-stop-status", '[[ $legacy_absent == 0 || $legacy_disabled$legacy_stopped == 00 ]]',
     "true", LEGACY_CLAUSES),
    ("ignore-legacy-enable-state", '&& enabled disabled 20 system_unit; then', '; then', LEGACY_CLAUSES),
    ("drop-rounding-allowance", '[[ -z ${budget_start:-} ]] || available=$((available - 1))',
     ":", "d3b_r1_rounding_allowance"),
]


if "--check" in sys.argv:
    source = (ROOT / WORKFLOW).read_text()
    for label, old, new, _ in CASES:
        assert old in source, label
        print(label, source.count(old))
    raise SystemExit(0)

with TemporaryDirectory(prefix="mutation-", dir=HERE) as name:
    scratch = Path(name)
    for relative in (WORKFLOW, TEST):
        (scratch / relative).parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT / relative, scratch / relative)
    original = (scratch / WORKFLOW).read_text()
    for label, old, new, selection in CASES:
        assert old in original, label
        source = original.replace(old, new)
        if label == "write-mark-after-stop":
            assert source.count(STOP) == 1
            source = source.replace(STOP, STOP + MARK)
        (scratch / WORKFLOW).write_text(source)
        result = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", str(TEST), "-k",
                      selection, "-q", "--tb=short"], cwd=scratch, timeout=70, capture_output=True, text=True)
        print(f"{label}: rc={result.returncode} ({selection})\n{result.stdout}{result.stderr}", flush=True)
        if result.returncode not in (0, 1) or " passed" not in result.stdout and " failed" not in result.stdout:
            raise SystemExit(f"{label}: selection did not run cleanly")
