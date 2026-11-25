import os
from pathlib import Path
import shutil
import subprocess
import sys


work = Path(__file__).resolve().parent
repo = work.parent.parent
scratch = work / 'mutant'
if not scratch.exists():
    shutil.copytree(repo / 'src', scratch / 'src', ignore=shutil.ignore_patterns('__pycache__'))
originals = {name: (repo / 'src/bts' / name).read_text() for name in ('scheduler.py', 'web.py')}
mutations = [
    ('top-prediction', 'scheduler.py', 'best_batter=name if', 'best_batter=summary["best_batter"] if', 'candidate_override'),
    ('below-on-equality', 'scheduler.py', 'if probability < bar:', 'if probability <= bar:', 'skip_surfaces'),
    ('unlabeled-K2', 'scheduler.py', '{description}, below {bar_text}', '{description.removeprefix("Top model prediction (eligibility unverified): ")}, below {bar_text}', 'skip_surfaces'),
    ('K5-before-K1', 'scheduler.py', 'if reason:\n        ending', 'if reason and fields.get("basis") != "no_decision":\n        ending', 'skip_surfaces'),
    ('K2-before-K1', 'scheduler.py', 'if reason:\n        ending', 'if reason and not (bar is not None and probability is not None and probability < bar):\n        ending', 'skip_surfaces'),
    ('old-K1-journal', 'scheduler.py', 'print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', 'print("  SKIP — best; streak holds." if skip_summary.get("reason") else f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', 'run_single_check_skip_journal'),
    ('old-K3-journal', 'scheduler.py', 'print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', 'print("  SKIP — below the pick bar." if "at or above" in format_skip_summary(skip_summary) else f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', 'run_single_check_skip_journal'),
    ('old-K4-journal', 'scheduler.py', 'print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', 'print("  SKIP — below the pick bar." if "No probability comparison" in format_skip_summary(skip_summary) else f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', 'run_single_check_skip_journal'),
    ('unescaped-name', 'web.py', 'body = html_lib.escape(format_skip_summary(summary))', 'body = html_lib.escape(format_skip_summary(summary))\n    if isinstance(summary.get("best_batter"), str):\n        body = body.replace(html_lib.escape(" ".join(summary["best_batter"].splitlines())), " ".join(summary["best_batter"].splitlines()))', 'malformed_persisted_fields'),
    ('unescaped-team', 'web.py', 'body = html_lib.escape(format_skip_summary(summary))', 'body = html_lib.escape(format_skip_summary(summary))\n    if isinstance(summary.get("best_team"), str):\n        body = body.replace(html_lib.escape(summary["best_team"]), summary["best_team"])', 'skip_surfaces'),
    ('unescaped-reason', 'web.py', 'body = html_lib.escape(format_skip_summary(summary))', 'body = html_lib.escape(format_skip_summary(summary))\n    if isinstance(summary.get("reason"), str):\n        body = body.replace(html_lib.escape(summary["reason"]), summary["reason"])', 'skip_surfaces'),
    ('missing-delivery-saver', 'scheduler.py', '"saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts "\n            "saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."', '"saver used, check bts saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."', 'maximum_detail'),
    ('missing-morning-saver', 'scheduler.py', '"saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "\n            "DIR. For uninitialized, see B8a. Check the latest fetch status."', '"saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."', 'maximum_detail'),
    ('old-late-refusal', 'scheduler.py', '"The entry check may report a mismatch if it verifies the entry in time.")', '"Enter a pick manually if you want one now.")', 'refusal_message'),
    ('old-void-refusal', 'scheduler.py', '"The entry check may report a mismatch if it verifies the entry in time.")', '"Enter a pick manually if you want one today.")', 'refusal_message'),
    ('string-number-raises', 'scheduler.py', 'if type(value) not in (int, float):', 'if isinstance(value, str):\n        return math.isfinite(value)\n    if type(value) not in (int, float):', 'malformed_persisted_fields'),
    ('old-heading', 'web.py', 'LAST RECORDED NO-PICK SUMMARY', 'NO PICK AT LAST CHECK', 'skip_surfaces'),
    ('old-pointer', 'scheduler.py', 'f"see B8a. On {warning_day.isoformat()}', 'f"follow B8b(1). On {warning_day.isoformat()}', 'maximum_detail'),
    ('no-first-check-guard', 'scheduler.py', 'Before the first check, if prior entries', 'If prior entries', 'maximum_detail'),
]

driver = 'import bts, pathlib, pytest, sys; assert pathlib.Path(bts.__file__).is_relative_to(sys.argv[1]); sys.exit(pytest.main(sys.argv[2:]))'
for label, filename, old, new, selector in mutations:
    source = originals[filename]
    if label == 'old-void-refusal':
        i = source.rindex(old)
    else:
        i = source.index(old)
    target = scratch / 'src/bts' / filename
    target.write_text(source[:i] + new + source[i + len(old):])
    try:
        result = subprocess.run(
            [sys.executable, '-B', '-c', driver, str(scratch), '-p', 'no:cacheprovider',
             'tests/test_wr_e_texts.py', '-k', selector, '-q', '--tb=short'],
            cwd=repo, env=dict(os.environ, PYTHONPATH=f'{scratch / "src"}:{repo}'),
            capture_output=True, text=True, timeout=30,
        )
        (work / ('mutation-' + label + '.log')).write_text(result.stdout + result.stderr)
        assert result.returncode in {0, 1}, (label, result.stdout, result.stderr)
        if result.returncode == 1:
            assert 'FAILED tests/test_wr_e_texts.py::' in result.stdout, (label, result.stdout)
        print(label, 'caught' if result.returncode else 'survived selected tests', flush=True)
    finally:
        target.write_text(source)
