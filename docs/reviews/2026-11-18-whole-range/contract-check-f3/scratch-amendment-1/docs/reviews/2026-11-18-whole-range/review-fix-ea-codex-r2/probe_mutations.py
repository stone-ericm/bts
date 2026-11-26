"""Make one change per scratch copy. A detection must be an assertion failure."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def mutation(label):
    variants = {
        "top-prediction": ("""            summary.update(
                best_batter=name if isinstance(name, str) else "?",
                best_team=team if isinstance(team, str) else "?",
                best_p=probability,
                candidate_verified=True,
            )""", '            summary["candidate_verified"] = True'),
        "below-equality": ("if probability < bar:", "if probability <= bar:"),
        "drop-label-k2": ('sentence = f"{description}, below {bar_text} using unrounded values."',
                          'sentence = f"{description}, below {bar_text} using unrounded values.".replace(" (eligibility unverified)", "")'),
        "swap-k1-k5": ("    if reason:\n        ending", '    if reason and fields.get("basis") != "no_decision":\n        ending'),
        "swap-k1-k2": ("    if reason:\n        ending", "    if reason and not (probability is not None and bar is not None and probability < bar):\n        ending"),
        "drop-delivery-saver": ('''"saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts "
            "saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."''',
                                '"saver used, check bts saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."'),
        "drop-morning-saver": ('''"saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "
            "DIR. For uninitialized, see B8a. Check the latest fetch status."''',
                               '"saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."'),
        "string-probability": ('    probability = _skip_display_number(fields.get("best_p"))',
                               '    if isinstance(fields.get("best_p"), str):\n        raise TypeError("string probability")\n'
                               '    probability = _skip_display_number(fields.get("best_p"))'),
    }
    if label in variants:
        return "scheduler.py", *variants[label], 0
    if label.startswith("old-journal-"):
        suffix = label.rsplit("-", 1)[1]
        phrase = {"k1": ". Reason:", "k3": ", at or above ", "k4": "No probability comparison"}[suffix]
        return "scheduler.py", '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)', (
            "                body = format_skip_summary(skip_summary)\n"
            f"                if {phrase!r} in body:\n"
            '                    body += " Will pick if a confirmed lineup clears it."\n'
            '                print(f"  SKIP — {body}", file=sys.stderr)'
        ), 0
    if label.startswith("unescaped-"):
        field = {"name": "best_batter", "team": "best_team", "reason": "reason"}[label.split("-")[-1]]
        original = "    body = html_lib.escape(format_skip_summary(summary))"
        changed = original + f"\n    raw = summary.get({field!r})\n" + (
            "    if isinstance(raw, str) and raw:\n"
            '        normalized = " ".join(raw.splitlines())\n'
            "        body = body.replace(html_lib.escape(normalized), normalized, 1)"
        )
        return "web.py", original, changed, 0
    if label.startswith("old-refusal-"):
        late = label.endswith("late")
        return ("scheduler.py", '"The entry check may report a mismatch if it verifies the entry in time.")',
                f'"Enter a pick manually if you want one {"now" if late else "today"}.")', int(not late))

    updates = {
        "old-heading": ("web.py", "LAST RECORDED NO-PICK SUMMARY", "NO PICK AT LAST CHECK"),
        "old-pointer": ("scheduler.py", '"For a later slate date, see B8a. "',
                        '"For a later date, follow B8b(1). "'),
        "no-first-check-guard": ("scheduler.py", "Before the first check, if prior entries",
                                 "If prior entries"),
        "restore-k5-streak": ("scheduler.py", "if streak is not None and not no_decision:",
                              "if streak is not None:"),
        "restore-late-headline": ("scheduler.py",
            "the {cutoff.strftime('%H:%M ET')} cutoff for {names} had been reached.",
            "{names} would have been sent {late_min:.0f} min after the {cutoff.strftime('%H:%M ET')} cutoff."),
        "restore-later-date": ("scheduler.py", "For a later slate date, see B8a.", "For a later date, see B8a."),
        "drop-stored-no-decision-streak": ("scheduler.py",
            '    summary["candidate_verified"] = False',
            '    if decision is None:\n        summary.pop("streak")\n    summary["candidate_verified"] = False'),
        "candidate-class-lookup-raises": ("scheduler.py",
            '        try:\n            if not isinstance(candidate, Mapping):\n                return summary',
            '        if not isinstance(candidate, Mapping):\n            return summary\n        try:'),
    }
    if label in updates:
        return (*updates[label], 0)
    raise ValueError(label)



groups = [
    ["top-prediction", "below-equality", "drop-label-k2", "swap-k1-k5", "swap-k1-k2",
     "drop-delivery-saver", "drop-morning-saver", "string-probability"],
    ["old-journal-k1", "old-journal-k3", "old-journal-k4", "unescaped-name",
     "unescaped-team", "unescaped-reason", "old-refusal-late", "old-refusal-void"],
    ["old-heading", "old-pointer", "no-first-check-guard", "restore-k5-streak",
     "restore-late-headline", "restore-later-date", "drop-stored-no-decision-streak",
     "candidate-class-lookup-raises"],
]

driver = (
    "from pathlib import Path; import bts, runpy, sys; "
    "assert Path(bts.__file__).is_relative_to(sys.argv[1]), bts.__file__; "
    "sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
)
for label in groups[int(sys.argv[1])]:
    with tempfile.TemporaryDirectory(dir=HERE) as name:
        source = Path(name) / "src"
        shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
        file, old, new, index = mutation(label)
        target = source / "bts" / file
        parts = target.read_text().split(old)
        assert len(parts) == (3 if label.startswith("old-refusal-") else 2), label
        parts[index] += new + parts.pop(index + 1)
        target.write_text(old.join(parts))
        args = [
            sys.executable, "-B", "-c", driver, str(source), str(HERE / "run_suite.py"),
            "-p", "no:cacheprovider", "-q", "--tb=short", "-x", "tests/test_wr_e_texts.py",
        ]
        result = subprocess.run(args, cwd=ROOT, env=os.environ | {"PYTHONPATH": f"{source}:{ROOT}"},
                                timeout=14, capture_output=True, text=True)
        (HERE / f"mutation-{label}.log").write_text(result.stdout + result.stderr)
        assert result.returncode in (0, 1), (label, result.returncode)
        if result.returncode:
            assert "FAILED tests/test_wr_e_texts.py::" in result.stdout, label
        print(label, "DETECTED" if result.returncode else "SURVIVED", flush=True)
