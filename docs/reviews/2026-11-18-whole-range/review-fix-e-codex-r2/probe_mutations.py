"""Make one change per scratch copy. A detection must be an assertion failure."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fixtures import HERE, ROOT


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
    raise ValueError(label)


label = sys.argv[1]
file, old, new, index = mutation(label)
with tempfile.TemporaryDirectory(dir=HERE) as name:
    source = Path(name) / "src"
    shutil.copytree(ROOT / "src/bts", source / "bts", ignore=shutil.ignore_patterns("__pycache__"))
    path = source / "bts" / file
    original = path.read_text()
    parts = original.split(old)
    assert len(parts) == (3 if label.startswith("old-refusal-") else 2)
    parts[index] += new + parts.pop(index + 1)
    path.write_text(old.join(parts))
    env = os.environ | {"WR_E_MUTATION_SRC": str(source), "PYTHONPATH": str(source)}
    args = [sys.executable, str(HERE / "run_suite.py"), "-x", "tests/test_wr_e_texts.py"]
    if len(sys.argv) > 2:
        args = [sys.executable, str(HERE / "run_suite.py"), "-x", "-k", "not socket_wiring", *sys.argv[2:]]
    result = subprocess.run(args, cwd=ROOT, env=env, timeout=70, capture_output=True, text=True)
    print(result.stdout, result.stderr)
    assert result.returncode in (0, 1), result.returncode
    if result.returncode:
        assert "FAILED " in result.stdout and "ERROR " not in result.stdout
    print(label, "DETECTED" if result.returncode else "SURVIVED", flush=True)
