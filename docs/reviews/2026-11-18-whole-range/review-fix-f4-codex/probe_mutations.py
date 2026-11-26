"""Apply one deliberately broken change to an external scratch copy, then run pytest."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
READ = '_streak_problem(Path(picks_dir) / "streak.json")'
RESULT = f"    return replace(outcome, streak_problem={READ})\n"


def change(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def mutate(src, mutation):
    path = src / "bts" / ("util.py" if mutation == "unbounded" else "day_outcome.py")
    text = path.read_text()
    if mutation == "early":
        text = change(text, "    void_games = 0\n", f"    streak_problem = {READ}\n    void_games = 0\n")
        text = change(text, RESULT, "    return replace(outcome, streak_problem=streak_problem)\n")
    elif mutation == "missing":
        text = change(text, "except FileNotFoundError:\n        return None",
                      'except FileNotFoundError:\n        return "unexpected missing file"')
    elif mutation == "bool":
        text = change(text, "    if type(streak) is not int or streak < 0:\n",
                      "    if not isinstance(streak, int) or streak < 0:\n")
    elif mutation == "unbounded":
        text = change(text, "            while len(data) <= max_bytes:\n"
                            "                chunk = os.read(fd, max_bytes + 1 - len(data))\n",
                      "            while True:\n                chunk = os.read(fd, 65536)\n")
        text = change(text, "            if len(data) > max_bytes:\n"
                            '                raise FileReadLimitError(f"{path}: exceeds read limit")\n', "")
    elif mutation == "no-compact":
        begin = text.index("        if len(body) > DM_MAX_CODE_POINTS:\n")
        end = text.index("        return body\n", begin)
        text = text[:begin] + text[end:]
    elif mutation == "recursion":
        text = change(text, "    except (ValueError, RecursionError) as exc:\n", "    except ValueError as exc:\n")
    elif mutation == "clean-head":
        text = change(text, "        if self.streak_problem is not None and self.state in OK_STATES:\n"
                            '            title = "MODEL STREAK FILE NEEDS REVIEW"\n', "")
    else:
        raise ValueError(mutation)
    path.write_text(text)


mutation, *test_paths = sys.argv[1:]
assert test_paths
with TemporaryDirectory(prefix=f"wr-f4-mutation-{mutation}-", dir="/tmp") as tmp:
    src = Path(tmp) / "src"
    shutil.copytree(ROOT / "src", src)
    mutate(src, mutation)
    env = os.environ | {"PYTHONPATH": os.pathsep.join((str(src), str(ROOT))), "PYTHONDONTWRITEBYTECODE": "1"}
    runner = (
        "import sys, bts, pytest; "
        "print('scratch bts.__file__ =', bts.__file__, flush=True); "
        "assert 'wr-f4-mutation-' in bts.__file__; "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    pytest_args = ["-o", "addopts=", "-p", "no:cacheprovider", "-q", "-x", "--tb=short", *test_paths]
    result = subprocess.run([sys.executable, "-c", runner, *pytest_args], env=env, cwd=ROOT, timeout=108)
    print(f"{mutation}: pytest exit={result.returncode}", flush=True)
    raise SystemExit(result.returncode)
