"""Run each requested fault in a private HEAD-source copy via the shared runner."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WRAPPER = Path.home() / "bts-runs/yield_run.sh"
CACHE = "bts/features/compute.py"
STATE = "bts/state/regenerate.py"
WRITE = '''        tmp = _probable_pitcher_cache_path.with_name(
            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))
            os.replace(tmp, _probable_pitcher_cache_path)
        finally:
            tmp.unlink(missing_ok=True)'''
WARNING = '''            print(
                f"WARNING: unreadable probable-pitcher cache {_probable_pitcher_cache_path} "
                f"({type(e).__name__}); rebuilding it from raw feeds",
                file=sys.stderr,
            )
'''
MUTATIONS = (
    ("in-place", CACHE, WRITE, '        _probable_pitcher_cache_path.write_text(\n'
                                  '            json.dumps({str(k): v for k, v in lookup.items()}))',
     ('test_interrupted_cache_write_leaves_the_previous_cache_intact', 'test_cache_write_replaces_the_file_atomically')),
    ("no-cleanup", CACHE, WRITE, WRITE.split("        finally:")[0].replace("        try:\n", "")
     .replace("            tmp.write_text", "        tmp.write_text").replace("            os.replace", "        os.replace").rstrip(),
     ('test_interrupted_cache_write_leaves_the_previous_cache_intact',)),
    ("no-warning", CACHE, WARNING, "", ('test_unreadable_cache_is_reported_and_rebuilt',)),
    ("swallow-error", CACHE, "        finally:\n            tmp.unlink(missing_ok=True)",
     "        except OSError:\n            pass\n        finally:\n            tmp.unlink(missing_ok=True)",
     ('test_interrupted_cache_write_leaves_the_previous_cache_intact',)),
    ("substring-void", STATE, "_VOID_RE.fullmatch(text.strip())", "_VOID_RE.search(text.strip())", ()),
    ("no-void-streak", STATE, 'result="void", streak_after=int(void.group(1))', 'result="void", streak_after=None',
     ('test_void_reply_round_trips', 'test_void_day_keeps_the_streak_and_the_saver_in_the_timeline')),
    ("void-spends-saver", STATE, '        if r.result != "miss":', '        if r.result not in ("miss", "void"):',
     ('test_void_day_keeps_the_streak_and_the_saver_in_the_timeline',)),
)


def run_mutation(name, target, before, after, failures):
    log = HERE / f"mut-{name}.log"
    with TemporaryDirectory(prefix=f"mut-{name}-", dir=HERE) as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", src)
        path = src / target
        code = path.read_text()
        assert code.count(before) == 1, (name, before)
        mutant = code.replace(before, after)
        compile(mutant, str(path), "exec")
        path.write_text(mutant)
        env = os.environ | {"PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1", "BTS_YIELD_SECONDS": "2"}
        argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "tests/test_leads_small_fixes.py",
                "-q", "--tb=short"]
        with subprocess.Popen([str(WRAPPER), str(log.relative_to(ROOT)), *argv], cwd=ROOT, env=env) as wrapped:
            exit_code = wrapped.wait()
        evidence = log.read_text()
    assert "collected 14 items" in evidence
    assert exit_code == bool(failures), (name, exit_code, evidence)
    failed = [line for line in evidence.splitlines() if line.startswith("FAILED ")]
    actual = {line.split("::", 1)[1].split(" ", 1)[0].split("[", 1)[0] for line in failed}
    assert actual == set(failures), (name, actual, failures)
    print(name, f"exit={exit_code}", [line.split("::", 1)[1] for line in failed], flush=True)


if __name__ == "__main__":
    for mutation in MUTATIONS:
        run_mutation(*mutation)
