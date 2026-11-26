"""Materialize scratch source copies for mutation testing without touching any tracked file."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
REPO_SRC = REVIEW_DIR.parent.parent / "src"
MUTANTS_ROOT = REVIEW_DIR / "mutants"

COMPUTE_ORIG_WRITE = """\
        tmp = _probable_pitcher_cache_path.with_name(
            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))
            os.replace(tmp, _probable_pitcher_cache_path)
        finally:
            tmp.unlink(missing_ok=True)"""

COMPUTE_ORIG_WARN = """\
            print(
                f"WARNING: unreadable probable-pitcher cache {_probable_pitcher_cache_path} "
                f"({type(e).__name__}); rebuilding it from raw feeds",
                file=sys.stderr,
            )"""


MUTATIONS = {
    "m1_write_in_place": (
        "bts/features/compute.py",
        COMPUTE_ORIG_WRITE,
        "        _probable_pitcher_cache_path.write_text(\n"
        "            json.dumps({str(k): v for k, v in lookup.items()})\n"
        "        )",
    ),
    "m2_no_tmp_cleanup": (
        "bts/features/compute.py",
        COMPUTE_ORIG_WRITE,
        '        tmp = _probable_pitcher_cache_path.with_name(\n'
        '            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"\n'
        '        )\n'
        '        tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))\n'
        '        os.replace(tmp, _probable_pitcher_cache_path)',
    ),
    "m3_drop_warning": (
        "bts/features/compute.py",
        COMPUTE_ORIG_WARN,
        "            pass",
    ),
    "m4_swallow_write_error": (
        "bts/features/compute.py",
        COMPUTE_ORIG_WRITE,
        '        tmp = _probable_pitcher_cache_path.with_name(\n'
        '            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"\n'
        '        )\n'
        '        try:\n'
        '            tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))\n'
        '            os.replace(tmp, _probable_pitcher_cache_path)\n'
        '        except Exception:\n'
        '            pass\n'
        '        finally:\n'
        '            tmp.unlink(missing_ok=True)',
    ),
    "m5_void_substring_search": (
        "bts/state/regenerate.py",
        "    void = _VOID_RE.fullmatch(text.strip())",
        "    void = _VOID_RE.search(text.strip())",
    ),
    "m6_void_streak_none": (
        "bts/state/regenerate.py",
        '            is_result=True, result="void", streak_after=int(void.group(1)),',
        '            is_result=True, result="void", streak_after=None,',
    ),
    "m7a_void_parsed_as_miss": (
        "bts/state/regenerate.py",
        '            is_result=True, result="void", streak_after=int(void.group(1)),',
        '            is_result=True, result="miss", streak_after=int(void.group(1)),',
    ),
    "m7b_void_treated_as_miss_in_timeline": (
        "bts/state/regenerate.py",
        '        if r.result != "miss":',
        '        if r.result not in ("miss", "void"):',
    ),
    "m8_cleanup_exception_only": (
        "bts/features/compute.py",
        COMPUTE_ORIG_WRITE,
        '        tmp = _probable_pitcher_cache_path.with_name(\n'
        '            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"\n'
        '        )\n'
        '        try:\n'
        '            tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))\n'
        '            os.replace(tmp, _probable_pitcher_cache_path)\n'
        '        except Exception:\n'
        '            tmp.unlink(missing_ok=True)\n'
        '            raise',
    ),
}


def main() -> None:
    if MUTANTS_ROOT.exists():
        shutil.rmtree(MUTANTS_ROOT)
    created = []
    for name, (rel_file, target, replacement) in MUTATIONS.items():
        dst_src = MUTANTS_ROOT / name / "src"
        shutil.copytree(REPO_SRC, dst_src)
        fpath = dst_src / rel_file
        txt = fpath.read_text()
        assert target in txt, f"{name}: target snippet not found in {rel_file}"
        mutated = txt.replace(target, replacement, 1)
        assert mutated != txt
        fpath.write_text(mutated)
        created.append(name)
    print(json.dumps({"created_mutants": created}, indent=2))


if __name__ == "__main__":
    main()
