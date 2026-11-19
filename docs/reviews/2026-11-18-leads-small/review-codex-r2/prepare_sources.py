"""Archive source snapshots and prepare isolated single-change mutants."""

from io import BytesIO
from pathlib import Path
from shutil import copytree
import subprocess
import tarfile


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
SOURCES = REVIEW / "sources"
MODULE = Path("src/bts/features/compute.py")


def archive(revision, name):
    destination = SOURCES / name
    destination.mkdir(parents=True)
    content = subprocess.check_output(["git", "archive", revision, "src"], cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(content)) as tar:
        tar.extractall(destination, filter="data")
    return destination


WRITE = '''        tmp = _probable_pitcher_cache_path.with_name(
            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))
            os.replace(tmp, _probable_pitcher_cache_path)
        finally:
            tmp.unlink(missing_ok=True)
'''
IN_PLACE = '''        _probable_pitcher_cache_path.write_text(
            json.dumps({str(k): v for k, v in lookup.items()})
        )
'''
WITHOUT_CLEANUP = '''        tmp = _probable_pitcher_cache_path.with_name(
            f".{_probable_pitcher_cache_path.name}.{os.getpid()}.tmp"
        )
        tmp.write_text(json.dumps({str(k): v for k, v in lookup.items()}))
        os.replace(tmp, _probable_pitcher_cache_path)
'''
WARNING = '''            print(
                f"WARNING: unreadable probable-pitcher cache {_probable_pitcher_cache_path} "
                f"({type(e).__name__}); rebuilding it from raw feeds",
                file=sys.stderr,
            )
'''
FINALLY = '''        finally:
            tmp.unlink(missing_ok=True)
'''
SWALLOW = '''        except OSError:
            pass
''' + FINALLY


def main():
    archive("df454725", "base")
    head = archive("fb1acd4c", "head")
    source = (head / MODULE).read_text()
    assert source == (ROOT / MODULE).read_text()
    for name, old, new in (
        ("write-in-place", WRITE, IN_PLACE),
        ("no-cleanup", WRITE, WITHOUT_CLEANUP),
        ("no-warning", WARNING, ""),
        ("swallow-write-error", FINALLY, SWALLOW),
    ):
        assert source.count(old) == 1, name
        mutant = SOURCES / name
        copytree(head, mutant)
        mutated = source.replace(old, new, 1)
        compile(mutated, str(mutant / MODULE), "exec")
        (mutant / MODULE).write_text(mutated)
        print(f"{name}: {mutant / MODULE}")


if __name__ == "__main__":
    main()
