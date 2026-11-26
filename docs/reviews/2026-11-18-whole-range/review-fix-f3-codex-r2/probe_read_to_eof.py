"""Run the byte-count and result checks on a control or a read-to-EOF scratch copy."""
from pathlib import Path
import sys


repo = Path(__file__).resolve().parents[2]
scratch = Path(sys.argv[1]).resolve()
assert not scratch.is_relative_to(repo)
target = scratch / "src/bts/day_outcome.py"
source = (repo / "src/bts/day_outcome.py").read_text()
assert target.read_text() == source
mutant = sys.argv[2] == "mutant"
before = '''            while len(raw) <= EVIDENCE_MAX_BYTES:
                chunk = os.read(fd, EVIDENCE_MAX_BYTES + 1 - len(raw))
                if not chunk:
                    return json.loads(raw.decode("utf-8"))
                raw.extend(chunk)
'''
after = '''            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                raw.extend(chunk)
            if len(raw) <= EVIDENCE_MAX_BYTES:
                return json.loads(raw.decode("utf-8"))
'''
assert source.count(before) == 1
try:
    if mutant:
        target.write_text(source.replace(before, after, 1))
    import bts
    import pytest
    from bts import day_outcome

    print(f"bts.__file__={bts.__file__}", flush=True)
    assert Path(day_outcome.__file__).resolve() == target
    raise SystemExit(pytest.main([
        "-p", "no:cacheprovider", "-q", "tests/test_wr_f3_skip_state_review.py",
        "-k", "evidence_read_byte_limit or (K5_bad_optional_evidence and oversize)",
    ]))
finally:
    if mutant:
        target.write_text(source)
