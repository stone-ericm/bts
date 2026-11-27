"""Check the committed production diff, checklist edit, and F2 hash pin."""

import subprocess


BASE = "a5fa7466"
ROUND_ONE = "34981a1a"
ROUND_TWO = "bf3d4f02"
CHECKLIST = "docs/ops/2027-season-start.md"
F2 = "tests/test_wr_f2_saver_flag.py"
OLD = b"**Enter it in the morning, before the scheduler's first lineup check.**"
NEW = (
    b"**Enter it in the morning, before the scheduler's first lineup check, when possible.** "
    b"If that check refused before choosing a pick, confirm this is still today's slate in ET "
    b"and that MLB has settled every prior entry or confirmed its absence after its window closed. "
    b"File the verified dated observation between checks while another lineup check or fallback remains."
)


def git(*args):
    return subprocess.check_output(["git", *args], timeout=5)


def main():
    assert git("diff", BASE, ROUND_TWO, "--", "src") == git("diff", BASE, ROUND_ONE, "--", "src")
    print("Base-to-round-two and base-to-round-one production diffs are byte-identical")
    before, after = (git("show", f"{rev}:{CHECKLIST}") for rev in (BASE, ROUND_TWO))
    assert before.count(OLD) == after.count(NEW) == 1
    assert after == before.replace(OLD, NEW, 1)
    print("The whole checklist differs from the base by exactly the authorized B8a replacement")
    before, after = (git("show", f"{rev}:{F2}") for rev in (BASE, ROUND_TWO))
    hunk = after[after.index(b"    original = original.replace(\n"):after.index(
        b"    assert hashlib.sha256(original).hexdigest() == (")]
    assert hunk.startswith(b"    original = original.replace(\n")
    assert after.replace(hunk, b"", 1) == before
    pin = b"21b6dd612d9019555a4436b0b3f68b5493d5751d466bda8a32dfabf5a156739d"
    assert before.count(pin) == after.count(pin) == 1
    assert git("diff", "--check", BASE, ROUND_TWO) == b""
    print("F2 gains one replacement statement and keeps its original expected SHA-256")


if __name__ == "__main__":
    main()
