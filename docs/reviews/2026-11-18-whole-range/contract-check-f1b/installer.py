"""Generate the proposed line with the existing fixtures in a tracked-file-free copy."""
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = HERE / "installer-copy"
SOURCE = Path("scripts/cron-setup-hetzner.sh")
OLD = "*/15 10-23 * * * $PREFIX $UV_BIN run bts check-pick-entered"
NEW = OLD.replace("$UV_BIN", "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout -k 30 840 $UV_BIN")
CONFIG = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'


def main():
    for name in (SOURCE, "tests/__init__.py", "tests/test_entry_intent.py", "tests/test_audit_f6_review_probes.py",
                 "tests/test_contest_fetch.py", "pyproject.toml"):
        target = SCRATCH / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(REPO / name, target)
    sys.path.insert(0, str(SCRATCH))
    from tests.test_entry_intent import box
    import pytest

    box_root = HERE / "installer-show"
    box_root.mkdir()
    fixture = box.__wrapped__(box_root)
    fixture.config.write_text(CONFIG)
    baseline = fixture.run("show")
    assert baseline.returncode == 0, baseline.stderr
    script = SCRATCH / SOURCE
    body = script.read_text()
    assert body.count(OLD) == 1
    script.write_text(body.replace(OLD, NEW))
    proposed = fixture.run("show")
    assert proposed.returncode == 0, proposed.stderr
    expected = baseline.stdout
    old_line, = [ln for ln in expected.splitlines() if ln.startswith("*/15 10-23")]
    new_line, = [ln for ln in proposed.stdout.splitlines() if ln.startswith("*/15 10-23")]
    assert expected.replace(old_line, new_line) == proposed.stdout
    (HERE / "generated-cron.txt").write_text(new_line + "\n")
    print(new_line, flush=True)

    foreign = "0 9 * * * echo foreign"
    untagged = old_line.rsplit(" # BTS-HETZNER", 1)[0]
    fixture.cron.write_text("\n".join((foreign, old_line, "# " + old_line, untagged, "# " + untagged)) + "\n")
    for _ in range(2):
        installed = fixture.run("install")
        assert installed.returncode == 0, installed.stderr
        lines = fixture.lines()
        assert lines[:3] == [foreign, untagged, "# " + untagged]
        assert lines.count(old_line) == 0 and lines.count(new_line) == 1
    assert fixture.run("remove").returncode == 0
    assert fixture.lines() == [foreign, untagged, "# " + untagged]
    print("Install twice, then remove: active and commented marked lines replaced; both unmarked lines preserved.", flush=True)
    os.chdir(SCRATCH)
    return pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short", "tests/test_entry_intent.py",
                        "tests/test_audit_f6_review_probes.py"])


if __name__ == "__main__":
    raise SystemExit(main())
