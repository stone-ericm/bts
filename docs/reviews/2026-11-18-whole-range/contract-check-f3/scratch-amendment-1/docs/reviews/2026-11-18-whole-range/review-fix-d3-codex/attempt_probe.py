"""Exercise recovery ordering and hostile filesystem entries with the original shims."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from shell_probe import DEPLOY, HERE, RECOVER, tests


def new_box(temp):
    box = tests.box.__wrapped__(Path(temp))
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    return box, runtime


def invoke(label, box, script=RECOVER, **kwargs):
    (box / "calls").unlink(missing_ok=True)
    result, calls = tests.run_box_script(box, script, "active", **kwargs)
    print(json.dumps(dict(case=label, status=result.returncode, output=result.stdout, errors=result.stderr,
                          commands=[call for call in calls if call[0] in ("systemctl", "git", "curl")]),
                     ensure_ascii=False), flush=True)
    return result, calls


def no_service(result, calls, message):
    assert result.returncode == 1 and message in result.stdout, result
    assert not any(call[0] in {"systemctl", "curl"} or call[:2] == ["git", "reset"] for call in calls), calls


def create(path, kind):
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "dangling-link":
        path.symlink_to(path.parent / "absent")
    elif kind == "existing-link":
        fixture = path.parent / "untouched"
        fixture.write_text("untouched")
        path.symlink_to(fixture)
    elif kind == "oversized":
        path.write_bytes(b"x" * 513)
    elif kind == "embedded-NUL":
        path.write_bytes(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -".encode() + b"\0\n")
    elif kind == "incompatible-fields":
        path.write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun reset verified verified verified\n")
    elif kind == "wrong-mode":
        path.touch(mode=0o644)
    else:
        raise ValueError(kind)


def special(kind_group):
    if kind_group == "marks":
        path, deploy_msg, recovery_msg = "cancelled.123.1", "cancelled failed", "cancellation mark could not be verified"
        kinds = ("directory", "fifo", "dangling-link", "existing-link", "oversized", "wrong-mode")
    else:
        path, deploy_msg, recovery_msg = "attempt", "attempt-record failed", "attempt record is unreadable"
        kinds = ("directory", "fifo", "dangling-link", "existing-link", "oversized", "embedded-NUL", "incompatible-fields")
    for kind in kinds:
        with TemporaryDirectory(prefix="wrd3-fs-", dir=HERE) as temp:
            box, runtime = new_box(temp)
            create(runtime / path, kind)
            for script, name, message in [(DEPLOY, "deploy", deploy_msg), (RECOVER, "recovery", recovery_msg)]:
                no_service(*invoke(f"{kind_group}-{kind}-{name}", box, script), message)
            if kind == "existing-link":
                assert (runtime / "untouched").read_text() == "untouched"


def order():
    with TemporaryDirectory(prefix="wrd3-order-", dir=HERE) as temp:
        box, runtime = new_box(temp)
        lock = runtime / "lock"
        lock.write_bytes(b"persistent-inode")
        lock.chmod(0o600)
        inode = lock.stat().st_ino
        with lock.open("ab") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            no_service(*invoke("contended-recovery", box), "lock unavailable")
            no_service(*invoke("contended-deploy", box, DEPLOY), "lock failed")
        old_result, old_calls = invoke("recovery-before-deploy", box)
        no_service(old_result, old_calls, "no phase-2 record on this boot")
        no_service(*invoke("late-deploy-refuses-mark", box, DEPLOY), "cancelled failed")
        assert lock.stat().st_ino == inode and lock.read_bytes() == b"persistent-inode"
        next_result, _ = invoke("actions-rerun-new-attempt", box, DEPLOY, attempt="2")
        assert next_result.returncode == 0
        no_service(*invoke("same-run-old-attempt-recovery", box), "a different attempt owns the box record")
        no_service(*invoke("different-run-recovery", box, run_id="124", attempt="2"), "a different attempt owns the box record")
        no_service(*invoke("committed-recovery", box, attempt="2"), "this attempt committed on the box")
        shutil.rmtree(runtime.parent)
        no_service(*invoke("post-reboot-recovery", box, attempt="2"), "no phase-2 record on this boot")


def refresh():
    with TemporaryDirectory(prefix="wrd3-refresh-", dir=HERE) as temp:
        box, runtime = new_box(temp)
        (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
        (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/a\n")
        first, calls = invoke("matching-begun-recovery", box)
        assert "DEPLOY FAILED at deploy-step-incomplete" in first.stdout
        assert sum(call[0] == "curl" for call in calls) == 1
        marker = box / "home/projects/bts/data/.heartbeat"
        before = json.loads(marker.read_text())
        marker.write_text('{"state":"running"}')
        second, calls = invoke("matching-contained-fresh-marker", box)
        after = json.loads(marker.read_text())
        assert before.keys() == after.keys() and before.pop("timestamp") != after.pop("timestamp")
        assert before == after and after["state"] == "deploy_failed"
        assert "DEPLOY RECOVERY — containment re-checked." in second.stdout
        assert not any(call[0] == "curl" for call in calls)


parser = argparse.ArgumentParser()
parser.add_argument("case", choices=("marks", "records", "order", "refresh"))
selected = parser.parse_args().case
if selected in ("marks", "records"):
    special(selected)
else:
    globals()[selected]()
