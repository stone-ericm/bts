"""Small local model of the V2 sequence and the exact V1 hold reader."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


SUBSTEPS = (
    "invalidate", "intent", "render", "backup", "cron-replace", "cron-readback",
    "scheduler-rename", "dashboard-rename", "reload", "verify", "record-rename",
    "record-fsync",
)


@dataclass(frozen=True)
class Box:
    head: str = "old"
    runtime: str = "old"
    cron: str = "old"
    fragments: tuple = ("old", "old")
    loaded: tuple = ("old", "old")
    installed: str | None = "old"
    hold: str | None = None

    def run_jobs(self):
        # Both D2 crontabs gate the eight writer invocations. Analytics,
        # monitor and backups still run with the checkout's runtime.
        return {
            "writer": None if self.hold or not self.cron else (self.cron, self.head),
            "unheld": (self.cron, self.head, self.runtime) if self.cron else None,
        }


def advance(box, step):
    if step == "invalidate":
        return replace(box, installed=None)
    if step == "cron-replace":
        return replace(box, cron="new")
    if step == "scheduler-rename":
        return replace(box, fragments=("new", box.fragments[1]))
    if step == "dashboard-rename":
        return replace(box, fragments=(box.fragments[0], "new"))
    if step == "reload":
        return replace(box, loaded=box.fragments)
    if step in {"verify", "record-rename", "record-fsync"}:
        assert box.cron == "new" and box.loaded == ("new", "new")
        return replace(box, installed="new") if step != "verify" else box
    return box


def ordinary_sequence():
    old = Box()
    assert old.run_jobs()["writer"] == ("old", "old")  # queue/failure before arm
    armed = replace(old, hold="pending")
    resetting = replace(armed, head="new")
    assert resetting.run_jobs() == {"writer": None, "unheld": ("old", "new", "old")}
    synced = replace(resetting, runtime="new")
    cuts = []
    for step in SUBSTEPS:
        synced = advance(synced, step)
        failed = replace(synced, hold="failed")
        # A process failure makes failed durable. Abrupt reboot sees pending
        # until a new owner reconciles it to failed. Neither admits gated work.
        assert synced.run_jobs()["writer"] is None
        assert failed.run_jobs()["writer"] is None
        cuts.append((step, synced))
    assert replace(synced, hold=None).run_jobs()["writer"] == ("new", "new")
    assert next(box for step, box in cuts if step == "cron-replace").loaded == ("old", "old")
    absent = next(box for step, box in cuts if step == "cron-replace")
    assert absent.installed is None and absent.cron == "new"
    # Root install when installed.json is absent requires no table. This
    # blocks a fingerprint refresh if cron was already replaced.
    assert absent.cron is not None
    return cuts


def v2_admits_without_job_proof(installed, hold_state):
    return installed is None and hold_state == "failed"


def recovery_and_boot():
    assert not v2_admits_without_job_proof(None, None)
    assert not v2_admits_without_job_proof(None, "pending")
    # Same preflight result for provisioning, an interrupted switch, and
    # an unrelated failed run whose installation record went missing.
    for scenario in ("provisioned", "switch-interrupted", "canary-failed"):
        assert scenario and v2_admits_without_job_proof(None, "failed")
    # An old cron invocation and an independently enabled timer do not
    # take the new gate. K3 guards only the scheduler's new starts.
    covered = {"new-gated-cron", "scheduler-start", "new-saver"}
    boot_jobs = {"october-cron", "scheduler-start", "october-dashboard", "persistent-resolver"}
    assert boot_jobs - covered == {"october-cron", "october-dashboard", "persistent-resolver"}


@contextmanager
def exclusive(path):
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def transaction_lock(directory, cuts):
    lock = directory / "transaction.lock"
    lock.touch()
    with exclusive(lock):
        for step, _ in cuts:
            try:
                with exclusive(lock):
                    raise AssertionError(f"hook took T during {step}")
            except OSError as error:
                assert error.errno in (errno.EAGAIN, errno.EACCES)
    with exclusive(lock):
        pass


def fingerprint(environment):
    return hashlib.sha256(environment).digest()


def manager_examples():
    original = b"PATH=/usr/bin:/bin\n"
    imported = original + b"SSH_AUTH_SOCK=/run/user/1000/new-agent\n"
    changed_runtime = original + b"UV_PROJECT_ENVIRONMENT=/home/bts/alternate\n"
    for value in (imported, changed_runtime):
        assert fingerprint(original) != fingerprint(value)
        assert fingerprint(imported) != fingerprint(changed_runtime)
    assert fingerprint(original + b"FOO=1\n") != fingerprint(b"FOO=1\n" + original)
    # Applying an environment change then deploying cannot reach install-jobs
    # when preflight requires the old hash.
    assert fingerprint(original) != fingerprint(changed_runtime)


def branch_object(revision, path):
    return subprocess.run(
        ["/usr/bin/git", "show", f"{revision}:{path}"],
        stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=2,
    ).stdout


def starting_reader(directory):
    source = branch_object("fix/wr-deploy", "scripts/install-deploy-guard-hetzner.sh")
    tool = source.split(b"<<'BTS_DEPLOY_HOLD_PYTHON'\n", 1)[1].split(
        b"\nBTS_DEPLOY_HOLD_PYTHON\n", 1,
    )[0]
    namespace = {"__name__": "fixture"}
    exec(compile(tool, "<starting guard>", "exec"), namespace)
    namespace.update(ROOT=directory, UID=os.getuid())
    with namespace["directory"]() as root:
        hold = {
            "schema": "bts_deploy_hold_v1", "deploy_id": "1-provision",
            "owner_unit": "bts-deploy-1-provision.service", "state": "failed",
            "step": "guard", "target": "a" * 40, "previous": "unknown",
            "armed_at": 1, "updated_at": 1,
        }
        namespace["publish_json"](root, "hold", hold)
        assert namespace["read_hold"](root) == hold
        for variant in (dict(hold, step="provisioned"),
                        dict(hold, step="provisioned", deploy_id=None, owner_unit=None)):
            namespace["publish_json"](root, "hold", variant)
            try:
                namespace["read_hold"](root)
                raise AssertionError("starting parser accepted new provisioning shape")
            except ValueError:
                pass
    return namespace


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise SystemExit("run this probe at nice 19")
    pgrep = subprocess.run(
        ["/usr/bin/pgrep", "-x", "victoria3"], stdin=subprocess.DEVNULL,
        capture_output=True, timeout=2,
    )
    assert pgrep.returncode in (0, 1), pgrep.stderr
    cuts = ordinary_sequence()
    recovery_and_boot()
    manager_examples()
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
        directory = Path(temp)
        directory.chmod(0o755)
        transaction_lock(directory, cuts)
        starting_reader(directory)
    resolver = branch_object("619abda7", "scripts/systemd/bts-live-forward-resolve.timer")
    assert b"Persistent=true\n" in resolver
    print(f"victoria3 present: {pgrep.returncode == 0}; nice 19")
    print(f"{len(cuts)} install cuts with failure/reboot controls and T overlap; "
          "queued deploy, unheld-job, missing-proof, interrupted fingerprint refresh, "
          "old boot and manager-hash sequences; starting hold reader rejects 2 provisioning shapes")


if __name__ == "__main__":
    main()
