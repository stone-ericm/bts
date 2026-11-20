"""Round-one review regressions on local guard, systemd and heartbeat shims."""

import fcntl
import hashlib
import json
import os
from datetime import datetime
from types import SimpleNamespace

import pytest

from tests.test_deploy_hold_tool import (
    INSTALLER, PREVIOUS, RUN_ID, TARGET, configure_transaction, hold_record,
    owner,
)


GUARD = "/usr/local/bin/bts-deploy-hold"
CONDITION = f"/usr/bin/python3 -I {GUARD} check-clear --scheduler"
HEARTBEAT = (
    "*/5 * * * * cd /home/bts/projects/bts && /usr/bin/python3 scripts/check_heartbeat.py "
    "--heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold # BTS-HETZNER"
)


def inventory():
    return [
        f"0 1 * * * {GUARD} run {job} -- /bin/sh -c "
        f"'uv run bts {command} --fixture' >> /home/bts/logs/cron.log 2>&1 # BTS-HETZNER"
        for job, command in [
            ("check-results", "check-results"), ("reconcile", "reconcile"), ("reconcile", "reconcile"),
            *[("fetch-contest-streak", "fetch-contest-streak")] * 4, ("nightly-preview", "preview"),
        ]
    ] + [HEARTBEAT]


def structured(command, *, path=None, ignore="no"):
    path = path or command.split()[0]
    return f"{{ path={path} ; argv[]={command} ; ignore_errors={ignore} ; start_time=[n/a] ; }}"


def preflight_shim(owner, cron=None, **overrides):
    source = INSTALLER.read_bytes()
    (owner.ROOT / "guard-install.sha256").write_text(hashlib.sha256(source).hexdigest() + "\n")
    cron = inventory() if cron is None else cron
    properties = {
        "LoadState": "loaded", "NeedDaemonReload": "no", "ExecCondition": structured(CONDITION),
        "SuccessExitStatus": "", "WorkingDirectory": owner.PROJECT,
        "Environment": "UV_CACHE_DIR=/tmp/uv-cache PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "EnvironmentFiles": f"{owner.PROJECT}/.env (ignore_errors=no)",
    }
    calls = []

    def call(argv, *, limit=10):
        calls.append(argv)
        if "/usr/bin/crontab" in argv:
            return 0, ("\n".join(cron) + "\n").encode()
        if "/usr/bin/systemctl" in argv:
            if "--user" not in argv:
                return 0, b"not-found\n"
            service = argv[argv.index("show") + 1]
            start = (f"{owner.UV} run --no-sync bts schedule --config {owner.CONFIG}"
                     if service == "bts-scheduler.service" else f"{owner.UV} run --no-sync python3 -m bts.web")
            loaded = {**properties, "ExecStart": structured(start),
                      **(overrides if service == "bts-scheduler.service" else {})}
            names = [value for i, value in enumerate(argv) if i and argv[i - 1] == "-p"]
            lines = [loaded[name] if "--value" in argv else f"{name}={loaded[name]}" for name in names]
            return 0, ("\n".join(lines) + "\n").encode()
        if "rev-parse" in argv:
            return 0, PREVIOUS.encode() + b"\n"
        if "cat-file" in argv:
            return 0, b"commit\n"
        if "show" in argv:
            return 0, source
        assert "config" in argv or "fetch" in argv, argv
        return 0, b""

    owner.call = call
    return calls


@pytest.mark.parametrize("variant", [
    "quoted", "module", "wrapped", "trailing", "commented-monitor", "wrong-monitor-path", "monitor-echo",
])
def test_f1_rejects_bypass_inventory(owner, variant):
    lines = inventory()
    extra = {
        "quoted": "0 4 * * * uv run 'bts' 'reconcile'",
        "module": "0 4 * * * .venv/bin/python -m bts.cli reconcile",
        "wrapped": "0 4 * * * /bin/sh -c 'uv run bts reconcile'",
    }
    if variant in extra:
        lines.append(extra[variant])
    elif variant == "trailing":
        lines[1] = lines[1].replace(" # BTS", "; uv run bts reconcile # BTS")
    elif variant == "commented-monitor":
        lines[-1] = "# " + lines[-1]
    elif variant == "monitor-echo":
        lines[-1] = '*/5 * * * * /bin/echo "&& /usr/bin/python3 scripts/check_heartbeat.py --deploy-hold /var/lib/bts-deploy/hold"'
    else:
        lines[-1] = lines[-1].replace("/var/lib/bts-deploy/hold", "/tmp/hold")
    preflight_shim(owner, lines)
    with pytest.raises(owner.Refused, match="crontab"):
        owner.preflight(1000, TARGET, fetch=False)


def test_f1_accepts_canonical_inventory_and_ignores_commented_writer(owner):
    preflight_shim(owner, inventory() + [" # uv run bts reconcile"])
    assert owner.preflight(1000, TARGET, fetch=False)[0] == PREVIOUS


@pytest.mark.parametrize("property,value", [
    ("ExecCondition", structured(CONDITION, path="/usr/bin/true")),
    ("ExecStart", structured("/home/bts/.local/bin/uv run --no-sync bts schedule --config /home/bts/.bts-orchestrator.toml",
                             path="/usr/bin/true")),
    ("SuccessExitStatus", "1"),
    ("SuccessExitStatus", "143 1"),
    ("WorkingDirectory", "/srv/alternate"),
    ("Environment", "UV_CACHE_DIR=/tmp/uv-cache PATH=/usr/bin"),
    ("EnvironmentFiles", "/home/bts/projects/bts/.env (ignore_errors=yes)"),
])
def test_f3_refuses_loaded_runtime_drift(owner, property, value):
    preflight_shim(owner, **{property: value})
    with pytest.raises(owner.Refused, match="guard"):
        owner.preflight(1000, TARGET, fetch=False)


def test_f3_accepts_scheduler_sigterm_exit_status(owner):
    preflight_shim(owner, SuccessExitStatus="143")
    assert owner.preflight(1000, TARGET, fetch=False)[0] == PREVIOUS


@pytest.mark.parametrize("state,pid,holder,alive", [
    ("active", 456, 456, True), ("active", 456, 789, False),
    ("deactivating", 456, 456, False), ("active", 0, 456, False),
])
def test_f2_liveness_binds_unit_pid_to_the_exact_kernel_lock(owner, monkeypatch, state, pid, holder, alive):
    record = hold_record("activating")
    contents = owner.ROOT / "proc-locks"
    info = (owner.ROOT / "transaction.lock").stat()
    contents.write_text(f"8: FLOCK ADVISORY WRITE {holder} {os.major(info.st_dev):02x}:"
                        f"{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n")
    owner.PROC_LOCKS = contents

    def query(argv, **kwargs):
        output = (f"ActiveState={state}\nMainPID={pid}\n".encode() if "MainPID" in argv else (state + "\n").encode())
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(owner.subprocess, "run", query)
    with owner.directory() as root, owner.lock(root, "transaction.lock", fcntl.LOCK_EX):
        assert owner.owner_alive(root, record) is alive
        owner.publish_json(root, "hold", hold_record("pending"))
        assert owner.clear(root) is False
        monkeypatch.setattr(owner.os, "execv", lambda *args: pytest.fail("pending writer escaped"))
        for job in owner.JOBS:
            assert owner.writer(job, ["/usr/bin/true"]) == 75


def test_f2_abandoned_work_is_reconciled_before_target_verification(owner):
    events = []
    configure_transaction(owner, events.append)

    def stale(*args, **kwargs):
        pytest.fail("target inspected before containing abandoned activation")

    owner.verified_target = stale
    with owner.directory() as root:
        owner.publish_json(root, "hold", hold_record("activating"))
        assert owner.owned_deploy(root, "1795197581-followup", TARGET, "") == 1
        assert owner.read_hold(root)["step"] == "abandoned"
    assert events == ["stop-old", "page"]


@pytest.mark.parametrize("failure", ["transaction", "uid", "stop"])
def test_f2_hook_failure_still_pages_and_reports(owner, failure):
    events = []
    configure_transaction(owner, events.append)
    original = owner.lock

    def contended(root, name, operation, timeout=0):
        if name == "transaction.lock" and failure == "transaction":
            assert timeout == 60
            raise BlockingIOError("fixture timeout")
        return original(root, name, operation, timeout)

    def bad_uid():
        raise owner.Refused("id")

    owner.lock = contended
    if failure == "uid":
        owner.bts_uid = bad_uid
    if failure == "stop":
        def bad_stop(uid):
            events.append("stop-old")
            raise OSError("fixture stop failure")
        owner.verified_stop = bad_stop
    with owner.directory() as root:
        owner.publish_json(root, "hold", hold_record("activating"))
        assert owner.reconcile(root, RUN_ID) == 1
        assert owner.read_hold(root)["state"] == ("activating" if failure == "transaction" else "failed")
        assert owner.read_json(root, owner.run_name(RUN_ID, "result"))["outcome"] == "failed"
    assert events == (["page"] if failure == "uid" else ["stop-old", "page"])


def test_f4_stops_old_dashboard_under_writer_exclusion_before_reset(owner):
    events = []
    configure_transaction(owner, events.append)
    original_stop = owner.verified_stop

    def stop_dashboard(uid):
        assert "reset" not in events
        with owner.directory() as root:
            assert owner.read_hold(root)["state"] == "pending"
            with pytest.raises(BlockingIOError), owner.lock(root, "writers.lock", fcntl.LOCK_SH):
                pass
        events.append("dashboard-stop")
        return True

    owner.verified_stop_dashboard = stop_dashboard
    owner.verified_stop = original_stop
    with owner.directory() as root:
        assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 0
    assert events.index("dashboard-stop") < events.index("reset")


@pytest.mark.parametrize("rollover,refusal", [(True, False), (False, True)])
def test_f6_rechecks_canary_after_activation_before_unlink(owner, capsys, rollover, refusal):
    events = []
    configure_transaction(owner, events.append)
    day = "2027-07-15"

    def canary():
        if "activation" in events:
            events.append("release-canary")
            if refusal:
                raise owner.Refused("canary-contest-state")
            if rollover:
                return "2027-07-16"
        return day

    owner.canary = canary
    with owner.directory() as root:
        assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 1
        assert owner.read_hold(root)["state"] == "failed"
        assert owner.read_json(root, owner.run_name(RUN_ID, "result"))["step"] == "activation"
    assert events.count("page") == 1
    assert events.index("activation") < events.index("release-canary") < events.index("page")
    assert "Activation failed after certification; the hold was kept." in capsys.readouterr().out


def test_f8_deploy_marker_precedes_first_verified_heartbeat(owner):
    configure_transaction(owner)
    owner.time.time = lambda: 1_800_000_000.5

    def activate(uid, authorized_at):
        assert authorized_at == 1_800_000_000.5
        owner.time.time = lambda: 1_800_000_031

    owner.activate_scheduler = activate
    with owner.directory() as root:
        assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 0
        assert owner.read_json(root, "last-release")["released_at"] == 1_800_000_000.5


def test_f6_midnight_during_release_record_write_preserves_hold(owner):
    events = []
    configure_transaction(owner, events.append)
    owner.canary = lambda: "2027-07-15"
    before = datetime.fromisoformat("2027-07-15T23:59:59.900-04:00").timestamp()
    owner.time.time = lambda: before
    original = owner.publish_json

    def publish(root, name, value):
        original(root, name, value)
        if name == "last-release":
            owner.time.time = lambda: before + 1

    owner.publish_json = publish
    with owner.directory() as root:
        assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 1
        assert owner.read_hold(root)["step"] == "activation"
        assert owner.read_json(root, owner.run_name(RUN_ID, "result"))["step"] == "activation"
    assert events.count("page") == 1


def test_f9_hand_release_uses_stopped_box_window(owner):
    configure_transaction(owner)
    owner.preflight = lambda *args, **kwargs: (TARGET, ["fixture-git"])
    now = owner.time.time()
    owner.time.time = lambda: now
    def window(recovery, target, deadline, *, minimum=1800):
        assert (recovery, target, deadline, minimum) == (TARGET, TARGET, now + 30, 30)
    owner.window = window
    with owner.directory() as root:
        owner.publish_json(root, "hold", hold_record())
        assert owner.release_by_hand(root, "held scheduler verified stopped") == 0
        assert owner.read_hold(root) is None


@pytest.mark.parametrize("result,final", [(204, "sent 204"), (None, "failed")])
def test_f11_retries_claim_before_marking_actual_post_outcome(owner, result, final):
    record = hold_record()
    def page(current, stopped):
        with owner.directory() as root:
            assert owner.read_json(root, owner.run_name(RUN_ID, "evidence"))["page"] == "claimed"
        return result
    owner.page = page
    with owner.directory() as root:
        name = owner.run_name(RUN_ID, "evidence")
        owner.publish_json(root, name, {"stop": "verified", "page": "claimed"})
        assert owner.containment(root, record, 1000)
        assert owner.read_json(root, name)["page"] == final


def test_f11_does_not_retry_a_second_lost_claim(owner):
    record = hold_record()
    def lost(*args):
        raise InterruptedError("fixture interruption after claiming the POST")
    owner.page = lost
    with owner.directory() as root:
        name = owner.run_name(RUN_ID, "evidence")
        owner.publish_json(root, name, {"stop": "verified", "page": "not-attempted"})
        for count in (1, 2):
            with pytest.raises(InterruptedError):
                owner.containment(root, record, 1000)
            assert owner.read_json(root, name)["page_attempts"] == count
        assert owner.containment(root, record, 1000)
        assert owner.read_json(root, name)["page"] == "claimed"


def test_f14_previous_sha_and_success_canary_line_survive(owner, capsys):
    preflight_shim(owner, WorkingDirectory="/old")
    owner.bts_uid = lambda: 1000
    with owner.directory() as root:
        assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 1
    assert "Previous SHA: aaaaaaa." in capsys.readouterr().out


def test_f14_preserves_the_canary_line(owner, capsys):
    line = b"deploy canary 2027-07-15T23:59:59-04:00: research; contest state skipped\n"
    owner.call = lambda *args, **kwargs: (0, line)
    assert owner.canary() == "2027-07-15"
    assert capsys.readouterr().out == line.decode()


def test_f7_operator_recovery_names_the_verified_tree_and_guarded_revision():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    incident = (root / "INCIDENT.md").read_text()
    assert "uv run --no-sync bts season-contract status --picks-dir" in incident
    assert "uv run --no-sync bts set-contest-streak --picks-dir" in incident
    assert "same guard protocol" in incident
    for name in ("CLAUDE.md", "docs/ops/2027-season-start.md"):
        document = (root / name).read_text()
        assert "git show" in document
        assert "systemctl --user stop bts-dashboard" in document


def test_f16_checklist_requires_read_only_canary_on_real_config():
    from pathlib import Path
    checklist = (Path(__file__).resolve().parents[1] / "docs/ops/2027-season-start.md").read_text()
    a7 = next(line for line in checklist.splitlines() if line.startswith("| A7 |"))
    assert "uv run --no-sync bts deploy-canary --config ~/.bts-orchestrator.toml" in a7
    assert "on the box as `bts`" in a7 and "Confirm it passes on the real configuration" in a7
