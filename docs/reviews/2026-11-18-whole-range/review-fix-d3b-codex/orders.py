"""Exercise recovery, re-runs, reboot and lost transport in shim boxes."""
from concurrent.futures import ThreadPoolExecutor
import os
from shutil import rmtree
import signal
import sys
import time

from review_box import OLD, RECOVER, PRIORS, box, contained, inject_shim, invoke, record, tests


def owner_orders():
    with box() as path:
        state, _ = invoke(path, "recovery wins before SSH", RECOVER)
        assert "no phase-2 record is present." in state["out"]
        assert (path / "run/bts-deploy/cancelled.123.1").exists()
        state, _ = invoke(path, "late same attempt fenced")
        assert "DEPLOY NOT STARTED — cancelled failed." in state["out"] and not state["writes"]
        state, _ = invoke(path, "next attempt commits", attempt="2")
        assert state["rc"] == 0 and state["attempt"] == record("committed", attempt="2")
        state, _ = invoke(path, "stale recovery after next attempt", RECOVER)
        assert "different attempt owns" in state["out"] and not state["writes"] and state["posts"] == 0
        assert state["attempt"] == record("committed", attempt="2")
        state, _ = invoke(path, "same attempt duplicate", attempt="2")
        assert "DEPLOY NOT STARTED — attempt-record failed." in state["out"] and not state["writes"]
        rmtree(path / "run")
        state, _ = invoke(path, "recovery sees no ownership after reboot", RECOVER, attempt="2")
        assert "no phase-2 record is present." in state["out"] and not state["writes"]
        state, _ = invoke(path, "older SSH after reboot lost its mark (L9)")
        assert state["rc"] == 0 and "CANARY PASSED" in state["out"]
    for order in (("123", "402"), ("402", "123")):
        with box(record(owner="402")) as path:
            state, _ = invoke(path, "interrupted unpublished begin with prior begun", fault="publish-begun:TERM:unpublished")
            assert "DEPLOY NOT STARTED" in state["out"] and state["attempt"] == record(owner="402")
            for owner in order:
                state, _ = invoke(path, f"ordered recovery {order} {owner}", RECOVER, run_id=owner)
                if owner == "402":
                    contained(state, "deploy-step-incomplete")
                else:
                    assert not state["writes"] and not state["posts"]
            state, _ = invoke(path, f"fresh run after {order}", run_id="501")
            assert state["rc"] == 0 and "CANARY PASSED" in state["out"]


def transport():
    for fault in ("", "reset:17", "stamp:TERM", "publish-committed:PIPE"):
        with box() as path:
            state, _ = invoke(path, f"output broken {fault=}", fault=fault, closed_output=True)
            assert not state["out"]
            if fault in ("", "publish-committed:PIPE"):
                assert state["rc"] == 0 and state["attempt"] == record("committed")
            else:
                assert state["rc"] == 1 and state["stopped"] and state["posts"] == 1
                assert state["link"] == "disabled" and "contained " in state["attempt"]
    with box() as path, ThreadPoolExecutor(max_workers=1) as executor:
        from review_box import DEPLOY
        background = executor.submit(tests.run_box_script, path, DEPLOY, "active", fault="reset:hold", limit=18)
        try:
            end = time.monotonic() + 8
            while time.monotonic() < end and not (path / "fault-fired").exists() and not background.done():
                time.sleep(.01)
            assert (path / "fault-fired").exists() and not background.done()
            owner = int((path / "owner.pid").read_text())
            os.kill(owner, signal.SIGKILL)
            state, _ = invoke(path, "recovery while lost-session mutator holds flock", RECOVER)
            assert "lock unavailable" in state["out"] and not state["writes"]
        finally:
            (path / "release").touch()
        assert background.result(timeout=15)[0].returncode == -signal.SIGKILL
        state, _ = invoke(path, "recovery after surviving mutator exits", RECOVER)
        contained(state, "deploy-step-incomplete")
        assert state["head"] != OLD


def lifecycle(signame):
    for prior in (*PRIORS, None):
        initial = record(prior, owner="802") if prior else None
        for fault in (f"fetch:{signame}", f"publish-committed:{signame}", f"log-pass:{signame}"):
            with box(initial) as path:
                state, _ = invoke(path, f"lifecycle {prior=} {fault=}", fault=fault)
                assert (path / "fault-fired").exists()
                if fault.startswith("fetch:") and signame != "INT":
                    assert state["rc"] != 0 and state["attempt"] == initial and not state["writes"]
                else:
                    # An owner-only SIGINT during successful foreground fetch is
                    # ignored by this untrapped Bash. Subsequent phase checks still run.
                    assert state["rc"] == 0 and state["attempt"] == record("committed")
                    assert state["link"] == "enabled" and not state["stopped"] and not state["posts"]
    shim = inject_shim(f'''
if command == "systemctl" and args[:2] == ["--user", "stop"]:
    (root / "interrupted-recovery").touch()
    os.kill(int((root / "owner.pid").read_text()), signal.SIG{signame})
''')
    with box(record(), shim=shim) as path:
        state, _ = invoke(path, f"signal during matching recovery stop {signame}", RECOVER)
        assert (path / "interrupted-recovery").exists()
        contained(state, "deploy-step-incomplete")
        assert state["out"].count("DEPLOY FAILED at deploy-step-incomplete") == 1


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
