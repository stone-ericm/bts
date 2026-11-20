"""Exercise provisioning in a temporary root and with a fake user manager."""

import fcntl
import hashlib
from pathlib import Path
import stat
import subprocess

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / "scripts/install-deploy-guard-hetzner.sh"
CONDITION = "/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold check-clear --scheduler"


@pytest.fixture
def installer(tmp_path):
    root = tmp_path / "image"
    root.mkdir()
    shims = tmp_path / "bin"
    shims.mkdir()
    commands = {
        "id": 'printf "%s\\n" 1000\n',
        "chown": 'printf "%s\\n" "$*" >> "$BTS_INSTALL_CALLS"\n',
        "runuser": (
            'printf "%s\\n" "$*" >> "$BTS_INSTALL_CALLS"\n'
            'case " $* " in\n'
            '  *" daemon-reload "*) exit 0;;\n'
            '  *" NeedDaemonReload "*) printf "%s\\n" "${BTS_RELOAD_PENDING:-no}";;\n'
            '  *" ExecStart "*)\n'
            '    uv=/home/bts/.local/bin/uv\n'
            '    case " $* " in\n'
            '      *" bts-scheduler.service "*)\n'
            '        args="$uv run --no-sync bts schedule --config /home/bts/.bts-orchestrator.toml"\n'
            f'        printf "%s\\n" "ExecCondition={{ path=${{BTS_CONDITION_PATH:-/usr/bin/python3}} ; argv[]={CONDITION} ; ignore_errors=no ; }}" "SuccessExitStatus=${{BTS_SUCCESS_STATUS:-}}";;\n'
            '      *) args="$uv run --no-sync python3 -m bts.web";;\n'
            '    esac\n'
            '    printf "%s\\n" "ExecStart={ path=$uv ; argv[]=$args ; ignore_errors=no ; }" \\\n'
            '      "WorkingDirectory=/home/bts/projects/bts" \\\n'
            '      "Environment=UV_CACHE_DIR=/tmp/uv-cache PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin" \\\n'
            '      "EnvironmentFiles=/home/bts/projects/bts/.env (ignore_errors=no)";;\n'
            '  *) exit 96;;\n'
            'esac\n'
        ),
    }
    for name, body in commands.items():
        shim = shims / name
        shim.write_text(f"#!/bin/sh\n{body}")
        shim.chmod(0o700)

    def run(**overrides):
        return subprocess.run(
            ["/bin/bash", str(INSTALLER)],
            env={
                "DESTDIR": str(root), "PATH": f"{shims}:/usr/bin:/bin",
                "BTS_INSTALL_CALLS": str(tmp_path / "calls"), **overrides,
            },
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
        )

    return root, run


def test_installer_keeps_lock_inodes_and_hold_on_reinstall(installer):
    root, run = installer
    result = run()
    assert result.returncode == 0, result.stderr
    directory = root / "var/lib/bts-deploy"
    locks = [directory / name for name in ("writers.lock", "transaction.lock", "state.lock")]
    assert len({lock.stat().st_ino for lock in locks}) == 3
    before = [(lock.stat().st_ino, stat.S_IMODE(lock.stat().st_mode)) for lock in locks]
    assert all(mode == 0o644 for _, mode in before)
    guard = root / "usr/local/bin/bts-deploy-hold"
    dropin = root / "etc/systemd/user/bts-scheduler.service.d/50-deploy-hold.conf"
    assert stat.S_IMODE(guard.stat().st_mode) == 0o755
    assert dropin.read_text() == f"[Service]\nExecCondition={CONDITION}\n"
    hold = directory / "hold"
    hold.write_bytes(b"keep")
    with locks[0].open("rb") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run()
        assert result.returncode == 0 and "already current" in result.stdout
    assert [(lock.stat().st_ino, stat.S_IMODE(lock.stat().st_mode)) for lock in locks] == before
    assert hold.read_bytes() == b"keep"


def test_installer_refuses_stale_manager(installer):
    _, run = installer
    result = run(BTS_RELOAD_PENDING="yes")
    assert result.returncode != 0
    assert "FAILED:" in result.stdout + result.stderr


def test_installer_records_exact_source_only_after_successful_verification(installer):
    root, run = installer
    result = run()
    assert result.returncode == 0, result.stderr
    record = root / "var/lib/bts-deploy/guard-install.sha256"
    assert record.read_bytes() == hashlib.sha256(INSTALLER.read_bytes()).hexdigest().encode() + b"\n"
    assert stat.S_IMODE(record.stat().st_mode) == 0o644
    result = run(BTS_RELOAD_PENDING="yes")
    assert result.returncode != 0
    assert not record.exists()


def test_f3_installer_uses_same_loaded_unit_validation(installer):
    root, run = installer
    accepted = run(BTS_SUCCESS_STATUS="143")
    assert accepted.returncode == 0, accepted.stderr
    for override in ({"BTS_CONDITION_PATH": "/usr/bin/true"}, {"BTS_SUCCESS_STATUS": "1"}):
        result = run(**override)
        assert result.returncode != 0
        assert "effective user unit differs" in result.stderr
        assert not (root / "var/lib/bts-deploy/guard-install.sha256").exists()
