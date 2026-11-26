"""Run repository scripts on a fake box or offline local Git remote."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fixtures import HERE, ROOT, fixture

ops = fixture("probe_ops")


def install_shim(root, offline):
    # Simulate Linux install unlinking a destination symlink. The systemctl shim
    # observes the mask at the invocation inside the real shell script.
    unit = root / ".config/systemd/user/bts-lineup-collect.service"
    unit.parent.mkdir(parents=True)
    unit.symlink_to("/dev/null")
    binaries = root / "bin"
    binaries.mkdir()
    install = binaries / "install"
    install.write_text(
        f"#!{sys.executable}\n"
        "import os, shutil, sys\n"
        "src, dest = sys.argv[-2:]\n"
        "if os.path.lexists(dest): os.unlink(dest)\n"
        "shutil.copyfile(src, dest)\n"
    )
    systemctl = binaries / "systemctl"
    systemctl.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        f"mask = Path({str(unit)!r})\n"
        "print('  internal daemon-reload, writer masked:', mask.is_symlink())\n"
        f"sys.exit({int(offline)})\n"
    )
    for tool in (install, systemctl):
        tool.chmod(0o700)
    env = dict(os.environ, HOME=str(root), PATH=f"{binaries}:/usr/bin:/bin")
    result = subprocess.run(
        ["/bin/bash", "scripts/install-systemd-hetzner.sh"],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=8,
    )
    print(result.stdout, result.stderr, result.returncode)
    assert result.returncode == int(offline)
    assert not unit.is_symlink()
    assert "internal daemon-reload, writer masked: False" in result.stdout


def pause(root):
    unit = root / ".config/systemd/user/bts-scheduler.service"
    enabled = unit.parent / "default.target.wants" / unit.name
    enabled.parent.mkdir(parents=True)
    enabled.symlink_to(unit)
    shutil.copyfile(ROOT / "scripts/systemd" / unit.name, unit)
    assert enabled.resolve().is_file()
    unit.unlink()
    unit.symlink_to("/dev/null")
    assert enabled.resolve() == Path("/dev/null")
    print("After stop only: enablement points to real unit. After step 4's mask: enablement points to /dev/null.")


if __name__ == "__main__":
    action = sys.argv[1]
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        root = Path(temporary)
        if action == "refs":
            ops.reviewed_ref(root)
            pause(root)
            ops.restore_masks(root)
        elif action == "install-offline":
            install_shim(root, True)
        elif action == "install-online":
            install_shim(root, False)
        elif action == "october":
            ops.october_shortcut(root)
        else:
            ops.deploy_or_cron(root, action)
