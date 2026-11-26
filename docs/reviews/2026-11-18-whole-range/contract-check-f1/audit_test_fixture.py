"""Run audit-driver assertions with only machine-specific transport discovery supplied as a fixture."""
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

real_run = subprocess.run


def offline_run(args, *other, **kwargs):
    if isinstance(args, (list, tuple)) and args:
        command = Path(args[0]).name
        if command == "rsync" and list(args[1:]) == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "rsync  version 3.2.7  protocol version 31\n", "")
        if command == "security" and len(args) > 1 and args[1] == "find-generic-password":
            raise FileNotFoundError("synthetic headless host")
        if command in {"rsync", "ssh", "scp", "curl", "wget", "oci", "aws"}:
            raise AssertionError(f"external command attempted in offline audit test: {command}")
    return real_run(args, *other, **kwargs)


with patch.object(subprocess, "run", offline_run):
    raise SystemExit(pytest.main(["-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short",
                                  "tests/test_audit_driver.py"]))
