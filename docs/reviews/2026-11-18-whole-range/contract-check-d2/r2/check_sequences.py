"""Run the sequence model and verify the starting guard's public gate behavior."""

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
from unittest.mock import patch

import probe_sequences


def check_starting_guard(directory):
    source = probe_sequences.branch_object(
        "fix/wr-deploy", "scripts/install-deploy-guard-hetzner.sh",
    )
    tool = source.split(b"<<'BTS_DEPLOY_HOLD_PYTHON'\n", 1)[1].split(
        b"\nBTS_DEPLOY_HOLD_PYTHON\n", 1,
    )[0]
    namespace = {"__name__": "fixture"}
    exec(compile(tool, "<starting guard>", "exec"), namespace)
    namespace.update(ROOT=directory, UID=os.getuid())
    writer_lock = Path(directory) / "writers.lock"
    writer_lock.touch()
    writer_lock.chmod(0o644)
    with namespace["directory"]() as root:
        hold = {
            "schema": "bts_deploy_hold_v1", "deploy_id": "1-provision",
            "owner_unit": "bts-deploy-1-provision.service", "state": "failed",
            "step": "abandoned", "target": "a" * 40, "previous": "unknown",
            "armed_at": 1, "updated_at": 1,
        }
        namespace["publish_json"](root, "hold", hold)
        assert namespace["read_hold"](root) == hold
        variants = (
            dict(hold, step="install-jobs"),
            dict(hold, step="provisioned"),
            dict(hold, step="provisioned", deploy_id=None, owner_unit=None),
        )
        with patch.object(sys, "argv", ["guard", "check-clear", "--scheduler"]):
            for variant in variants:
                namespace["publish_json"](root, "hold", variant)
                assert namespace["main"]() == 1
                try:
                    namespace["read_hold"](root)
                except ValueError:
                    continue
                raise AssertionError("starting reader accepted the new hold shape")

        namespace["publish_json"](root, "hold", dict(hold, step="activation", state="activating"))
        namespace["owner_alive"] = lambda root, record: True
        command = ["/bin/true"]
        with patch.object(os, "execv", side_effect=FileNotFoundError) as execute:
            with redirect_stdout(StringIO()):
                namespace["writer"]("reconcile", command)
            execute.assert_called_once_with(command[0], command)
    print("Starting guard rejected 3 new hold shapes and admitted a writer during live activation.")


if __name__ == "__main__":
    probe_sequences.starting_reader = check_starting_guard
    probe_sequences.main()
