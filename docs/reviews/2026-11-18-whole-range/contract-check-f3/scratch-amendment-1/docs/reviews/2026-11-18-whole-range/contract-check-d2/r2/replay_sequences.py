"""Replay the unchanged sequence model with a valid starting-point hold control."""

import os

import probe_sequences


def check_starting_reader(directory):
    source = probe_sequences.branch_object(
        "fix/wr-deploy", "scripts/install-deploy-guard-hetzner.sh",
    )
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
        for variant in variants:
            namespace["publish_json"](root, "hold", variant)
            try:
                namespace["read_hold"](root)
            except ValueError:
                assert not namespace["clear"](root)
            else:
                raise AssertionError("starting parser accepted the new hold shape")
    print("Starting reader accepted the abandoned control and rejected all 3 changed records.")


if __name__ == "__main__":
    probe_sequences.starting_reader = check_starting_reader
    probe_sequences.main()
