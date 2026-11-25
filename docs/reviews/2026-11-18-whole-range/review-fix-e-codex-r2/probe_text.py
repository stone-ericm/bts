"""Verify the literal contract against real HEAD selection and messages."""
import datetime as calendar
import re
import socket
import tempfile
from pathlib import Path

from fixtures import HERE, REVIEWS, fixture
from pytest import MonkeyPatch

ops = fixture("probe_operators")
ops.CONTRACT = REVIEWS / "kickoff-fix-e.md"
handoff = fixture("probe_handoffs")
archived = fixture("probe_text", "contract-check-e")


def corrected_contract(label, **values):
    lines = ops.CONTRACT.read_text().splitlines()
    if label == "In the morning":
        original, replacement = re.findall(r"`([^`]+)`", next(
            line for line in lines if "**Codex B1 (the morning warning).**" in line
        ))
        template, = [line.split("`")[1] for line in lines if line.startswith(f"- **{label}:** `")]
        assert template.count(original) == 1
        return template.replace(original, replacement).format(**values)
    template, = [line.split("`")[1] for line in lines if line.startswith(f"- **{label}:** `")]
    return template.format(**values)


def guarded_settlement(root, mp):
    # The CLI can still install an override. Record separately what the revised
    # sentence permits before settlement and the consequence of ignoring it.
    ops.morning_settlement(root, mp)
    warning = ops.alert.send_dm.call_args_list[0].args[1]
    guard = "If prior entries are settled or absent and MLB differs"
    assert warning.index(guard) < warning.index("bts set-contest-streak")
    print("Pending prior game: morning text does not authorize the override.")
    checklist = (ops.ROOT / "docs/ops/2027-season-start.md").read_text()
    assert "(for the morning warning: only if prior entries are settled or absent)" in checklist
    assert "After verifying settlement or no prior entry in MLB" in checklist


def main(action):
    ops.exact_body = corrected_contract
    actions = {
        "skip": ops.inspect_skip,
        "warnings": ops.warnings_and_refusals,
        "settlement": guarded_settlement,
        "saver": ops.saver_cases,
        "repair": lambda root, mp: ops.sent_pick(root, mp, True),
        "control": lambda root, mp: ops.sent_pick(root, mp, False),
        "cli": ops.ops_cli,
        "research": ops.research,
        "restore": ops.first_boot,
        "paths": handoff.status_paths,
        "before-send": lambda root, mp: handoff.between_selection_and_send(root, mp, "before"),
        "during-send": lambda root, mp: handoff.between_selection_and_send(root, mp, "during"),
        "entry": handoff.private_entry_window,
    }
    with tempfile.TemporaryDirectory(dir=HERE) as name, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", archived.offline)
        mp.setattr(socket, "create_connection", archived.offline)
        mp.setattr(socket, "getaddrinfo", archived.offline)
        actions[action](Path(name), mp)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
