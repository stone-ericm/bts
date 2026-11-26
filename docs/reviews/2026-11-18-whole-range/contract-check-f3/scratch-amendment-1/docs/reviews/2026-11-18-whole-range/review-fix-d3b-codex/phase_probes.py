"""Test post-begin failures and signals on each phase command."""
import sys

from review_box import OLD, box, contained, invoke, record, tests

PHASES = (
    dict(tests.PHASE_TWO),
    {**dict.fromkeys(tests.PHASE_THREE), "stage-committed": "commit-record", **tests.PHASE_FOUR},
)


def run_cases(group, status, prior="none"):
    initial = None if prior == "none" else record(prior, owner="402" if prior != "contained" else "123")
    attempt = "2" if prior == "contained" else "1"
    for command, name in PHASES[int(group)].items():
        if name is None:
            name = command
        signal = not status.isdigit()
        with box(initial) as path:
            scenario = "legacy-present" if command.startswith("legacy-") else "active"
            fault = f"{command}:{status}"
            if signal and command in ("stage-committed", "publish-committed"):
                fault += ":unpublished"
            step = "interrupted" if signal else name
            state, calls = invoke(path, f"phase {fault=} {prior=}", scenario=scenario, fault=fault, attempt=attempt)
            assert (path / "fault-fired").exists()
            contained(state, step)
            assert state["out"].count(f"DEPLOY FAILED at {step}") == 1
            assert ["git", "reset", "--hard", OLD] not in calls


if __name__ == "__main__":
    run_cases(*sys.argv[1:])
