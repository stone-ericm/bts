"""Repeat the delivery probe with a live-mode observation for the manual-path fixture."""

from pathlib import Path
import sys
from unittest.mock import patch

import delivery_probe
from probe_support import BASE, HEAD, archive, offline, python, scratch


def corrected_manual_commit(root):
    from bts import contest_state

    # Manual run legitimately requires an automatic or dated live observation.
    # Keep the real chosen_on and saved_pick_refusal logic in the CLI.
    snapshot = contest_state.DecisionStreakState(9, True, True, "contest", "fresh", 9, True,
                                                observation_kind="automatic")
    with patch.object(contest_state, "DecisionStreakState", return_value=snapshot):
        return delivery_probe.manual_run_commit_original(root)


if __name__ == "__main__":
    with offline():
        if len(sys.argv) > 1:
            delivery_probe.manual_run_commit_original = delivery_probe.manual_run_commit
            delivery_probe.manual_run_commit = corrected_manual_commit
            delivery_probe.scenario_runs(Path(sys.argv[1]))
        else:
            with scratch("delivery-followup") as work:
                work = Path(work)
                for revision in (BASE, HEAD):
                    source = archive(revision, work / revision, "src") / "src"
                    print(revision, flush=True)
                    try:
                        proc = python(source, __file__, work / f"data-{revision}", timeout=20)
                        print(proc.stdout, end="")
                    except Exception as error:
                        if hasattr(error, "stdout"):
                            print(error.stdout, error.stderr, file=sys.stderr)
                        raise
                delivery_probe.crash_marker(work)
