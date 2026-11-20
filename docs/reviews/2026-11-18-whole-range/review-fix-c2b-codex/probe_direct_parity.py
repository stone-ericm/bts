"""Compare the direct marked-attempt branch on the base and HEAD."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from unittest.mock import Mock

REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW / "replay" / "codex"))
from probe_support import attempt_direct, day_pick, deadline, fixtures, raw, status_body, world_at

from bts import daily_decision, dm, picks, posting, scheduler
from bts.health import alert


DAY = "2027-03-25"


def branch(root, patch, opening, version, prior):
    with world_at(root, patch, opening, opening.at(DAY, "18:20")) as (config, directory, _, _, scoped):
        daily = day_pick(DAY)
        daily.delivery_attempted = True
        disk_pick = picks.save_pick(daily, directory)
        path = daily_decision.decision_path(DAY, directory)
        if prior == "delivered-record":
            daily_decision.write_decision(DAY, directory, action="single", source="mdp", scoreable=True,
                                          delivery_status="delivered", primary={"batter_id": daily.pick.batter_id})
        if prior == "writer-none":
            scoped.setattr(daily_decision, "write_decision", Mock(return_value=None))
        before = disk_pick.read_bytes()
        observed = []

        def capture(_, body):
            state = scheduler.load_state(DAY, directory)
            observed.append((body, state.pick_locked, state.committed_pick_written, raw(path)))
            return "direct-parity-id"

        alert.send_dm.side_effect = capture
        snapshots = []
        with redirect_stderr(StringIO()) as stderr:
            for _ in range(2):
                assert attempt_direct(config, directory, daily) is False
                state = scheduler.load_state(DAY, directory)
                record = raw(path)
                assert disk_pick.read_bytes() == before
                snapshots.append([state.pick_locked, state.pick_locked_at, state.committed_pick_written,
                                  record.hex() if record is not None else None])
        assert len(observed) == 1
        body, locked, committed, record = observed[0]
        assert "DELIVERY OUTCOME UNKNOWN" in body
        if version == "head":
            assert body == status_body(DAY)
        assert locked and committed is (prior != "writer-none") and record == raw(path)
        dm.send_dm.assert_not_called()
        posting.post_to_bluesky.assert_not_called()
        assert "NOT re-sending" in stderr.getvalue()
        return {"snapshots": snapshots, "pick": before.hex(), "journal": stderr.getvalue()}


if __name__ == "__main__":
    version, = sys.argv[1:]
    expected = REVIEW / "direct-base-traces.json"
    phases = ("fresh", "delivered-record", "writer-none")
    with deadline(40), fixtures("direct-parity") as (root, patch, opening):
        found = {phase: branch(root / phase, patch, opening, version, phase) for phase in phases}
    if version == "base":
        expected.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n")
    else:
        assert found == json.loads(expected.read_text())
    print(f"{version}: {len(phases)} direct scenarios with captured pick/decision bytes, locks and journal")
