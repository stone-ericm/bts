"""Re-run the round-one table inputs against the checked-out workflow."""
import json
import sys

from shared import HERE, RECOVER, box, invoke, record
import required
import timing_model as timing


def timing_windows():
    for recovery in (False, True):
        for scenario in ("early", "cap-stress", "late-successful"):
            timing.model(recovery, scenario)
        timing.model(recovery, "cap-stress", "old-proof")


def final_print():
    initial = record("contained", tail="head unverified unverified unverified")
    with box(initial, shim=timing.SHIM) as path:
        result, calls = invoke(path, "Codex B3 original late blocked-output model",
                               timing.instrument(RECOVER), "blocked-final-recheck")
        timeline = [json.loads(line) for line in (path / "timeline").read_text().splitlines()]
        print(json.dumps(timeline), flush=True)
        print("final calls", [(e["label"], e["start"], e["end"], e["term"], e["grace"])
                              for e in timeline if e["label"] in ("report", "recheck")])
        assert result["rc"] == 1 and result["attempt"] == initial and result["posts"] == 1 and not result["out"]
        assert [(x["label"], x["start"], x["end"]) for x in timeline[-1:]] == [("report", 277, 279)]
        assert not any(e["label"] == "recheck" for e in timeline)
        print("PASS shared print bounded by 2 seconds")
    with box(initial) as path:
        result, calls = invoke(path, "unblocked repeat with fresh stop, marker and page", RECOVER)
        assert result["out"].endswith("DEPLOY RECOVERY — containment re-checked.\n")
        final = [c for c in calls if c[0] == "timeout" and any("Scheduler stopped:" in word for word in c)]
        assert len(final) == 1 and final[0][1:4] == ["-k", "1", "1"]
        assert any("containment re-checked." in word for word in final[0])


def commit_control():
    transcript = (HERE / "table-commit.log").read_text().splitlines()
    cases = [json.loads(line) for line in transcript if line.startswith("{")]
    assert len(cases) == 9
    for state in cases:
        expected = "126" if "K15" in state["case"] else "17" if ":17" in state["case"] else "timeout"
        assert f"Check commit-record failed ({expected})." in state["out"]
        assert state["rc"] == 1 and state["stopped"] and "contained commit-record" in state["attempt"]
    with box() as path:
        state, _ = invoke(path, "K5 rename performed, wrapper exits 17", scenario="committed-with-error")
        assert state["rc"] == 0 and "CANARY PASSED" in state["out"] and state["attempt"] == record("committed")


if __name__ == "__main__":
    name, *args = sys.argv[1:]
    if name in ("timing_windows", "final_print", "commit_control"):
        globals()[name]()
    else:
        getattr(required, name)(*args)
