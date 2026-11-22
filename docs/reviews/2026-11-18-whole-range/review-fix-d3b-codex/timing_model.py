"""Advance a virtual elapsed clock through each real shell timeout decision."""
import json
import sys

from review_box import DEPLOY, RECOVER, box, inject_shim, invoke, record, replace

SHIM = inject_shim(r'''
in_handler = root / "in-handler"
virtual_time = root / "home/projects/bts/data/virtual-elapsed"
if command == "python3" and args[2] == "clock" and virtual_time.exists():
    in_handler.touch()
    raise SystemExit(124)

if command == "timeout" and in_handler.exists():
    grace, term = map(int, args[1:3])
    delta = virtual_time
    elapsed = int(delta.read_text())
    prog = Path(args[3]).name
    words = args[4:]
    if prog == "sudo":
        prog = "systemctl" if "systemctl" in words else Path(words[2]).name
    if prog == "systemctl":
        scope = "user" if "--user" in words else "legacy"
        action = next(w for w in words if w in ("disable", "stop", "show", "list-jobs", "is-enabled"))
        label = f"{scope}-{action}"
        if action == "show" and words[-2:] == ["-p", "ControlGroup"] and "ActiveState" not in words:
            label += "-group"
    elif prog == "python3":
        label = words[words.index("-c") + 2]
    elif prog == "bash":
        label = "recheck" if any("containment re-checked" in word for word in words) else "log"
        if any("Scheduler stopped:" in word for word in words):
            label = "report"
    else:
        label = prog
    delays = {
        "id": 4, "user-show-group": 4, "user-disable": 4, "user-stop": 54,
        "legacy-show": 4, "legacy-list-jobs": 4, "legacy-show-group": 4,
        "legacy-disable": 4, "legacy-stop": 54, "user-is-enabled": 14,
        "user-show": 14, "user-list-jobs": 14, "cgroup": 14, "legacy-is-enabled": 14,
        "pgrep": 40, "marker": 40, "endpoint": 14, "curl": 10,
        "stage-record": 20,
    }
    wait = delays.get(label, 0)
    if label == "legacy-list-jobs" or label == "legacy-show" and elapsed >= 180:
        wait = 20
    if label == "endpoint":
        wait = min(wait, term - 1)
    if scenario == "early":
        wait = 0
    if scenario == "late-successful" and label in ("pgrep", "marker"):
        wait = min(wait, term - 1)
    if scenario == "late-successful" and label == "stage-record":
        wait = 1
    rounded_success = False
    if scenario == "blocked-final-recheck":
        if label.startswith("legacy-") and elapsed >= 198 or label == "cgroup" and elapsed >= 198:
            wait = 0
        if label in ("pgrep", "marker", "endpoint", "stage-record"):
            wait = term
            # The command finishes just before TERM. Fractional starts and ends
            # can advance whole Bash SECONDS by the nominal term.
            rounded_success = True
        if label in ("mv", "rm", "report", "recheck"):
            wait = term + grace
    timed_out = wait >= term and not rounded_success
    end = elapsed + (term + grace if timed_out else wait)
    delta.write_text(str(end))
    with (root / "timeline").open("a") as log:
        print(json.dumps(dict(label=label, start=elapsed, end=end, term=term, grace=grace,
                              timed_out=timed_out, scope=prog)), file=log)
    if timed_out:
        raise SystemExit(124)

if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"]:
    states = dict(Id=args[1], ActiveState="inactive", MainPID="0", LoadState="loaded", ControlGroup="/legacy")
    for key in args:
        if key in states:
            print(f"{key}={states[key]}")
    raise SystemExit(0)
''')


def instrument(script):
    script = replace(script, "local budget_start=$SECONDS budget_origin\n",
                     'SECONDS=0\n    printf 0 > "$checkout/data/virtual-elapsed"\n    local budget_start=$SECONDS budget_origin\n')
    script = replace(script, 'if [[ -n ${budget_start:-} ]]; then\n        now=',
                     'if [[ -n ${budget_start:-} ]]; then\n'
                     '        if [[ -e $checkout/data/virtual-elapsed ]]; then\n'
                     '            IFS= read -r SECONDS < "$checkout/data/virtual-elapsed"\n'
                     '        fi\n        now=')
    return script


def model(recovery, scenario, comparison=None):
    script = RECOVER if recovery else DEPLOY
    if comparison == "old-proof":
        script = replace(script, "deadline=$((containment_deadline - 75000))",
                         "deadline=$((containment_deadline - 60000))")
    with box(record() if recovery else None, shim=SHIM) as path:
        state, _ = invoke(path, f"K13/K14 {recovery=} {scenario=} {comparison=}", instrument(script),
                          scenario=scenario, fault="" if recovery else "reset:17")
        timeline = [json.loads(line) for line in (path / "timeline").read_text().splitlines()]
        print(json.dumps(timeline), flush=True)
        assert (path / "in-handler").exists() and state["rc"] == 1
        steps = {entry["label"]: entry for entry in timeline}
        if comparison:
            assert "pgrep" not in steps
            return
        residual, marker, endpoint, post = (steps[key] for key in ("pgrep", "marker", "endpoint", "curl"))
        assert residual["term"] >= 5 and residual["end"] <= 220
        assert marker["start"] >= residual["end"] and marker["end"] <= 240
        assert endpoint["end"] <= 254 and post["term"] >= 10 and post["end"] <= 270
        assert not steps["endpoint"]["timed_out"] and state["posts"] == 1
        assert timeline[-1]["end"] <= 280
        if scenario == "early":
            assert marker["start"] < 220 and post["start"] < 240
        else:
            assert residual["start"] >= 195
            assert steps["endpoint"]["start"] >= (230 if scenario == "cap-stress" else 220)
            assert "Scheduler stopped: unverified." in state["out"] or comparison


if __name__ == "__main__":
    if len(sys.argv) == 2:
        initial = record("contained", tail="head unverified unverified unverified")
        with box(initial, shim=SHIM) as path:
            state, _ = invoke(path, "M2 blocked final reports", instrument(RECOVER),
                              "blocked-final-recheck")
            timeline = [json.loads(line) for line in (path / "timeline").read_text().splitlines()]
            print(json.dumps(timeline), flush=True)
            assert state["rc"] == 1 and state["attempt"] == initial
            assert state["stopped"] and state["posts"] == 1 and not state["out"]
            assert [(x["label"], x["start"], x["end"]) for x in timeline[-2:]] == [
                ("report", 277, 279), ("recheck", 279, 281)]
    else:
        for recovery in (False, True):
            for scenario in ("early", "cap-stress", "late-successful"):
                model(recovery, scenario)
            model(recovery, "cap-stress", "old-proof")
