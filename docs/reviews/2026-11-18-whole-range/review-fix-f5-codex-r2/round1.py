"""Re-run the original reviewer functions, checking the formerly failing rows explicitly."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from support import legacy, outcome, source

alerts = legacy("alert_probe")


def record(function, root, version, *args):
    state = {}

    def trace(frame, event, arg):
        if frame.f_code == function.__code__ and event == "return":
            state.update(frame.f_locals)
        return trace

    sys.settrace(trace)
    try:
        function(root, version, *args)
    except AssertionError:
        if version == "base":
            raise
        print("original round-1 expected-failure assertion fired:", function.__name__)
    else:
        assert version == "base", "the original round-1 assertion unexpectedly still passed"
    finally:
        sys.settrace(None)
    return state


def warnings(trace):
    slots = ("10:00", "10:15", "10:30")
    assert all("pending" in trace[slot]["calls"] and len(trace[slot]["sent"]) == 1 for slot in slots)
    return [(slot, trace[slot]["sent"]) for slot in slots]


def run(root, version):
    alerts.closing_fixture(root, version)
    for case in ("bind-wrong", "remove-entry", "late-commit"):
        alerts.shared_window(root, version, case)
    corrected = record(alerts.corrected_pitch, root, version)
    trace = dict(corrected["trace"])
    if version == "f5":
        assert trace["09:45"]["receipt"]["qualification"]["slots"][0]["state"] == "unit_unverified"
    assert corrected["morning"][1]["state"] == "entry_unverifiable" and len(corrected["morning"][2]) == 1
    print("R1", warnings(trace), corrected["morning"][1]["state"])
    replaced = record(alerts.revised_window, root, version, False)
    assert replaced["morning"][1]["state"] == "entry_unverifiable" and len(replaced["morning"][2]) == 1
    print("R2", warnings(dict(replaced["trace"])), replaced["morning"][1]["state"])
    missing = record(alerts.revised_window, root, version, True)
    assert "BTS pick NOT entered" in missing["ten"]["sent"][0] and len(missing["ten"]["sent"]) == 1
    print("R3", missing["ten"]["sent"])
    health = record(alerts.previous_date_health, root, version)
    assert [alert.level for alert in health["alerts"]] == ["CRITICAL"]
    assert len(health["sent"]) == 1 and "the operator was never reached" in health["sent"][0]
    if version == "f5":
        assert outcome(health["result"]) == "observed"
    print("R4", health["sent"])


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        run(Path(directory), version)
    if version == "f5":
        legacy("mutation_witness").replay("candidate")
