"""Runtime adapters for the pinned clock, a denied socket and a raw stdout case."""

import argparse
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent


def run_with(source, find, replace):
    path = HERE / source
    content = path.read_text()
    assert content.count(find) == 1, find
    exec(compile(content.replace(find, replace), str(path), "exec"), {"__file__": str(path)})


parser = argparse.ArgumentParser()
parser.add_argument("case", choices=("pin-both-clocks", "skip-denied-unix-socket", "nul-inside-prefix"))
case = parser.parse_args().case
if case == "pin-both-clocks":
    find = '        stack.enter_context(patch.object(cli_module, "_now_et", return_value=now))'
    run_with("canary_probe.py", find, find + '\n        stack.enter_context(patch.object(cli_module, "_today_et", return_value=now.date().isoformat()))')
elif case == "skip-denied-unix-socket":
    run_with("heartbeat_probe.py", '"socket": None, ', "")
else:
    namespace = runpy.run_path(str(HERE / "shell_probe.py"))
    run = namespace["run"]
    old = r"not a live configuration (delivery private);\x00 contest state not checked."
    new = r"not a li\x00ve configuration (delivery private); contest state not checked."
    shims = run.__globals__["SHIM"]
    assert shims.count(old) == 1
    run.__globals__["SHIM"] = shims.replace(old, new)
    data = run("contest-nul-inside-prefix", scenario="contest-nul")
    assert data["status"] == 0 and "CANARY PASSED" in data["stdout"]
