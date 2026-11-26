import importlib.util
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent / "scratch"
CARBON = ROOT / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3"


def read_module(name):
    spec = importlib.util.spec_from_file_location(name, CARBON / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.REPO_ROOT = ROOT
    return module


def run(module_name, name, captured_lines):
    fn = getattr(read_module(module_name), name)
    captured = {}

    def trace(frame, event, _):
        if event == "line" and frame.f_code is fn.__code__ and frame.f_lineno in captured_lines:
            captured.update(frame.f_locals)
        return trace

    with tempfile.TemporaryDirectory() as temp:
        sys.settrace(trace)
        try:
            fn(Path(temp))
        except AssertionError as error:
            print(f"{name} stopped with assertion: {error}", flush=True)
        else:
            print(f"{name} passed", flush=True)
        finally:
            sys.settrace(None)
    return captured


dated = run("test_slice3_base_and_edges",
            "test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms", {267})
r_out = dated["r_out"]
import json
out = json.loads(r_out.output)
sent = dated["sent_dms"]
print("Archived dated assertion 267", r_out.exit_code == 0)
print("Archived dated assertion 269", out["state"] == "deliberate_skip" and out["ok"] is True)
print("Archived dated assertion 271", len(sent) == 1 and sent[0][0] == "16:10" and "No pick yet" in sent[0][1])

revoke = run("test_slice3_interactions", "test_item1d_live_refusal_vs_health_and_day_outcome", {724})
print("Archived revoke assertion 725", revoke["out2"].state == "deliberate_skip" and revoke["out2"].ok is True)
print("Archived revoke assertion 726", revoke["cli_do"].exit_code == 0)
