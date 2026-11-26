"""Pin the exact new test payload and the existing symlink precheck behavior."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile

from probe import ROOT, NOW, BAD, before, head, observation, error, normalized

deep = runpy.run_path("tests/test_contest_state_decoder_limits.py")["DEEP"]
assert deep == BAD["deep"]
result = {"actual_test_DEEP": {"sha256": hashlib.sha256(deep).hexdigest(), "bytes": len(deep), "decode": error(lambda: json.loads(deep.decode()), RecursionError)}, "model_symlink": []}
for kind in (None, "manual", "auto"):
    for required in (False, True):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="symlink-model-") as tmp:
            p = Path(tmp)
            path = p / "streak.json"
            path.symlink_to("streak.json")
            if kind:
                observation(p, kind)
            raw = error(lambda: path.read_text(encoding="utf-8"), OSError)
            assert not path.exists()
            current = normalized(lambda: head.load_decision_streak_state(p, require_contest_state=required, now=NOW))
            old = normalized(lambda: before.load_decision_streak_state(p, require_contest_state=required, now=NOW))
            assert current == old
            parsed = json.loads(current)
            if kind:
                state = parsed["result"]
                assert state["streak"] == 7 and state["model_streak"] == 0
                assert state["model_saver_available"] is True and state["model_state_error"] is None
            elif required:
                assert parsed["exception"] == "ContestStateError" and "required but missing" in parsed["message"]
            else:
                assert parsed["result"]["streak"] == 0 and parsed["result"]["status"] == "model_only"
            result["model_symlink"].append({"observation": kind, "required": required, "raw_read": raw, "head_and_before": parsed})
(ROOT / "supplemental-results.json").write_text(json.dumps(result, indent=2))
print("Actual new test DEEP raises RecursionError; all six symlink-model cases unchanged.")
