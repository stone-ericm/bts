"""Broader legacy/current-season matrices; compare normalized JSONL across src versions."""
import inspect
import itertools
import json
import sys
import tempfile
from pathlib import Path
from bts import picks as p

root = Path(__file__).resolve().parent
label = sys.argv[1]
seasonal = "season" in inspect.signature(p.load_streak).parameters
assert seasonal == (label == "head")
run = Path(tempfile.mkdtemp(prefix=label + "-matrix-", dir=root / "runs"))
out = []
raws = [b"{", b"\xff", b"[]", b"null", b"true", b"42", b'"text"']
for tag, streak, saver, dates in itertools.product(
        ["absent", 2027, 2026, True, "2027", 2027.0, None],
        ["absent", 0, 12, "wrong", None], ["absent", True, False],
        ["absent", [], ["2027-06-10"], [7, "2027-06-09"], None]):
    obj = {k: v for k, v in [("season", tag), ("streak", streak), ("saver_available", saver), ("applied_dates", dates)] if v != "absent"}
    raws.append(json.dumps(obj).encode())
for i, b in enumerate(raws):
    for op in ["read", "save", "explicit-save", "update", "dated-current"]:
        d = run / str(i) / op
        d.mkdir(parents=True)
        f = d / "streak.json"
        f.write_bytes(b)
        try:
            if op == "read":
                result = [p.load_streak(d), p.load_saver_available(d), p.load_applied_dates(d)]
            elif op == "save":
                result = p.save_streak(7, d)
            elif op == "explicit-save":
                result = p.save_streak(7, d, saver_available=False)
            elif op == "update":
                result = p.update_streak([True], d)
            else:
                # All valid objects tagged current season must retain prior dated semantics.
                try:
                    obj = json.loads(b)
                except Exception:
                    continue
                if not isinstance(obj, dict) or type(obj.get("season")) is not int or obj["season"] != 2027:
                    continue
                result = p.update_streak([False], d, date="2027-06-10")
            outcome = {"value": result}
        except BaseException as exc:
            outcome = {"error": type(exc).__name__}
        try:
            saved = json.loads(f.read_bytes())
            if isinstance(saved, dict):
                saved.pop("updated", None)
                saved.pop("season", None)  # expected intentional tag preservation
        except Exception:
            saved = f.read_bytes().hex()
        out.append({"input": b.hex(), "op": op, "outcome": outcome, "saved": saved})
(root / (label + "-matrix.json")).write_text(json.dumps(out, indent=2) + "\n")
print(label, len(out), "cases", p.__file__)
