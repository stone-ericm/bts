"""Run the actual inline workflow Python using the stock 3.11 interpreter."""
from pathlib import Path

from review_box import box, contained, invoke, record, RECOVER, tests

PY311 = Path("/Users/eric/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none/bin/python3.11")

for mode in ("active", "contest-nul", "recovery-clock-every"):
    recovery = mode.startswith("recovery")
    with box(record() if recovery else None) as path:
        runner = path / "shims/python3"
        runner.write_text(f"#!{PY311}\n" + tests.SHIM)
        common = {"scenario": mode}
        if recovery:
            common["script"] = RECOVER
        state, _ = invoke(path, "stock Python 3.11 " + mode, **common)
        if mode == "active":
            assert state["rc"] == 0 and state["attempt"] == record("committed")
        else:
            contained(state)
            if mode == "contest-nul":
                assert "Check contest-state failed (query)." in state["out"]
