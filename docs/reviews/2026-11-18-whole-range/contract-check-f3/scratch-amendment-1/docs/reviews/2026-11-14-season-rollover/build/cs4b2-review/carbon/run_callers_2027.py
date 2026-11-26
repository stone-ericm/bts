"""Run probe_callers_v4 with dt.datetime patched to 2027-07-15 BEFORE bts modules import `from datetime import datetime`."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
import pandas  # noqa: F401 - bind real C datetime types first

_REAL_DT = dt.datetime
_NOW = _REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))


class _Clock2027(_REAL_DT):
    @classmethod
    def now(cls, tz=None):
        return _NOW.astimezone(tz) if tz is not None else _NOW.replace(tzinfo=None)


dt.datetime = _Clock2027

_orig_dumps = json.dumps
json.dumps = lambda obj, *a, **kw: _orig_dumps(obj, *a, **({"default": bool} | kw))

import probe_callers_v4

if __name__ == "__main__":
    if "--worker" in sys.argv:
        idx_s = sys.argv.index("--src")
        idx_o = sys.argv.index("--out")
        probe_callers_v4.run_worker(Path(sys.argv[idx_s + 1]), Path(sys.argv[idx_o + 1]))
    else:
        import subprocess
        repo = Path(__file__).resolve().parents[2]
        work = Path(__file__).resolve().parent
        base_p = work / "callers_base_2027.json"
        branch_p = work / "callers_branch_2027.json"
        me = Path(__file__).resolve()
        for src, out_p in [(work / "base/src", base_p), (repo / "src", branch_p)]:
            proc = subprocess.run(
                [sys.executable, str(me), "--worker", "--src", str(src), "--out", str(out_p)],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"worker failed ({src}):\n{proc.stdout}\n{proc.stderr}")
        base_c = json.loads(base_p.read_text(encoding="utf-8"))
        branch_c = json.loads(branch_p.read_text(encoding="utf-8"))
        assert base_c["denied_network"] == [] and branch_c["denied_network"] == []
        for sub in ("post_save", "skip", "locked_catchup"):
            assert branch_c[f"orchestrate_mid_revoke_{sub}"]["raised"] is True
            assert branch_c[f"orchestrate_mid_revoke_{sub}"]["posts"] == 0
        assert branch_c["preview_plus_fallback_cascade_failure"]["still_uncovered_dm_delivered"] is False
        assert branch_c["preview_plus_fallback_cascade_failure"]["after_dated_56_cascade_fail_delivered_cached_double"] is True
        print(json.dumps({"branch_2027": branch_c}, indent=2))
