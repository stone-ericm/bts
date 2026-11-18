"""Verify libclock_interpose.dylib shifts Python wall clock in 2027 and 2028
cleanly with pandas, ZoneInfo, and bts modules while leaving monotonic time untouched.
"""

from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
DYLIB = REVIEW_DIR / "libclock_interpose.dylib"

for year, iso in ((2027, "2027-07-15T12:00:00-04:00"), (2028, "2028-07-15T12:00:00-04:00")):
    epoch = int(datetime.fromisoformat(iso).timestamp())
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["DYLD_INSERT_LIBRARIES"] = str(DYLIB)
    env["CS6B2_TARGET_EPOCH"] = str(epoch)
    code = (
        "from datetime import date, datetime; import time, pandas as pd; "
        "from zoneinfo import ZoneInfo; from bts import scheduler, strategy; "
        "et = ZoneInfo('America/New_York'); now = datetime.now(et); "
        f"assert now.year == {year} and (now.month, now.day) == (7, 15), now; "
        f"assert date.today().year == {year}, date.today(); "
        "m0 = time.monotonic(); time.sleep(0.02); m1 = time.monotonic(); "
        "assert 0.01 <= (m1 - m0) < 1.0, (m0, m1); "
        "print('OK', now.isoformat(), date.today().isoformat())"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"{year} failed: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    print(year, proc.stdout.strip())
