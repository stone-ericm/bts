"""Tick against a saved monotonic function so test monkeypatches cannot consume clock calls."""
from pathlib import Path
import time
review_real_monotonic = time.monotonic
review_tick_start = review_real_monotonic()
source = (Path(__file__).resolve().parent / "test_family.py").read_text()
source = source.replace("value = fixed.astimezone(tz) if tz else fixed.astimezone().replace(tzinfo=None)",
    "current = fixed + dt.timedelta(seconds=review_real_monotonic() - review_tick_start)\n            value = current.astimezone(tz) if tz else current.astimezone().replace(tzinfo=None)")
exec(compile(source, __file__, "exec"))
