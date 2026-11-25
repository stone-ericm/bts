"""Exercise candidate classification failure with a usable prediction frame."""
import importlib.util
import sys
from types import SimpleNamespace

import pandas as pd

from fixtures import HERE
from bts import scheduler

spec = importlib.util.spec_from_file_location("base_scheduler", HERE / "base/src/bts/scheduler.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class Candidate:
    @property
    def __class__(self):
        raise ValueError("candidate type unavailable")


frame = pd.DataFrame([{"p_game_hit": .75, "batter_name": "Usable", "team": "A"}])
decision = SimpleNamespace(reason=None, objective="fixture")
print("base:", base.build_skip_summary(frame, 10, .8, decision))
print("head control:", scheduler.build_skip_summary(frame, 10, .8, decision, None))
try:
    print("head with raising candidate:", scheduler.build_skip_summary(frame, 10, .8, decision, Candidate()))
except ValueError as exc:
    print("HEAD RAISED:", exc)
else:
    raise AssertionError("the type probe no longer escapes the display-only guard")
