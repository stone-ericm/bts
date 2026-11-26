"""Round-one tick2 clock harness, fresh manifest/JUnit, and observed model fixture values."""
from pathlib import Path
import json
import time
from unittest.mock import patch
import sys
import pytest

review_real_monotonic=time.monotonic
review_tick_start=review_real_monotonic()
H=Path(__file__).resolve().parent
observations=[]
class ReviewPlugin:
    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self,item):
        if item.name not in ('test_run_uses_fresh_contest_streak_for_live_action',
                             'test_preview_uses_fresh_contest_streak_for_projected_pick'):
            return (yield)
        import bts.contest_state as cs
        real=cs.load_decision_streak_state
        seen=[]
        def record(*args,**kw):
            state=real(*args,**kw)
            seen.append(dict(streak=state.streak,model_streak=state.model_streak,
                             model_saver=state.model_saver_available))
            return state
        with patch.object(cs,'load_decision_streak_state',side_effect=record):
            result=yield
        assert seen and all(x['model_streak']==4 and x['streak']==7 for x in seen),seen
        observations.append(dict(test=item.nodeid,observed=seen))
        (H/f'r2-model-observations-{sys.argv[1]}.json').write_text(json.dumps(observations,indent=2))
        return result

source=(H/'test_family.py').read_text()
source=source.replace('value = fixed.astimezone(tz) if tz else fixed.astimezone().replace(tzinfo=None)',
    'current = fixed + dt.timedelta(seconds=review_real_monotonic() - review_tick_start)\n            value = current.astimezone(tz) if tz else current.astimezone().replace(tzinfo=None)')
source=source.replace('changed-files.txt','r2-changed-files.txt')
source=source.replace('"tests/test_streak_season_cs2_review.py"]',
    '"tests/test_streak_season_cs2_review.py", "tests/test_streak_season_cs2b_review.py"]')
source=source.replace("mode + '.xml'","'r2-' + mode + '.xml'")
source=source.replace("f\"--junitxml={HERE / ('r2-' + mode + '.xml')}\"]))",
    "f\"--junitxml={HERE / ('r2-' + mode + '.xml')}\"], plugins=[ReviewPlugin()]))")
assert 'plugins=[ReviewPlugin()]' in source
exec(compile(source,__file__,'exec'))
