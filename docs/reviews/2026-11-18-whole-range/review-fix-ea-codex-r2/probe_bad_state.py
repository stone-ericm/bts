from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import bts
from bts import scheduler, web
from scripts.audit.season_walk import walk


work = Path(__file__).resolve().parent
mode = sys.argv[1]
src = work / 'base/src' if mode.startswith('base') else work.parent.parent / 'src'
assert Path(bts.__file__).is_relative_to(src), bts.__file__
print('Package:', bts.__file__)
day = '2027-07-15'
picks = work / 'state-wire' / 'picks'
old = dict(best_batter='A', best_team='T', best_p=0.75, bar=0.8, streak=8)
new = old | dict(basis='decision', candidate_verified=True)


def state(summary):
    return scheduler.SchedulerState(day, 'pinned', [], [], [], False, None, None, None,
                                    skip_summary=summary)


def render(summary):
    encoded = scheduler.save_state(state(summary), picks)
    with patch.object(web, 'PICKS_DIR', picks):
        decoded = web.load_scheduler_state(day)['skip_summary']
    assert decoded == json.loads(encoded.read_text())['skip_summary'] or 'NaN' in encoded.read_text()
    loaded = scheduler.load_state(day, picks).skip_summary
    dm = scheduler.format_skip_dm(day, loaded)
    banner = web.render_skip_banner(decoded)
    if banner:
        assert f'<span>{escape(dm.split(". ", 1)[1])}</span>' in banner
        with patch.object(web, 'PICKS_DIR', picks), patch.object(web, 'fetch_bluesky_posts', return_value=[]):
            assert banner in web.render_page()
    return dm, banner


with walk.offline(datetime.fromisoformat(day + 'T18:35:00-04:00')):
    if mode == 'base-write':
        scheduler.save_state(state(old), picks)
    elif mode == 'base-read':
        assert scheduler.load_state(day, picks).skip_summary == new
        print('base read branch file with added keys')
    else:
        class UnreadableClass:
            @property
            def __class__(self):
                raise RuntimeError('class lookup')
        decision = SimpleNamespace(objective='reach57', reason=None)
        frame = pd.DataFrame([{'batter_name': 'Top', 'team': 'T', 'p_game_hit': .75}])
        baseline = scheduler.build_skip_summary(frame, 8, .80, decision)
        assert scheduler.build_skip_summary(frame, 8, .80, decision, UnreadableClass()) == baseline
        assert baseline['candidate_verified'] is False
        print('Unreadable candidate leaves the base summary')
        assert scheduler.load_state(day, picks).skip_summary == old
        dm, banner = render(old)
        assert 'eligibility unverified' in dm and 'below the streak-8 bar (80.0%)' in dm
        print('branch read base file and rendered K2 unverified:', dm)
        base = dict(best_batter='<A>\r\nB', best_team='<T&>', best_p=.75,
                    bar=.8, streak=8, candidate_verified=True)
        bad = [-.1, 1.0000001, 2, 1e308, 1.5e308, 10**1000, float('nan'),
               float('inf'), float('-inf'), '0.99', [], {}, True, None, {'nested': ['<em>']}]
        for field in ('best_p', 'bar'):
            for value in bad:
                dm, banner = render(base | {field: value})
                assert 'No probability comparison' in dm and 'below the' not in dm + banner
                assert 'inf%' not in dm and '<A>' not in banner and '\n' not in dm + banner
                if field == 'best_p':
                    assert '(\u003cT&\u003e) ?.' in dm
        for summary in (
            base | {'reason': '\n \t\r\n', 'bar': None},
            base | {'reason': '<em>Stop!\nNow!</em>'},
            base | {'basis': 'no_decision', 'candidate_verified': False},
            base | {'basis': 'no_decision', 'reason': 'Tail reason', 'candidate_verified': False},
            base | {'basis': {'x': 'no_decision'}, 'bar': None, 'streak': '<b>'},
            base | {'basis': 1.0, 'bar': None, 'streak': 10**1000},
            dict(best_batter=['<em>'], best_team={'name': '<td>'}, best_p='x', bar=[], reason={},
                 streak=True, candidate_verified=1, basis=['decision']),
            None, [], ['X'], '', '<em>', 7, {},
        ):
            dm, banner = render(summary)
            assert 'Reason: .' not in dm and 'below the' not in dm + banner
            assert not any(tag in banner for tag in ('<em>', '<td>', '<A>'))
            print('stored malformed:', dm)
        dm, banner = render(base | {'reason': 'Observations\nbelow the expected count'})
        assert 'below the' in dm and 'below the' in banner
        print('Literal phrase can enter K1 from the persisted reason:', dm)
        scheduler.save_state(state(new), picks)
