from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import probe_live as live
from bts import scheduler, web


def dash(directory, expected):
    with patch.object(web, 'PICKS_DIR', directory), patch.object(web, 'fetch_bluesky_posts', return_value=[]), \
            patch.object(web, '_build_live_game_data', return_value=([], None)):
        html = web.render_page()
    print('dashboard', f'class="streak-number">{expected}<' in html, 'leaderboard',
          live._tab_streak_fields('active_streak', dict(streak=expected, activeStreak=expected)),
          live._tab_streak_fields('all_season', dict(streak=expected, activeStreak=12)))
    assert f'class="streak-number">{expected}<' in html


classification = scheduler._write_classification_decision


def finalize(directory, day, *, selection=None, **kwargs):
    if selection is None:
        return classification(directory, day, **kwargs)
    state = kwargs['state']
    scheduler._capture_fallback_skip(state, SimpleNamespace(selection=selection))
    return scheduler._write_endofday_skip(directory, day, state)


live.dash = dash
with patch.object(scheduler, '_write_classification_decision', finalize):
    live.main()
