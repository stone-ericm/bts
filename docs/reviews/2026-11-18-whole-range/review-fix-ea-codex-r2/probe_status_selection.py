from contextlib import redirect_stderr
from datetime import datetime, timedelta
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import orchestrator, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk


now = datetime.fromisoformat('2027-07-15T10:00:00-04:00')
day = now.date().isoformat()
work = Path(__file__).resolve().parent


def install_dated(path, streak, saver):
    rc, output = walk._invoke('set-contest-streak', '--date', day, '--streak', streak,
                             '--best-streak', 14, '--saver', saver, '--picks-dir', path)
    assert rc == 0, output


with TemporaryDirectory(dir=work) as tmp, walk.offline(now):
    path = Path(tmp) / 'picks'
    path.mkdir()
    cfg = walk._config(path) | {'bluesky': {'dm_recipient': 'capture.invalid'}}
    assert walk.install_state(path, now.date(), walk.State('automatic', 12, 14, 'active', .9)) == []
    auto = path / 'account_state/contest_streak.json'
    record = json.loads(auto.read_text()) | {'recorded_at': (now - timedelta(hours=20)).isoformat()}
    auto.write_text(json.dumps(record))
    walk._set_clock(now)
    capture = []
    with patch.object(alert, 'send_dm', side_effect=lambda _, body: capture.append(body) or 'alert-id'):
        scheduler._morning_state_check(cfg, path, day, now, now + timedelta(hours=8))
    assert 'if prior entries are settled or absent and MLB differs from status' in capture[0]
    install_dated(path, 3, 'used')
    rc, status = walk._invoke('season-contract', 'status', '--picks-dir', path)
    assert rc == 0 and 'streak=3; status=fresh; saver=used' in status
    assert 'automatic path alone: accepted (streak 12' in status and 'saver: active; season: 2027' in status
    print('Reading the morning warning before the first lineup check, MLB 12/14/active.')
    print(status)
    walk._set_clock(now + timedelta(hours=8))
    with redirect_stderr(StringIO()), patch.object(picks, 'get_game_statuses_detailed', return_value={
            pk: {'abstract': 'P', 'detailed': 'Scheduled'} for pk in (70000, 70001, 70002)}) as statuses:
        differences = []
        for p in (0.70, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.90, 0.95, 0.99):
            frame = walk._predictions(now.date(), p)
            with patch.object(orchestrator, 'run_cascade', return_value=(frame, 'pinned')):
                install_dated(path, 3, 'used')
                _, apparent_match, _ = orchestrator.run_and_pick(cfg, day)
                install_dated(path, 12, 'active')
                _, live_corrected, _ = orchestrator.run_and_pick(cfg, day)
            if apparent_match.action != live_corrected.action:
                differences.append((p, apparent_match.action, live_corrected.action))
    assert statuses.call_count > 0 and differences
    print('Probability, action with wrong dated reading, action with correct dated reading:', differences)
