"""Check loss of the last eligible early warning when a live all-clear takes 30 seconds."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from review_support import DAY, RIGHT, World, assert_source, health, row


source, tree = sys.argv[1:]
assert_source(Path(tree))
with TemporaryDirectory() as folder:
    root = Path(folder)
    current = World(root)
    current.commit_at('10:05', at='09:15')
    current.capture([RIGHT], f'{DAY}T132000Z')
    assert len(current.invoke('09:30', manual=True)['sent']) == 1
    tick = 1000

    def slow_live_dm(_, msg):
        global tick
        assert msg.startswith('✅') and current.read_marker()['status'] == 'confirmed'
        current.commit_at('10:05', at='09:44:55', game=102, batter=1020)
        early = current.cron('09:45', pending=[row()], dm='failed')
        assert early['receipt']['marker_status'] == 'dm_failed' and early['attempts'] and not early['sent']
        assert 'Batter 1020' in early['attempts'][0] and 'Batter 1010' not in early['attempts'][0]
        tick += 30

    live = current.invoke('09:44:50', manual=True, pending=[row()], patches={
        'bts.dm.send_dm': slow_live_dm, 'time.monotonic': lambda: tick,
    })
    receipt = live['receipt']
    assert receipt['marker_status'] == 'confirmed' and receipt['published_at'].endswith('09:45:20-04:00')
    assert current.early_state()['observed_at'].endswith('09:45:00-04:00')
    assert current.cron('10:00')['receipt']['outcome'] == 'outside_window'
    yesterday = World(root, day='2027-03-23')
    yesterday.commit_at('19:10', at='17:50')
    assert yesterday.invoke('18:00', manual=True)['marker']['date'] == yesterday.day
    alerts = health(current)
    print('old live marker wrote by 09:44:50; new pick warning failed 09:45; old receipt 09:45:20; '
          '10:00 outside window; displaced live marker; end-of-day alerts:', alerts)
    assert [level for level, _ in alerts] == (['CRITICAL'] if source == 'r1' else [])
