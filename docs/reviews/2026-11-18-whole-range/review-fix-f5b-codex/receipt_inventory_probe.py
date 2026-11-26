from datetime import datetime, timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from bts import entry_receipt, receipt_io
from review_support import ROOT, RIGHT, World, assert_source, health, row


def health_with_sealed_recovery(root, count):
    day = '2027-07-15'
    w = World(root, day=day)
    w.commit_at('06:10', at='04:55')
    previous = World(root, day='2027-07-14')
    previous.commit_at('19:10', at='17:55')
    assert previous.invoke('18:00', manual=True, pending=[row()])['marker']['status'] == 'confirmed'

    assert w.cron('05:00', dm='failed')['receipt']['marker_status'] == 'dm_failed'
    assert health(w)[0][0] == 'CRITICAL'
    first = datetime(2027, 3, 25)
    for index in range(count):
        stamp = first + timedelta(hours=8 * index)
        w.capture([{**RIGHT, 'id': 1000 + index}], stamp.strftime('%Y-%m-%dT%H%M%SZ'))
    w.capture([RIGHT], f'{day}T085000Z')
    fixed = w.invoke('05:15', manual=True, pending=[row()])
    assert fixed['receipt']['qualification']['all_confirmed'] and fixed['receipt']['marker_status'] == 'confirmed'
    path, = [path for path in receipt_io.discover(entry_receipt.receipts_root(w.picks) / day)
             if fixed['receipt']['attempt_id'] in path.name]
    size = path.stat().st_size
    assert (size > 64 * 1024) is (count > 1)
    assert health(w) == []
    assert previous.invoke('18:15', manual=True, pending=[row()])['marker']['date'] == previous.day
    alerts = health(w)
    print('capture count', count + 1, 'sealed receipt bytes', size, 'observed health', alerts, flush=True)
    assert alerts == [], 'a confirmed sealed recovery must prevent an old early health alert from returning'


def precedence_without_seal(root):
    w = World(root)
    w.commit_at('06:10', at='04:55')
    assert w.cron('05:00', dm='failed')['receipt']['marker_status'] == 'dm_failed'
    w.capture([RIGHT], f'{w.day}T085000Z')

    def fail(*_):
        raise OSError('synthetic publication failure')

    fixed = w.invoke('05:15', manual=True, patches={'bts.entry_receipt.receipt_io.publish': fail}, pending=[row()])
    assert fixed['marker']['status'] == 'confirmed' and fixed['receipt'] is None
    print('same-date live confirmed with no sealed receipt', health(w))
    assert health(w) == []


if __name__ == '__main__':
    assert_source(ROOT / 'src')
    with TemporaryDirectory() as dirname:
        root = Path(dirname)
        if sys.argv[1:] == ['precedence']:
            precedence_without_seal(root)
        else:
            health_with_sealed_recovery(root / 'short', 1)
            health_with_sealed_recovery(root / 'full', 325)
