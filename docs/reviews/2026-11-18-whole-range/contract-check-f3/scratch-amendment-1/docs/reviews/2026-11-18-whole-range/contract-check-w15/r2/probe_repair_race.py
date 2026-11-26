from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os

from probe_features import ROOT, SIG, compute, feed, put


def main():
    with TemporaryDirectory(prefix='repair-race-', dir=ROOT / 'runs') as name:
        work = Path(name)
        cache = work / 'data/models/probable_pitcher_lookup.json'
        prepared = cache.with_name('prepared-restore.json')
        raw = work / 'data/raw'
        put(cache, {'111': SIG})
        put(prepared, {'111': SIG, '777': None, '888': SIG})
        feed(raw / '2025/100.json', SIG)
        compute._probable_pitcher_cache_path = cache
        replace = os.replace
        actions = []

        def concurrent_replace(staged, destination):
            assert destination == cache
            actions.append(('scanner staged', list(json.loads(staged.read_text()))))
            replace(prepared, destination)
            actions.append(('other writer committed', list(json.loads(cache.read_text()))))
            replace(staged, destination)
            actions.append(('scanner committed', list(json.loads(cache.read_text()))))

        with patch.object(os, 'replace', concurrent_replace):
            assert compute._build_probable_pitcher_lookup(str(raw)) == {111: SIG, 100: SIG}
        assert list(json.loads(cache.read_text())) == ['111', '100']
        for label, keys in actions:
            print(label, keys)
        print('A clean read satisfies the v2 write gate, yet its replace erased the later cache-only game and bad member.')


if __name__ == '__main__':
    main()
