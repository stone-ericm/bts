from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os

from probe_features import ROOT, compute, feed, feature, put


def main():
    with TemporaryDirectory(prefix='raw-', dir=ROOT / 'runs') as name:
        previous = Path.cwd()
        os.chdir(name)
        try:
            cache = Path('data/models/cache.json')
            raw = Path('data/raw')
            compute._probable_pitcher_cache_path = cache
            put(cache, {'100': True})
            assert compute._build_probable_pitcher_lookup() == {100: True}
            print('boolean whole entry:', *feature()[:2])
            put(cache, {})

            raw.write_text('{}')
            print('regular file in place of raw root:', *feature()[:2])
            raw.unlink()
            feed(raw / '2025/².json')
            assert '²'.isdigit() and not '²'.isdecimal()
            print('isdigit raw stem outside guarded extraction:', *feature()[:2])
            (raw / '2025/².json').unlink()
            feed(raw / '2025/-100.json')
            assert compute._build_probable_pitcher_lookup() == {}
            print('signed cache key is accepted, corresponding signed raw filename is ignored')

            status = Path('status.json')
            status.mkdir()
            temp = status.with_suffix('.tmp')
            temp.write_text('{}')
            try:
                os.replace(temp, status)
            except OSError as error:
                print('status directory rename fails:', type(error).__name__)

            feed(raw / '2025/100.json')
            with patch.object(os, 'replace', side_effect=PermissionError('fixture permission denied')):
                print('ordinary regular-cache replacement error propagates:', *feature()[:2])
            assert cache.read_text() == '{}' and list(cache.parent.iterdir()) == [cache]
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    main()
