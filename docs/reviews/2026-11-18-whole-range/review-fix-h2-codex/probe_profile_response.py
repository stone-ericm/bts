import json
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from bts import cli, contest_state
from bts.leaderboard import auth
from support import DAY, at, counts, install, invoke, live, walk


if __name__ == '__main__':
    cap = getattr(contest_state, 'MAX_SEASON_STREAK', 57)
    with TemporaryDirectory() as tmp, walk.offline(at()):
        directory = Path(tmp) / 'picks'
        install(directory, 'automatic', 12, 12)
        state_dir = directory / 'account_state'
        paths = [state_dir / name for name in ('contest_streak.json', 'contest_ledger.jsonl', 'saver_state.json')]
        cases = [(dict(counts(58, 58), predictions=[{'roundId': 1, 'result': 'hit'}]), 0 if cap == 400 else 2),
                 (dict(counts(57, 401), predictions=[{'roundId': 1, 'result': 'hit'}]), 2),
                 (dict(counts(58, 58), predictions={'roundId': 1}), 2),
                 (dict(counts(58, 58), predictions=[{'roundId': [], 'result': 'hit'}]), 2)]
        session = SimpleNamespace(user_id=walk.UID, username=walk.ACCOUNT, xsid='synthetic')
        for payload, expected in cases:
            before = [path.read_bytes() for path in paths]
            request = httpx.Request('GET', 'https://example.invalid/profile')
            response = httpx.Response(200, request=request, json={'success': payload})
            with patch.object(auth, 'load_session_cookies', return_value={}), \
                    patch.object(auth, 'extract_uid', return_value=walk.UID), \
                    patch.object(auth, 'fetch_login_session', return_value=session), \
                    patch.object(cli, '_fetch_rounds', return_value={1: DAY - timedelta(days=1)}), \
                    patch.object(httpx, 'get', return_value=response) as get:
                result = invoke('fetch-contest-streak', '--picks-dir', directory, '--expected-username', walk.ACCOUNT)
            assert get.call_count == 1 and result[0] == expected, (result, get.call_args_list)
            if expected:
                assert [path.read_bytes() for path in paths] == before
            else:
                assert live(directory).streak == 58
                assert json.loads(paths[1].read_bytes().splitlines()[-1])['active_streak'] == 58
            print(cap, expected, result[1].replace(str(directory), '<picks>'))
