"""Exercise the real day-outcome and DM command with the repo's offline chat limit."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from unittest.mock import patch

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.codex-review/scratch-f4/src'), str(ROOT)]

from bts import day_outcome, dm, scheduler
from bts.cli import cli
from tests.test_day_outcome import DATE, _bound_receipt, _box, _publish_receipt, _write_decision

spec = importlib.util.spec_from_file_location('base_day_outcome', ROOT / 'src/bts/day_outcome.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
scratch_gather = day_outcome.gather

attempts = []


def request(req, **kwargs):
    route = req.full_url.rsplit('/', 1)[-1]
    if route == 'chat.bsky.convo.sendMessage':
        text = json.loads(req.data)['message']['text']
        attempts.append(text)
        assert kwargs['idempotent'] is False
        if len(text) > 1000:
            raise HTTPError(req.full_url, 400, 'offline chat maxGraphemes=1000', {}, None)
        data = {'id': 'offline-confirmed'}
    elif route == 'com.atproto.server.createSession':
        data = {'accessJwt': 'offline-token'}
    elif route.startswith('chat.bsky.convo.getConvoForMembers'):
        data = {'convo': {'id': 'offline-convo'}}
    else:
        raise AssertionError(req.full_url)
    return BytesIO(json.dumps(data).encode())


with TemporaryDirectory() as tmp:
    picks, config = _box(Path(tmp))
    # A restored but parseable decision can contain an arbitrary pick display string.
    _write_decision(picks, primary={'batter_name': 'A' * 700})
    _publish_receipt(picks, _bound_receipt(picks, confirmed=False, reason='no_pick'))
    (picks / 'streak.json').write_text('[]')
    argv = ['day-outcome', '--date', DATE, '--config', str(config), '--picks-dir', str(picks),
            '--expected-username', 'stonehengee', '--dm-recipient', 'did:plc:offline']
    for name, gather in [('base', base.gather), ('scratch', scratch_gather)]:
        with patch.object(day_outcome, 'gather', gather), \
             patch.object(scheduler, 'fetch_schedule', lambda _: [{'gamePk': 900}]), \
             patch.object(dm, 'get_dm_password', lambda: 'offline'), patch.object(dm, 'retry_urlopen', request):
            result = CliRunner().invoke(cli, argv)
        assert result.exit_code == 1 and json.loads(result.stdout)['state'] == 'not_entered'
        assert attempts[-1].startswith(f'BTS day check {DATE}: NOT ENTERED.')
        print(name, 'chars', len(attempts[-1]), 'stderr', result.stderr.strip())
        assert ('DM failed: HTTPError' in result.stderr) is (name == 'scratch')
    assert len(attempts) == 2
