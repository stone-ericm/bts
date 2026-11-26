import json
import logging
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from bts import dm
from bts.features import probable_pitcher_cache as producer
from bts.health import alert, runner

from review_evidence import Clock, build, seed
from review_health import DAY, RECEIPT, ROOT, SOURCE, STATUS, Wall, run_eod


def main():
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=ROOT, prefix='retention-run-') as path:
        os.chdir(path)
        try:
            with patch.object(producer, 'datetime', Clock):
                seed(b'{"100":null}', feeds=False)
                build()
                Clock.at = datetime(2027, 1, 1, 5, 1, tzinfo=timezone.utc)
                build()
            dated = sorted(file.name for file in STATUS.glob('*.json'))
            assert len(dated) == 2 and all('invalid_entries' in name for name in dated)
            requests = []
            safe_run = runner._safe_run

            def api(request, **kw):
                if request.full_url.endswith('createSession'):
                    return BytesIO(b'{"accessJwt":"synthetic"}')
                if 'getConvoForMembers' in request.full_url:
                    return BytesIO(b'{"convo":{"id":"synthetic"}}')
                assert request.full_url.endswith('chat.bsky.convo.sendMessage')
                message = json.loads(request.data)['message']['text']
                assert message.isascii()
                requests.append(len(message))
                if len(message) > 1000:
                    raise HTTPError(request.full_url, 400, 'synthetic chat maxGraphemes=1000', {}, None)
                return BytesIO(b'{"id":"synthetic"}')

            with (patch.object(runner, '_safe_run', lambda src, cb: safe_run(src, cb) if src == SOURCE else []),
                  patch.object(runner.realized_calibration, '_current_deploy_iso', return_value=None),
                  patch.object(runner, 'datetime', Wall), patch.object(alert, 'datetime', Wall),
                  patch.object(dm, 'get_dm_password', return_value='synthetic'), patch.object(dm, 'retry_urlopen', api)):
                assert len(run_eod(DAY)) == len(run_eod(DAY)) == 1
                assert len(requests) == 1 and json.loads(RECEIPT.read_bytes())['status'] == 'sent'
                assert len(run_eod(DAY + timedelta(days=1))) == 2
                result = json.loads(RECEIPT.read_bytes())
                assert result['status'] == 'failed' and result['critical_count'] == 0
            print('actual producer observations for unchanged bad cache:', dated)
            print('processed D then D+1 received and failed, ASCII body lengths:', requests)
        finally:
            os.chdir(ROOT)


if __name__ == '__main__':
    main()
