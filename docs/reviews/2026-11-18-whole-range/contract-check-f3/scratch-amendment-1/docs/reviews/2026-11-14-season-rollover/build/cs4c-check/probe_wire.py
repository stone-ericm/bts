"""Exercise pick-file readers, exact-byte consumers, and timestamp edge cases."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe_main  # Denies DNS and socket operations, and supplies offline fixtures.

from atproto_client.models.chat.bsky.convo.defs import MessageInput
from bts import contest_state, picks, web
from bts.entry_receipt import EntryReceipt
from bts.experiment.artifacts import load_production_pick_snapshot
from bts.state.export import _pick_to_historical
from bts.state.regenerate import _hist_to_pick_file
from scripts import live_forward_capture_once as capture
from scripts import live_forward_provenance_inventory as inventory
from scripts.audit.season_ledger.sources.pick_files import parse_pick_file
from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, daily_pick, set_automatic, set_dated


def reader_cases(root):
    pick_dir = root / 'readers'
    pick_dir.mkdir()
    path = pick_dir / f'{DAY}.json'
    shadow = pick_dir / f'{DAY}.shadow.json'
    base = asdict(daily_pick())
    record = {'schema': 'bts_pick_automatic_reading_v1', 'recorded_at': '2027-07-14T22:00:00Z',
              'age_hours': 24.58, 'stale': True, 'reason': 'old'}
    projections, hashes, identities, historical = [], [], [], []
    for doc in [base, dict(base, automatic_reading=None), dict(base, automatic_reading=record),
                dict(base, automatic_reading=dict(record, age_hours=25.0))]:
        data = json.dumps(doc, indent=2).encode()
        path.write_bytes(data)
        shadow.write_bytes(data)
        loaded, held = picks.load_pick_bytes(DAY, pick_dir)
        assert held == data and picks.load_pick(DAY, pick_dir) == loaded == picks.load_shadow_pick(DAY, pick_dir)
        receipt = EntryReceipt(et_date=DAY, started_at=NOW, clock=lambda: NOW,
                               expected_username=None, picks_dir=pick_dir)
        receipt.selection(loaded, held, None, None)
        row, = parse_pick_file(path.name, data).rows
        assert receipt.record()['selection']['pick_file_sha256'] == row['content_sha256'] == sha256(data).hexdigest()
        assert json.loads(row['record_raw_json']) == doc
        with patch.object(web, 'PICKS_DIR', pick_dir):
            assert web.load_all_picks() == [doc]
        snapshot = load_production_pick_snapshot(path, expected_date=DAY)
        assert snapshot['source_sha256'] == sha256(data).hexdigest()
        projections.append([capture.decision_snapshot(doc), inventory._decision_snapshot(doc)])
        historical.append(_pick_to_historical(doc))
        hashes.append(sha256(data).hexdigest())
        identities.append(row['obs_id'])
    assert len(set(hashes)) == len(set(identities)) == 4
    assert projections[0] != projections[1] and projections[2] != projections[3]
    assert all(v == historical[0] for v in historical)
    assert 'automatic_reading' not in _hist_to_pick_file(historical[0])
    print('both live-forward comparators currently differ on missing/null and change of value')
    print('dashboard/ledger retain raw object, receipt/ledger/snapshot hashes differ, export is stable')
    # The asdict encoder adds more than one key to a genuinely historical file.
    path.write_text(json.dumps({k: v for k, v in base.items() if k not in {'chosen_on', 'policy_decision'}}))
    before = json.loads(path.read_text())
    picks.save_pick(picks.load_pick(DAY, pick_dir), pick_dir)
    assert set(json.loads(path.read_text())) - set(before) == {'chosen_on', 'policy_decision'}


def timestamp_cases(root):
    directory = root / 'times'
    set_automatic(directory)
    path = directory / 'account_state/contest_streak.json'
    data = json.loads(path.read_text())
    for raw in [None, 7, '', '2027-07-14T18:30:00', '2027-07-14T18:30:00Z',
                '2027-07-14\n18:30:00Z', '2027-07-14\ud80018:30:00Z',
                '2027-07-14T18:30:00.' + '0' * 12000 + 'Z', '0001-01-01T00:00:00+01:00']:
        path.write_text(json.dumps(dict(data, recorded_at=raw)))
        obs = contest_state.load_contest_streak_state(directory, now=NOW, require_dated_auto=True)
        assert obs.recorded_at == (raw if isinstance(raw, str) else None)
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw)
                print('ISO', repr(raw[:40]), len(raw), parsed, 'utc', parsed.astimezone(timezone.utc))
            except (ValueError, OverflowError) as exc:
                print('ISO exception', repr(raw[:40]), type(exc).__name__)
            if len(raw) > 12000:
                try:
                    MessageInput(text='warning ' + raw)
                except ValueError:
                    print('long syntactically valid timestamp exceeds installed client text limit')
                else:
                    raise AssertionError('expected transport model to reject the length')
    set_dated(directory)
    dated = contest_state.load_contest_streak_state(directory, now=NOW, require_dated_auto=True)
    assert dated.recorded_at == NOW.isoformat()
    manual = directory / 'account_state/contest_streak.manual.json'
    manual.unlink()
    account_file(directory, 'contest_streak.manual.json', {'streak': 7, 'recorded_at': '',
                 'override_expires_at': (NOW + timedelta(hours=1)).isoformat()})
    assert contest_state.load_contest_streak_state(directory, now=NOW).recorded_at == ''
    for diff in [-1, 0, 1]:
        elapsed = timedelta(hours=14, microseconds=diff)
        assert round(elapsed.total_seconds() / 3600.0, 2) == 14.0
        print('boundary microseconds', diff, 'is stale:', elapsed > timedelta(hours=14))
    broken = dict(asdict(daily_pick()), automatic_reading={'timestamp': '\ud800'})
    row, = parse_pick_file(f'{DAY}.json', json.dumps(broken).encode()).rows
    try:
        row['record_raw_json'].encode('utf-8')
    except UnicodeEncodeError:
        print('decoded surrogate in raw season-ledger JSON cannot encode as UTF-8')
    else:
        raise AssertionError('surrogate was not carried through the raw ledger projection')


if __name__ == '__main__':
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        reader_cases(Path(tmp))
        timestamp_cases(Path(tmp))
