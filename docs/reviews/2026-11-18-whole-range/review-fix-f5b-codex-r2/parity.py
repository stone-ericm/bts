"""Repeat the paired round-one cases with identically pinned raw receipt bytes."""

from itertools import count
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND_ONE = ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'
sys.path.append(str(ROUND_ONE))

import parity_probe
import review_support
from bts.cli import cli
from review_support import World, assert_source, test_entry_intent
from probe_support import no_network


def morning_bytes(world):
    messages = []
    with no_network() as stack:
        stack.enter_context(patch('bts.scheduler.fetch_schedule', lambda _: [{'gamePk': 101}, {'gamePk': 102}]))
        stack.enter_context(patch('bts.dm.send_dm', lambda _, msg: messages.append(msg)))
        result = world.runner.invoke(cli, ['day-outcome', '--date', world.day, '--config', str(world.config),
                                           '--picks-dir', str(world.picks), '--expected-username', world.username,
                                           '--dm-recipient', world.recipient])
    assert result.exit_code in (0, 1) and result.stdout, (result.output, result.exception)
    return result.exit_code, result.stdout, result.stderr, messages


def serialize_bytes(value):
    assert isinstance(value, bytes)
    return {'exact_bytes_hex': value.hex()}


mode, scratch_arg = sys.argv[1:]
scratch = Path(scratch_arg)
review_support.HERE = parity_probe.HERE = HERE
if mode == 'compare':
    for family in ('parity', 'live-health'):
        base, candidate = ((HERE / f'{family}-{version}.json').read_bytes() for version in ('base', 'f5'))
        assert base == candidate, family
        print(f'{family}: {len(json.loads(base))} paired cases, byte-for-byte equality')
else:
    assert_source((scratch / 'base' if mode == 'base' else ROOT) / 'src')
    if mode == 'schedule':
        with TemporaryDirectory(dir=scratch) as folder:
            installer = test_entry_intent.box.__wrapped__(Path(folder))
            with patch.object(test_entry_intent, 'box', SimpleNamespace(__wrapped__=lambda _: installer)):
                parity_probe.schedules(Path(folder), scratch / 'base')
    else:
        # Recreate the same absolute fixture path for each tree so embedded paths match too.
        fixture = scratch / 'paired-fixture'
        fixture.mkdir()
        tokens = count(1)
        try:
            with patch('bts.entry_receipt.uuid.uuid4', lambda: UUID(int=next(tokens))), patch.object(
                    World, 'morning', morning_bytes):
                values = parity_probe.collect(fixture, mode)
            # The original health matrix has a command-line entry point, so load just the definitions.
            from runpy import run_path

            sys.argv = ['live_health_parity.py', mode, scratch_arg]
            run_path(str(ROUND_ONE / 'live_health_parity.py'))
            (HERE / f'parity-{mode}.json').write_text(json.dumps(values, default=serialize_bytes, indent=2) + '\n')
            print(len(values), 'cases with pinned clocks and UUIDs, exact files and CLI output')
        finally:
            shutil.rmtree(fixture)
