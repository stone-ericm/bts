"""Check the adopted sentences and render actual help and early deferral."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'))

from bts.cli import cli
from review_support import ROOT, World, assert_source


assert_source(ROOT / 'src')
architecture = (ROOT / 'ARCHITECTURE.md').read_text()
guide = (ROOT / 'docs/ops/pick-entry-receipt-v1.md').read_text()
assert ('Early confirmation is terminal while the saved leg identities and first-pitch times match the early '
        'ledger. The live ledger retains its existing confirmation and retry rules.') in architecture
assert ('A pre-10:00 ET cron check that passes the early gate writes any resulting marker to '
        '`data/health_state/pick_entry_check.early/<ET date>.json`.') in guide
assert ('The morning report uses qualifying early observations only with a readable final pick that has the same '
        'legs and no nominal original-hours slot.') in guide
assert ('A live marker write whose receipt is published in the milliseconds after an early observation '
        'is counted as following it.') in guide

result = CliRunner().invoke(cli, ['check-pick-entered', '--help'])
assert result.exit_code == 0
help_text = ' '.join(result.stdout.split())
for text in (
    'Each watchdog CLI invocation attempts to publish one entry receipt and its seal.',
    'The installed watchdog launches every 15 minutes ET. It prints the outcome when it skips a check. '
    'An eligible early check uses the dated early ledger. Manual and original-hours checks use the live ledger. '
    'While eligible, an unresolved entry is checked again, with throttled reminders at 30 and 15 minutes to cutoff '
    'and an all-clear when a warned entry is confirmed.',
    'Before 10:00 ET, defer an in-window check of a committed pick if a nominal 10:00–23:45 ET quarter-hour '
    'is also inside that window.',
):
    assert text in help_text, text
print(result.stdout)
with TemporaryDirectory() as folder:
    w = World(Path(folder))
    w.commit_at('10:05:01', at='09:40')
    deferred = w.cron('09:45')
    assert deferred['exit'] == 0 and not deferred['calls'] and not deferred['sent']
    assert 'early_cron_run' not in deferred['receipt']
    assert deferred['stdout'] == ('check-pick-entered: early check deferred because a nominal 10:00–23:45 ET '
                                 'quarter-hour is inside this pick\'s entry window\n')
    print(deferred['stdout'].strip())
    original = w.cron('10:00')
    assert original['exit'] == 1 and original['calls'] and len(original['sent']) == 1
    print(original['stdout'].strip(), original['sent'])
