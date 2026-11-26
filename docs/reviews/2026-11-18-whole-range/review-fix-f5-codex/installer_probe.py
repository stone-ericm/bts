"""Compare the real base and F5 installers with the repository's crontab shims."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests import test_entry_intent as installer

ARTIFACTS = Path(__file__).resolve().parent
ENTER = installer.LIVE_DM + 'contest_state_required = true\n'
ENTRY = installer.ENTRY_CMD
OLD = '*/15 10-23 * * * cd /old && uv run bts check-pick-entered # BTS-HETZNER'
FOREIGN = f'0 1 * * * echo foreign  \n{OLD.removesuffix(" # BTS-HETZNER")}\n'


def run_box(script, temp, home, config, scenario):
    # Reuse the same absolute HOME for byte parity. Each previously used fixture
    # is moved out of the way before preparing the next real installer run.
    with patch.object(installer, 'SCRIPT', script):
        box = installer.box.__wrapped__(home)
        if config is not None:
            box.config.write_text(config)
        calls = Path(f'{box.cron}.calls')
        if scenario != 'absent':
            box.cron.write_text(FOREIGN + OLD + '\n# ' + OLD + '\n')
        if scenario == 'unreadable':
            Path(f'{box.cron}.unreadable').touch()
        trace = []
        for action in ('show', 'install', 'install', 'remove'):
            result = box.run(action)
            trace.append((result.returncode, result.stdout, result.stderr,
                          box.cron.read_bytes() if box.cron.is_file() else None,
                          calls.read_text() if calls.is_file() else ''))
        home.rename(temp / f'used-{len(list(temp.glob("used-*")))}')
        return trace


def normalized(text):
    if text is None:
        return None
    binary = isinstance(text, bytes)
    decoded = text.decode() if binary else text
    changed = ''.join(line.replace('*/15 * * * * ', '*/15 10-23 * * * ', 1)
                      .replace(f'{ENTRY} --cron ', f'{ENTRY} ', 1)
                      if line.startswith('*/15 * * * * ') and ENTRY in line else line
                      for line in decoded.splitlines(keepends=True))
    return changed.encode() if binary else changed


def compare(base_script):
    generated = {}
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        scope = (('enter', ENTER, 'absent'), ('enter', ENTER, 'marked'),
                 ('enter', ENTER, 'unreadable'), ('research', installer.RESEARCH, 'marked'),
                 ('research', installer.RESEARCH, 'absent'),
                 ('missing-config', None, 'marked'), ('missing-intent', '[scheduler]\n', 'marked'),
                 ('malformed', '[scheduler\n', 'marked'),
                 ('disagreeing', ENTER.replace('"dm"', '"private"'), 'marked'),
                 ('live-gate', installer.LIVE_DM, 'marked'))
        for label, config, scenario in scope:
            traces = [run_box(script, root, root / 'fixture', config, scenario)
                      for script in (base_script, installer.SCRIPT)]
            old, new = traces
            for old_step, new_step in zip(old, new):
                assert old_step == tuple(normalized(part) if isinstance(part, (str, bytes)) else part
                                         for part in new_step), (label, scenario, old_step, new_step)
            if label in ('enter', 'research') and scenario == 'marked':
                for version, trace in zip(('base', 'f5'), traces):
                    shown = trace[0][1].split('Would install:\n', 1)[1].splitlines()
                    emitted = [line for line in trace[1][3].decode().splitlines()
                               if line.endswith('# BTS-HETZNER')]
                    assert shown == emitted
                    assert trace[1][3] == trace[2][3]
                    assert trace[3][3] == FOREIGN.encode()
                    lines = [line for line in emitted if ENTRY in line]
                    assert len(lines) == (label == 'enter')
                    generated[f'{version}_{label}'] = emitted
            if label == 'enter' and scenario == 'unreadable':
                assert [step[0] for step in new] == [0, 1, 1, 1]
                assert '(none)' in new[0][1]
            if label not in ('enter', 'research'):
                assert [(step[0], step[4]) for step in new[:3]] == [(1, '')] * 3
            print(label, scenario, [step[0] for step in new])
    (ARTIFACTS / 'generated.json').write_text(json.dumps(generated, indent=2) + '\n')


if __name__ == '__main__':
    compare(Path(sys.argv[1]).resolve())
