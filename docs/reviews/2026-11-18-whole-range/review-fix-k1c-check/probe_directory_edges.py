"""Check B8a directory setup and the inherited status examples."""

from pathlib import Path
import sys

from replay_space import B8A, FILING, MANUAL, REQUESTS, ROOT, SHELLS, STATUS
from replay_space import correct_filing, literal_assignment, replay, run_pair


def filing_outside_service(run, cwd, live, setup, wrong):
    code, output = run_pair(run, setup)
    assert code == 0 and 'filing exit=0' in output and 'live decision now: accepted;' in output, output
    assert (wrong / MANUAL).exists() and (wrong / REQUESTS).exists()
    assert not (live / MANUAL).exists()
    check, live_status = run(f'{literal_assignment(live)}\n{STATUS}')
    assert check == 0 and 'live decision now: refused;' in live_status, live_status
    return f'CLI exits 0/0, status accepts {wrong.relative_to(cwd) if wrong.is_relative_to(cwd) else wrong.name}, live status still refused'


def unset(run, cwd, live):
    guarded = FILING.replace('"$DIR"', '"${DIR:?Set DIR in this shell}"')
    code, output = run(guarded)
    assert code != 0 and 'Set DIR in this shell' in output and not (cwd / MANUAL).exists(), output
    return filing_outside_service(run, cwd, live, '', cwd)


def empty(run, cwd, live):
    return filing_outside_service(run, cwd, live, "DIR=''", cwd)


def other_shell(run, cwd, live):
    code, output = run(f'{literal_assignment(live)}\nprintf "%s\\n" "$DIR"')
    assert code == 0 and output.strip() == str(live), output
    return filing_outside_service(run, cwd, live, '', cwd)


def wrong_base(run, cwd, live):
    wrong = cwd / 'data/custom-picks'
    return filing_outside_service(run, cwd, live, literal_assignment(wrong), wrong)


def expanded_dollar(run, cwd, live):
    setup = f'unset wrk1c_suffix\nDIR="{live}"'
    return filing_outside_service(run, cwd, live, setup, live.with_name('cash'))


def read_literal(run, cwd, live):
    code, output = run(f'IFS= read -r DIR\n{FILING}\nprintf "filing exit=%s\\n" "$?"\n{STATUS}',
                       f'{live}\n')
    assert code == 0 and 'filing exit=0' in output and 'live decision now: accepted;' in output, output
    assert not (cwd / MANUAL).exists() and not (cwd / 'data/picks' / MANUAL).exists()
    return 'literal read preserves spaces, apostrophe, double quote and dollar sign, CLI exits 0/0, status accepted'


def old_status(run, cwd, live):
    from tests.test_pick_chosen_on_cs5a import set_automatic

    code, output = run(f'{literal_assignment(live)}\n{STATUS}')
    assert code == 0 and 'live decision now: refused;' in output, output
    old = 'bts season-contract status --picks-dir DIR'
    assert f'`{old}`' in B8A
    code, output = run(old.replace('DIR', str(live)))
    assert code == 2 and 'unexpected extra argument (picks)' in output, output

    # Another checkout's default directory is on automatic. Bare status sees it.
    default = cwd / 'data/picks'
    set_automatic(default, streak=12)
    before = {path: path.read_bytes() for path in default.rglob('*.json')}
    code, output = run('bts season-contract status')
    assert code == 0 and 'automatic path alone: accepted (streak 12,' in output, output
    assert {path: path.read_bytes() for path in default.rglob('*.json')} == before
    assert not (live / MANUAL).exists()
    correct_filing(run, cwd, live)
    return 'old unquoted status exits 2, bare status gives an automatic all-clear for other directory, quoted live filing succeeds'


if __name__ == '__main__':
    unit = (ROOT / 'scripts/systemd/bts-scheduler.service').read_text().splitlines()
    box_workdir = Path(next(line.partition('=')[2] for line in unit if line.startswith('WorkingDirectory=')))
    assert box_workdir.is_absolute()
    shell = SHELLS[int(sys.argv[1])]
    space = Path('data/live picks')
    specials = Path("data/Owner's \"cash\" $picks")
    default = Path('data/picks')
    replay(shell, 'unset DIR', space, unset, expected_delivery=False)
    replay(shell, 'empty DIR', space, empty, expected_delivery=False)
    replay(shell, 'DIR set in a different shell', space, other_shell, expected_delivery=False)
    replay(shell, 'relative config resolved against operator cwd', Path('data/custom-picks'),
           wrong_base, expected_delivery=False)
    replay(shell, 'expanded dollar sign during assignment', Path('data/cash$wrk1c_suffix'),
           expanded_dollar, expected_delivery=False)
    replay(shell, 'literal assignment with quote and dollar sign', specials, correct_filing)
    replay(shell, 'literal read with quote and dollar sign', specials, read_literal)
    replay(shell, 'default data/picks', default, correct_filing)
    replay(shell, f'box path {box_workdir / default} under a temporary root', default, correct_filing,
           service_suffix=box_workdir.relative_to('/'))
    replay(shell, 'older B8a status commands', space, old_status)
