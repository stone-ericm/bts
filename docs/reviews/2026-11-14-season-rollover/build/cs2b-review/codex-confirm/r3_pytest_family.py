"""Immutable single wrapped family: current tests, old-source regression proof, race trace."""
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
RACE_DAY = '2027-03-26'


def run_child(mode):
    source = (H/'c9c1e342' if mode == 'old' else ROOT)/'src'
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(ROOT))
    socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden'))
    socket.socket.connect_ex = socket.socket.connect
    import bts.picks as p
    import pytest
    assert Path(p.__file__).resolve() == source/'bts/picks.py'
    events = []

    def trace(frame, event, arg):
        name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        if not (filename.endswith(('/bts/picks.py', '/bts/scheduler.py', '/bts/cli.py',
                                  '/test_streak_season_cs2b_review.py'))):
            return None
        day = frame.f_locals.get('date', frame.f_locals.get('day'))
        if type(day) is str and day == RACE_DAY:
            selected = ((name == 'resolve_with_peer_interleave' and event in ('call', 'return')) or
                        (name == 'update_streak' and event in ('call', 'return')) or
                        (name == 'run_result_polling' and event == 'line' and
                         frame.f_lineno in (2590, 2599, 2605, 2609)))
            if selected:
                directory = frame.f_locals.get('picks_dir')
                stack = []
                parent = frame.f_back
                while parent:
                    stack.append(parent.f_code.co_name)
                    parent = parent.f_back
                state = json.loads((directory/'streak.json').read_text()) if directory else None
                pickfile = directory/(RACE_DAY+'.json') if directory else None
                result = json.loads(pickfile.read_text())['result'] if pickfile else None
                events.append(dict(function=name, event=event, line=frame.f_lineno,
                                   stack=stack, streak=state['streak'] if state else None,
                                   result=result, returned=arg if name=='update_streak' else None))
        return trace

    class RacePlugin:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            if item.name == 'test_full_rollover_lifecycle_void_hit_double_race_crash_and_reconcile':
                sys.settrace(trace)
                try:
                    yield
                finally:
                    sys.settrace(None)
                    (H/'r3-race-trace.json').write_text(json.dumps(events, indent=2)+'\n')
            else:
                yield

    args = ['-p', 'no:cacheprovider', '-q', '-o', 'addopts=',
            '--basetemp='+str(H/('pytest-tmp-'+mode)),
            '--junitxml='+str(H/('r3-'+mode+'.xml'))]
    if mode == 'old':
        args += ['tests/test_streak_season_cs2.py', '-k', 'type-name-trap']
    else:
        args += ['tests/test_streak_season_cs2.py', 'tests/test_streak_season_cs2_review.py',
                 'tests/test_streak_season_cs2b_review.py']
    raise SystemExit(pytest.main(args, plugins=[RacePlugin()] if mode == 'current' else []))


def main():
    outcomes = {}
    for mode in ('current', 'old'):
        result = subprocess.run([sys.executable, __file__, mode], cwd=ROOT)
        outcomes[mode] = result.returncode
        assert result.returncode == (1 if mode == 'old' else 0), outcomes
    tree = ET.parse(H/'r3-old.xml')
    failures = tree.findall('.//failure')
    assert len(failures) == 1
    assert 'RuntimeError' in failures[0].text and 'metaclass name hook' in failures[0].text
    events = json.loads((H/'r3-race-trace.json').read_text())
    resolves = [row for row in events if row['function']=='resolve_with_peer_interleave' and row['event']=='call']
    updates = [row for row in events if row['function']=='update_streak' and row['event']=='call']
    peer_reads = [row for row in events if row['function']=='run_result_polling' and row['line']==2605]
    assert len(resolves) == 2, events
    assert len(updates) == 1 and 'resolve_with_peer_interleave' in updates[0]['stack'], events
    assert updates[0]['streak'] == 1 and updates[0]['result'] is None, events
    assert len(peer_reads) == 1 and peer_reads[0]['streak'] == 3 and peer_reads[0]['result'] == 'hit', events
    assert not any(row['function']=='run_result_polling' and row['line']==2609 for row in events), events
    # Existing assertion expressions are all preserved, apart from an added failure diagnostic.
    old_text = subprocess.check_output(['git', 'show', 'c9c1e342:tests/test_streak_season_cs2b_review.py'], cwd=ROOT, text=True)
    new_text = (ROOT/'tests/test_streak_season_cs2b_review.py').read_text()
    def assertions(text):
        return sorted(ast.dump(node.test) for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Assert))
    assert assertions(old_text) == assertions(new_text)
    summary = dict(outcomes=outcomes, race_resolver_entries=len(resolves), race_scoring_updates=len(updates),
                   polling_observed_peer_streak=peer_reads[0]['streak'], assertions_unchanged=True)
    (H/'r3-pytest-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1:
        run_child(sys.argv[1])
    else:
        main()
