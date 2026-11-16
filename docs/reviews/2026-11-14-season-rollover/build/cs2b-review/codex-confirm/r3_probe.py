"""Immutable narrow confirmation probes; no network or tracked-tree writes."""
import ast
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
OLD = H / 'c9c1e342'


def emit(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')


def child(version, label):
    sys.path.insert(0, str((OLD if version == 'old' else ROOT) / 'src'))
    sys.path.insert(1, str(ROOT))
    import bts.picks as p
    assert Path(p.__file__).resolve() == (OLD if version == 'old' else ROOT) / 'src/bts/picks.py'
    # Execute the reviewer's original definitions, stopping BEFORE its main program.
    original = H / 'r2_attack.py'
    definitions = original.read_text().split("if len(sys.argv)>1")[0]
    ns = {'__file__': str(original)}
    exec(compile(definitions, str(original), 'exec'), ns)
    directory = H / 'probe-state' / version / label
    directory.mkdir(parents=True, exist_ok=True)
    ns['H'] = directory
    events = ns['events']

    def trap(*args, **kwargs):
        events.append('own value/type hook')
        raise SystemExit('own hook ran')

    class AllMeta(type):
        __getattribute__ = trap
        __eq__ = trap
        __repr__ = trap
        __str__ = trap
        __hash__ = trap

    class AllObject(metaclass=AllMeta):
        __getattribute__ = trap
        __repr__ = trap
        __str__ = trap
        __format__ = trap
        __bool__ = trap
        __eq__ = trap
        __len__ = trap
        __iter__ = trap
        __hash__ = trap
        __fspath__ = trap
        __index__ = trap

    class AllString(str, metaclass=AllMeta):
        __getattribute__ = trap
        __repr__ = trap
        __str__ = trap
        __format__ = trap
        __bool__ = trap
        __eq__ = trap
        __len__ = trap
        __getitem__ = trap
        __hash__ = trap

    class EqNone:
        __eq__ = trap
        __bool__ = trap
        __repr__ = trap
        __str__ = trap

    values = {
        'name-lookup': ns['BadDate'], 'name-format': ns['BadNameDate'],
        'name-IO': ns['SideEffectDate'], 'name-stall': ns['SleepingDate'],
        'all-hooks-object': AllObject,
        'all-hooks-string': lambda: AllString('2027-03-24'),
        'class-as-date': lambda: AllObject, 'eq-none': EqNone,
        'opaque-object': object,
    }
    value = values[label]()
    state = directory / 'streak.json'
    state.write_bytes(b'{"season":2027,"streak":5,"saver_available":false,"applied_dates":[]}\n')
    before = {f.name: f.read_bytes().hex() for f in directory.iterdir() if f.is_file()}
    reads = []
    io_events = []
    hook_calls = []
    active = [False]

    def audit(event, args):
        if active[0] and (event == 'open' or event.startswith(('os.', 'pathlib.'))):
            io_events.append(event)

    def profile(frame, event, arg):
        if active[0] and event == 'call' and frame.f_code.co_filename == str(original):
            if frame.f_code.co_name in ('__getattribute__', '__format__'):
                hook_calls.append(frame.f_code.co_name)

    def observe(name, real):
        def wrapped(*args, **kwargs):
            reads.append(name)
            return real(*args, **kwargs)
        return wrapped

    sys.addaudithook(audit)
    sys.setprofile(profile)
    start = time.monotonic()
    with contextlib.ExitStack() as stack:
        for name in ('load_streak', 'load_saver_available', 'load_applied_dates', 'save_streak'):
            stack.enter_context(patch.object(p, name, observe(name, getattr(p, name))))
        active[0] = True
        try:
            helper = p.season_of_date(value)
            try:
                p.update_streak([True], directory, date=value)
                outcome = {'return': 'unexpected success'}
            except BaseException as exc:
                outcome = {'error': type(exc).__name__, 'message': str(exc)}
        finally:
            active[0] = False
    sys.setprofile(None)
    after = {f.name: f.read_bytes().hex() for f in directory.iterdir() if f.is_file()}
    row = dict(version=version, case=label, helper=helper, outcome=outcome,
               events=events, hook_calls=hook_calls, state_calls=reads,
               io_events=io_events, before=before, after=after,
               seconds=time.monotonic()-start)
    if version == 'current':
        assert helper is None and outcome['error'] == 'ValueError', row
        assert not events and not hook_calls and not reads and not io_events, row
        assert before == after, row
    print(json.dumps(row))


def exact_strings():
    sys.path.insert(0, str(ROOT / 'src'))
    import bts.picks as p
    socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden'))
    socket.socket.connect_ex = socket.socket.connect
    directory = H / 'exact-string-state'
    directory.mkdir(exist_ok=True)
    state = directory / 'streak.json'
    before = b'{"season":2027,"streak":5}\n'
    state.write_bytes(before)
    inputs = ['', 'junk', '2027-02-30', '2027-03-24\x00', '2027-03-24\ud800',
              '\udfff', '\x00', '2027-03-24\n', '٢٠٢٧-٠٣-٢٤',
              '2'*5_000_000, '2027-03-'+'9'*5_000_000, ' '*5_000_000]
    # Every Unicode code point at each date position is sampled, plus all ASCII characters.
    # Deterministic, bounded input corpus, independent of the existing tests.
    for index in range(10):
        for char in list(map(chr, range(128))) + ['\ud800', '\uffff', '٤', '４', '𝟜', '😀']:
            text = '2027-03-24'
            inputs.append(text[:index]+char+text[index+1:])
    rows = []
    for value in inputs:
        parsers = []
        for parse in (dt.date.fromisoformat, lambda text: dt.datetime.strptime(text, '%Y-%m-%d')):
            try:
                parsers.append({'year': parse(value).year})
            except BaseException as exc:
                parsers.append({'error': type(exc).__name__})
        expected = next((result['year'] for result in parsers if 'year' in result), None)
        assert p.season_of_date(value) == expected
        rendered = repr(value)
        assert type(rendered) is str
        if expected is None:
            calls = []
            def forbidden(*a, **k):
                calls.append('state'); raise AssertionError('state accessed')
            with contextlib.ExitStack() as stack:
                for name in ('load_streak', 'load_saver_available', 'load_applied_dates', 'save_streak'):
                    stack.enter_context(patch.object(p, name, forbidden))
                try:
                    p.update_streak([True], directory, date=value)
                except BaseException as exc:
                    assert type(exc) is ValueError
                    assert str(exc) == 'cannot determine season of contest date: '+rendered
                else:
                    raise AssertionError('invalid string was accepted')
            assert not calls and state.read_bytes() == before
        rows.append(dict(input=rendered if len(value) < 100 else '<length '+str(len(value))+'>',
                         parsers=parsers, season=expected))
    # Cancellation is interpreter-wide, not a special date string: reproduce in a child.
    source = "import signal\nfrom bts.picks import season_of_date\ndef stop(*a): raise KeyboardInterrupt('external signal')\nsignal.signal(signal.SIGALRM, stop)\nsignal.setitimer(signal.ITIMER_REAL, .01)\ntry:\n while True: season_of_date('junk')\nexcept KeyboardInterrupt as exc:\n print(type(exc).__name__, str(exc))\nfinally: signal.setitimer(signal.ITIMER_REAL, 0)\n"
    result = subprocess.run([sys.executable, '-c', source], cwd=ROOT, capture_output=True,
                            text=True, timeout=5, env={**os.environ, 'PYTHONPATH': str(ROOT/'src')})
    assert result.returncode == 0 and 'KeyboardInterrupt external signal' in result.stdout
    emit(H/'r3-exact-strings.json', dict(cases=rows, signal=result.stdout.strip(),
                                      python=sys.version, state_unchanged=state.read_bytes()==before))
    return len(rows), sum(row['season'] is None for row in rows)


def main():
    OLD.mkdir(exist_ok=True)
    archive = subprocess.check_output(['git', 'archive', 'c9c1e342', 'src'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(OLD, filter='data')
    labels = ['name-lookup', 'name-format', 'name-IO', 'name-stall',
              'all-hooks-object', 'all-hooks-string', 'class-as-date', 'eq-none', 'opaque-object']
    rows = []
    for version in ('current', 'old'):
        for label in labels if version == 'current' else labels[:4]:
            try:
                result = subprocess.run([sys.executable, __file__, 'child', version, label],
                                        capture_output=True, text=True, timeout=3, cwd=ROOT)
                assert result.returncode == 0, (version, label, result.stdout, result.stderr)
                row = json.loads(result.stdout)
            except subprocess.TimeoutExpired:
                assert version == 'old' and label == 'name-stall'
                row = dict(version=version, case=label, timeout=3,
                           outcome='only this subprocess killed by subprocess.run timeout')
            rows.append(row)
    emit(H/'r3-attacks.json', rows)
    count, rejected = exact_strings()
    # Source equivalence covers the six read sites and the helper, independent of tests.
    equivalent = {}
    for name in ('scheduler.py', 'cli.py', 'strategy.py'):
        old = (OLD/'src/bts'/name).read_bytes()
        new = (ROOT/'src/bts'/name).read_bytes()
        assert old == new
        equivalent[name] = hashlib.sha256(new).hexdigest()
    def helper(path):
        text = path.read_text()
        node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                    and node.name == 'season_of_date')
        return ast.get_source_segment(text, node)
    assert helper(OLD/'src/bts/picks.py') == helper(ROOT/'src/bts/picks.py')
    emit(H/'r3-source-equivalence.json', dict(files=equivalent, season_of_date_identical=True))
    print(json.dumps(dict(current_attacks=9, old_counterexamples=4, exact_strings=count,
                          rejected_strings=rejected, unchanged_read_site_files=3)))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[2], sys.argv[3])
    else:
        main()
