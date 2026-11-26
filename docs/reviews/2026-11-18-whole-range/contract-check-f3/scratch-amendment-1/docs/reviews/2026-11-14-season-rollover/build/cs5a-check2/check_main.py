"""Run the first review's offline probes and inspect the legacy posting saves."""

from contextlib import ExitStack
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / 'docs/reviews/2026-11-14-season-rollover/build/cs5a-check'),
]
import reproduce
import paths
from reproduce import (
    DAY, NOW, PREDICTIONS, STATUS, CliRunner, TemporaryDirectory, alert, cli,
    contest_state, dm, json, network_attempts, orchestrator, patch, picks,
    posting, predict, scheduler,
)


PROVENANCE = {
    'model_git_sha', 'model_pickle_sha256', 'policy_npz_sha256',
    'tail_policy_sha256', 'feature_env_schema_version', 'feature_env',
    'feature_env_hash',
}
MARKERS = {'bluesky_posted', 'bluesky_uri'}


def required_inputs_and_alert_failure(root):
    picks_dir = root / 'picks'
    account = picks_dir / 'account_state'
    account.mkdir(parents=True)
    health = root / 'health_state' / 'health_dm_delivery_status.json'
    health.parent.mkdir()

    for kind in ('model', 'legacy_manual'):
        if kind == 'legacy_manual':
            paths.write_json(account / 'contest_streak.manual.json', {
                'streak': 7, 'source': 'mlb_bts_profile',
                'override_expires_at': f'{DAY}T23:59:00-04:00',
            })
        optional = contest_state.load_decision_streak_state(picks_dir, now=NOW)
        assert optional.streak == (0 if kind == 'model' else 7)
        try:
            contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=NOW)
        except contest_state.ContestStateError:
            print(f'{kind}: optional read succeeds and required read refuses')
        else:
            raise AssertionError(f'required read accepted {kind}')

    health.write_text('[]')
    config = {'orchestrator': {'picks_dir': str(picks_dir)}, 'bluesky': {'dm_recipient': 'local.invalid'}}
    with patch.object(alert, 'send_dm') as send:
        try:
            scheduler._alert_contest_state_failure(config, contest_state.ContestStateError('sample refusal'))
        except AttributeError:
            send.assert_not_called()
            print('non-object health status JSON raises from the refusal alert before transport')
        else:
            raise AssertionError('expected the health alert status reader to raise')


def difference(before, after):
    return {key: after.get(key) for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)}


def unlocked_reuse(root, command, case):
    folder = root / f'{command}-{case}'
    account = folder / 'account_state'
    models = folder / 'models'
    path = folder / f'{DAY}.json'
    evolution = folder / f'lineup_evolution_{DAY}.jsonl'
    paths.automatic(account)
    paths.dated(account)
    cfg = folder / 'worker.toml'
    cfg.write_text(f'tiers = []\n[orchestrator]\npicks_dir = "{folder}"\n'
                   f'models_dir = "{models}"\n[bluesky]\ndm_recipient = "local.invalid"\n'
                   '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\n')

    clock = NOW.replace(hour=15) if case == 'closed' else NOW
    status = STATUS
    loader = contest_state.load_decision_streak_state
    required_reads = []
    posts = []
    dms = []

    def snapshot():
        return json.loads(path.read_bytes()), [json.loads(line) for line in evolution.read_text().splitlines()]

    def read(*args, **kwargs):
        if kwargs.get('require_contest_state'):
            required_reads.append(snapshot())
            if (case == 'read-error' and len(required_reads) == 2) or case == 'first-read-error':
                (account / 'contest_streak.manual.json').rename(account / 'manual.set-aside')
                paths.automatic(account, revoked=True)
        return loader(*args, **kwargs)

    def post(text):
        posts.append((text, snapshot()))
        if case == 'transport-error':
            raise RuntimeError('captured transport failure')
        return 'captured-public-post'

    with ExitStack() as stack:
        for module, name, replacement in (
            (contest_state, 'datetime', reproduce.PinnedDateTime),
            (contest_state, 'load_decision_streak_state', read),
            (picks, 'get_game_statuses_detailed', lambda _: status),
            (picks, 'get_game_statuses', lambda _: dict.fromkeys(STATUS, 'P')),
            (posting, '_now_et', lambda: clock),
            (posting, 'post_to_bluesky', post),
            (predict, 'run_pipeline', lambda *a, **kw: PREDICTIONS),
            (predict, 'load_blend', lambda *_: None),
            (orchestrator, 'run_cascade', lambda *_: (PREDICTIONS, 'offline')),
            (scheduler, 'fetch_schedule', lambda *_: []),
            (dm, 'send_dm', lambda *args: dms.append(args) or 'captured-dm'),
            (alert, 'send_dm', lambda *args: dms.append(args) or 'captured-alert'),
        ):
            stack.enter_context(patch.object(module, name, replacement))

        (account / 'contest_streak.manual.json').rename(account / 'manual.set-aside')
        local_flags = ['--picks-dir', str(folder), '--models-dir', str(models)]
        before_preview = CliRunner().invoke(cli.cli, ['preview', '--date', DAY, *local_flags],
                                            catch_exceptions=False)
        assert before_preview.exit_code == 0, before_preview.output
        (account / 'manual.set-aside').rename(account / 'contest_streak.manual.json')
        status = dict.fromkeys(STATUS, {'abstract': 'P', 'detailed': 'Warmup', 'code': 'W'})
        models.mkdir()
        (models / f'blend_{DAY}.pkl').write_bytes(b'current blend bytes')
        prior, initial_lines = snapshot()
        assert prior['double_down'] is not None
        prior.update(dict.fromkeys(PROVENANCE, 'prior version'))
        prior['feature_env'] = {'revision': 'prior version'}
        path.write_text(json.dumps(prior, indent=4))
        before_bytes = path.read_bytes()
        flags = local_flags if command == 'run' else ['--config', str(cfg)]
        result = CliRunner().invoke(cli.cli, [command, '--date', DAY, *flags])
        after, lines = snapshot()
        assert required_reads[0][0] == prior
        if case == 'first-read-error':
            assert result.exit_code == 1 and len(required_reads) == 1 and not posts
            assert path.read_bytes() == before_bytes and lines == initial_lines
            assert len(dms) == (command == 'orchestrate')
        else:
            assert len(required_reads) == 2 and not dms
            presend, presend_lines = required_reads[1]
            assert difference(prior, presend).keys() == PROVENANCE
            assert path.read_bytes() != before_bytes and len(presend_lines) == len(initial_lines) + 1
            if case in {'posts', 'transport-error'}:
                assert len(posts) == 1 and 'Streak: 56' in posts[0][0] and posts[0][1] == required_reads[1]
                assert result.exit_code == 0 and result.exception is None
                expected = MARKERS if case == 'posts' else set()
            else:
                assert not posts
                expected = set()
                if case == 'read-error':
                    assert result.exit_code == 1
                    assert isinstance(result.exception, contest_state.ContestStateError if command == 'orchestrate'
                                      else SystemExit)
                else:
                    assert result.exit_code == 0
            assert difference(presend, after).keys() == expected
            assert len(lines) == len(presend_lines) + bool(expected)
        for line in lines[len(initial_lines):]:
            assert {k: v for k, v in line.items() if k != 'captured_at'} == {
                k: v for k, v in initial_lines[-1].items() if k != 'captured_at'
            }
        print(json.dumps({
            'command': command, 'case': case, 'exit': result.exit_code,
            'exception': type(result.exception).__name__ if result.exception else None,
            'new_evolution_lines': len(lines) - len(initial_lines),
            'changed': difference(prior, after), 'posts': len(posts), 'dms': len(dms),
            'stderr': result.stderr.splitlines(),
        }, sort_keys=True))


if __name__ == '__main__':
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        reproduce.main(root / 'reproduction')
        paths.run(root / 'paths')
        required_inputs_and_alert_failure(root / 'alerts')
        for command in ('run', 'orchestrate'):
            for case in ('posts', 'closed', 'transport-error', 'read-error', 'first-read-error'):
                unlocked_reuse(root, command, case)
    assert not network_attempts, network_attempts
