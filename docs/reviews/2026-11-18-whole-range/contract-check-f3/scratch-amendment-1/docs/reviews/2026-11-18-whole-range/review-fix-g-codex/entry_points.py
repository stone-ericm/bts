"""Follow the guarded action through the CLI, scheduler, shadow and dashboard."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import json

from click.testing import CliRunner

from bts import cli, dm, orchestrator, picks, posting, scheduler, strategy, web
from bts.model import predict
from commit_paths import DAY, GUARD, NOW, choose, data, record, setup


def assert_saved(root, shadow=False):
    name = f'{DAY}.shadow.json' if shadow else f'{DAY}.json'
    body = json.loads((root / name).read_bytes())
    assert body['pick']['batter_id'] == 100 and body['double_down'] is None
    assert body['policy_decision']['action'] == 'single' and body['policy_decision']['guard'] == GUARD
    if not shadow:
        assert body['chosen_on']['kind'] == 'dated'
    return body


def invoke(*args):
    result = CliRunner().invoke(cli.cli, [str(val) for val in args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.output


def cli_and_shadow():
    for name in ('run', 'preview', 'orchestrate', 'shadow', 'run_and_pick', 'check'):
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            frame = data(.96)
            cfg, statuses = setup(stack, root, frame)
            send = stack.enter_context(patch.object(posting, 'post_to_bluesky', Mock(return_value='captured-uri')))
            stack.enter_context(patch.object(posting, 'should_post_now', lambda *_: True))
            stack.enter_context(patch.object(predict, 'run_pipeline', lambda *a, **k: frame))
            if name in ('run', 'preview'):
                code, out = invoke(name, '--date', DAY, '--picks-dir', root, '--models-dir', root / 'models')
                assert code == 0 and 'Fixture batter 100' in out, (code, out)
                assert_saved(root)
            elif name == 'orchestrate':
                toml = root / 'config.toml'
                toml.write_text(f'[orchestrator]\npicks_dir={json.dumps(str(root))}\n'
                                '[scheduler]\npick_delivery="dm"\nentry_intent="enter"\ncontest_state_required=true\n'
                                '[health_checks]\ncontest_state_expected=true\n'
                                '[bluesky]\ndm_recipient="capture"\n[[tiers]]\nname="probe"\n')
                code, out = invoke(name, '--date', DAY, '--config', toml)
                assert code == 0 and 'Fixture batter 100' in out, (code, out)
                assert_saved(root)
            elif name == 'shadow':
                (root / 'streak.json').write_text(json.dumps({'season': 2027, 'streak': 50, 'saver_available': False}))
                stack.enter_context(patch.object(scheduler, 'predict_local_shadow', lambda *a, **k: frame))
                scheduler._run_shadow_prediction(cfg, DAY, 'Fixture batter 100')
                assert_saved(root, shadow=True)
            elif name == 'run_and_pick':
                day, sel = choose(cfg)
                assert sel.action == 'single' and day.policy_decision['guard'] == GUARD
                assert sel.primary_candidate['batter_id'] == 100
                bar = strategy.effective_pick_bar(50, DAY, False)
                assert bar is not None and bar <= .83
                print('direct pick bar', bar)
            else:
                out = scheduler.run_single_check(DAY, list(statuses), set(), cfg, .03)
                assert out['should_post'] and out['selection'].decision.guard == GUARD
                assert_saved(root)
            if name in ('run', 'orchestrate'):
                send.assert_called_once()
                assert 'Fixture batter 100' in send.call_args.args[0]
            else:
                send.assert_not_called()
            print(name, 'single 100 with guard')


def scheduler_fallbacks():
    for branch in ('in-loop', 'final'):
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            frame = data(.96, projected=True)
            cfg, _ = setup(stack, root, frame)
            games = scheduler.fetch_schedule(DAY)
            for game in games:
                game['teams'] = {side: {'team': {'name': f'{side}-{game["gamePk"]}'}}
                                 for side in ('home', 'away')}
            stack.enter_context(patch.object(scheduler, 'fetch_schedule', lambda *_: games))
            send = stack.enter_context(patch.object(dm, 'send_dm', Mock(return_value='captured-dm')))
            if branch == 'final':
                cached, _ = choose(cfg)
                picks.save_pick(cached, root)
                stack.enter_context(patch.object(orchestrator, 'run_cascade', Mock(side_effect=[(None, None), (frame, 'real-refresh')])))
            real_delivery = scheduler._deliver_and_lock_pick
            delivered = stack.enter_context(patch.object(scheduler, '_deliver_and_lock_pick', wraps=real_delivery))
            scheduler.run_day(DAY, cfg)
            label = delivered.call_args.args[5]
            assert (label == 'final fallback') == (branch == 'final'), label
            assert delivered.call_count == 1
            send.assert_called_once()
            assert 'Fixture batter 100' in send.call_args.args[1]
            assert_saved(root)
            rec, _ = record(root)
            assert rec['action'] == 'single' and rec['policy_guard'] == GUARD
            assert rec['primary']['batter_id'] == 100
            for key, value in {'PICKS_DIR': root, 'fetch_bluesky_posts': lambda: [],
                               'load_orchestrator_config': lambda: cfg,
                               '_build_live_game_data': lambda *_: ([], None)}.items():
                stack.enter_context(patch.object(web, key, value))
            html = web.render_page()
            assert web._decision_action(DAY) == 'single' and 'Fixture batter 100' in html
            assert '<strong>SKIP TODAY</strong>' not in html
            print(branch, 'delivery label:', label, 'record and dashboard show single 100')


if __name__ == '__main__':
    cli_and_shadow()
    scheduler_fallbacks()
