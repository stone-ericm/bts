from contextlib import ExitStack, nullcontext, redirect_stderr
from dataclasses import replace
from datetime import date, datetime, timedelta
from html import escape
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import orchestrator, picks, scheduler, web
from scripts.audit.season_walk import walk
from tests.test_scheduler_tail_objective import _sel, _tail_skip_decision


def state(day, **fields):
    return scheduler.SchedulerState(str(day), 'pinned', [], [], [], False, None, None, None, **fields)


def render(label, day, streak, probability, sentence, statuses=None, mode=None):
    now = datetime.fromisoformat(f'{day}T18:35:00-04:00')
    with TemporaryDirectory(dir=Path(__file__).parent) as temp, ExitStack() as stack:
        path = Path(temp) / 'picks'
        path.mkdir()
        cfg = walk._config(path)
        cfg['bluesky'] = {'dm_recipient': 'capture.invalid'}
        frame = walk._predictions(day, probability)
        if mode == 'removed':
            frame.loc[0, 'p_game_hit'] = 0.999
        if mode in {'tail', 'missing_bar'}:
            reason, bar = ('Stop!', .8) if mode == 'tail' else (None, None)
            selection = _sel(replace(_tail_skip_decision(), reason=reason, pick_bar=bar, streak=streak))
            stack.enter_context(patch.object(orchestrator, 'run_and_pick', return_value=(frame, selection, 'probe')))
        if mode == 'sel_none':
            stack.enter_context(patch.object(orchestrator, 'run_and_pick', return_value=(frame, None, 'probe')))
        def get_statuses(day):
            if mode == 'unreadable':
                raise OSError('captured status failure')
            values = statuses or ['Scheduled'] * 3
            return {70000 + i: {'abstract': 'P' if v == 'Scheduled' else 'F', 'detailed': v}
                    for i, v in enumerate(values)}
        stack.enter_context(walk.offline(now))
        stack.enter_context(patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')))
        stack.enter_context(patch.object(picks, 'get_game_statuses_detailed', get_statuses))
        stack.enter_context(patch.object(scheduler, 'count_new_confirmations', return_value=1))
        stack.enter_context(patch.object(scheduler, 'heartbeat_watchdog', side_effect=lambda *a, **kw: nullcontext()))
        stack.enter_context(patch.object(web, 'PICKS_DIR', path))
        stack.enter_context(patch.object(web, 'fetch_bluesky_posts', return_value=[]))
        assert walk.install_state(path, day, walk.State('dated', streak, streak, 'used', probability)) == []
        walk._set_clock(now)
        def check():
            with redirect_stderr(StringIO()) as journal:
                result = scheduler.run_single_check(str(day), [70000, 70001, 70002], set(), cfg, 0.03)
            return result, [line for line in journal.getvalue().splitlines() if line.startswith('  SKIP — ')]
        result, journal = check()
        summary = result['skip_summary']
        full = sentence if result['selection'] is None or result['selection'].decision is None else (
            sentence + f' Streak used at this check: {streak}.')
        dm = scheduler.format_skip_dm(str(day), summary)
        assert dm == f'BTS {day}: No pick at this check. {full}', dm
        assert journal == [f'  SKIP — {full}'], journal
        banner = ('<div class="skip-banner"><strong>LAST RECORDED NO-PICK SUMMARY</strong>'
                  f'<span>{escape(full)}</span></div>')
        assert web.render_skip_banner(summary) == banner
        current = state(day, skip_summary=summary)
        captured = []
        assert scheduler.maybe_notify_skip(current, summary, cfg, now_iso=now.isoformat(),
                                           send=lambda _, body: captured.append(body))
        assert captured == [dm]
        scheduler.save_state(current, path)
        assert banner in web.render_page()
        print(label, dm, '|', journal[0], '|', banner)
        if mode == 'removed':
            walk._set_clock(now + timedelta(minutes=20))
            with patch.object(orchestrator, 'run_and_pick', return_value=(None, None, None)):
                second, journal = check()
            assert second['skip_summary'] is None and not journal
            assert banner in web.render_page()
            restarted = scheduler.carry_forward_skip_state(state(day), scheduler.load_state(str(day), path))
            assert restarted.skip_summary == summary and restarted.skip_notified_at == now.isoformat()
            assert not scheduler.maybe_notify_skip(restarted, summary, cfg, now_iso=now.isoformat(), send=None)
            print('C11 18:55 check without predictions, then same-day restart:', banner)
        if label.startswith('C4'):
            chosen = result['selection']
            print('NO POLICY DECISION:', None if chosen is None else chosen.decision,
                  'stored display streak:', summary['streak'], 'and suffix omitted')


day = date(2027, 7, 15)
render('C1', date(2027, 8, 30), 56, 0.99,
       'Candidate used in the decision: Batter 0 (T0) 99.0%, at or above the streak-56 bar (82.5%).')
render('C2', date(2027, 6, 25), 56, 0.99,
       'Candidate used in the decision: Batter 0 (T0) 99.0%. No probability comparison is available for this check.')
render('C3', day, 10, 0.76, 'Candidate used in the decision: Batter 1 (T1) 75.0%, below the streak-10 bar '
       '(79.6%) using unrounded values.', ['Final', 'Scheduled', 'Scheduled'], mode='removed')
for scenario in ('none_eligible', 'unreadable', 'sel_none'):
    render('C4-' + scenario, day, 10, 0.999,
           'No candidate was chosen. Top model prediction (eligibility unverified): Batter 0 (T0) 99.9%.',
           ['Final'] * 3 if scenario == 'none_eligible' else None, mode=scenario)
render('K1', day, 10, 0.75, 'Candidate used in the decision: B1 (NYM) 72.0%. Reason: Stop!', mode='tail')

render('K4', day, 10, 0.75, 'Candidate used in the decision: B1 (NYM) 72.0%. '
       'No probability comparison is available for this check.', mode='missing_bar')
