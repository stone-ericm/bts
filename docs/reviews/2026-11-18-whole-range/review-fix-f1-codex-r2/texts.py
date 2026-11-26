"""Render the whole alerts with mixed leg times, missing fields and cutoff timing."""
from shlex import quote
import sys

from bts import entry_receipt
from support import DAY, DOUBLE, MATCH, PITCH, RIGHT, RIGHT_DD, WRONG, WRONG_DD, dump, print_source, world

FIRST = ('⚠️ BTS entry game needs verification. {legs}. Check the delivered pick and the game shown in MLB '
         'for each listed leg. Change only a leg whose correct game you can verify and that MLB still lets you edit. '
         'Keep any correct leg as entered.')
REMINDER = ('⏰ BTS entry game verification is still needed. {legs}. '
            'Check the delivered pick against the game shown in MLB.')
LOCKED = ('⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send batter 1010 + DD Batter 1020. '
          'MLB game evidence needs verification. {legs}. Check your DMs and the MLB entry before making any change. '
          'Run `bts season-contract status --picks-dir {directory}`.')
SECOND_PITCH = f'{DAY}T23:55:00Z'
DOUBLE_LEG = ('double-down Batter 1020: recommended game 201 (first pitch 7:55 PM ET), '
              'the entry is recorded in game 202')


def check_texts():
    with world('quoted directory') as w:
        w.commit(pitch=PITCH, dd=(1020, 201, SECOND_PITCH), name='')
        w.capture([RIGHT, WRONG_DD])
        assert w.check('18:30', DOUBLE)['dm'] == [FIRST.format(legs=DOUBLE_LEG)]
        dump('only affected leg first warning', w.check('19:05', DOUBLE)['dm'])
        assert w.check('19:05', DOUBLE)['dm'] == []
        assert w.check('19:20', DOUBLE)['dm'] == [REMINDER.format(legs=DOUBLE_LEG)]
        assert w.check('19:25', DOUBLE)['stdout'] == (
            f'check-pick-entered: {DAY} game verification unresolved (double-down); re-verifying each run until cutoff\n')

    with world('quoted directory') as w:
        w.commit(pitch=PITCH, dd=(1020, 201, SECOND_PITCH), name=None, delivery='locked_unconfirmed')
        w.capture([RIGHT, WRONG_DD])
        expected = LOCKED.format(legs=DOUBLE_LEG, directory=quote(str(w.picks)))
        for at in ('18:30', '19:05', '19:20'):
            result = w.check(at, DOUBLE)
            assert result['dm'] == [expected]
        dump('locked warning at all three tiers', expected)

    with world('case-b-text') as w:
        w.commit(pitch=PITCH, dd=(1020, 201, SECOND_PITCH), name=None)
        w.capture([WRONG, RIGHT_DD])
        assert w.check('18:30', DOUBLE)['marker']['reason'] == 'wrong_game'
        w.commit(game=None, pitch=PITCH, dd=(1020, 201, SECOND_PITCH), name=None)
        w.capture([RIGHT], f'{DAY}T230000Z')
        result = w.check('19:05', DOUBLE)
        expected = 'pick batter 1010: recommended game ID unavailable (first pitch 7:40 PM ET)'
        assert result['dm'] == [REMINDER.format(legs=expected)]
        dump('case (b) with unbound game and missing name', result['dm'])

    with world('capture-crosses-cutoff') as w:
        w.commit(pitch=PITCH)
        w.capture([WRONG])
        seconds = [1000]
        read = entry_receipt.load_units

        def timed_capture(*args):
            result = read(*args)
            seconds[0] = 1061
            return result

        result = w.check('19:34', MATCH, more_patches=[('bts.entry_receipt.load_units', timed_capture),
                                                       ('time.monotonic', lambda: seconds[0])])
        assert result['receipts'][0]['observation']['before_cutoff'] is True
        assert result['marker']['game_unresolved'] is True and not result['attempted_dm']
        assert result['stdout'] == ('check-pick-entered: submission cutoff passed during this check; '
                                    'skipping the nag (entry can no longer change)\n')
        dump('capture crosses cutoff', result['stdout'])


def unreadable_pitch():
    with world('unreadable-pitch') as w:
        w.commit(pitch=PITCH, dd=(1020, 201, 'broken timestamp'))
        w.capture([RIGHT, WRONG_DD])
        result = w.check('18:30', DOUBLE)
        assert result['exception'] == 'ValueError' and result['fetches'] == []
        assert result['marker'] is None and not result['attempted_dm']
        dump('unreadable second pitch raises before fetch', result['exception'])


if __name__ == '__main__':
    print_source()
    unreadable_pitch()
    if sys.argv[1] == 'candidate':
        check_texts()
