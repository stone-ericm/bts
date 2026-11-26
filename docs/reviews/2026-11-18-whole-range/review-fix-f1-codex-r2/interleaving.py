"""An interrupted fetch lets a second invocation write before the first resumes."""
from support import ALL_CLEAR, DAY, MATCH, PITCH, RIGHT, WRONG, current_receipt, dump, print_source, world


def report(result, w):
    receipt = current_receipt(w, result)
    return {
        'exit': result['exit'], 'marker': result['marker'], 'stdout': result['stdout'],
        'attempted_dm': result['attempted_dm'], 'qualification': receipt.get('qualification'),
    }


def check_overlap():
    # Simultaneous invocations are modeled by nesting the second CLI call in
    # the first invocation's captured pending fetch. No code or marker is patched.
    from click.testing import CliRunner

    with world('overlap') as w:
        w.commit(pitch=PITCH)
        assert w.check('18:25')['marker']['escalations'] == ['initial']
        w.capture([WRONG])
        original_runner = w.runner
        second = {}

        def older_pending(*args, **kwargs):
            w.runner = CliRunner()
            try:
                second.update(w.check('18:45', MATCH, dm='failure'))
            finally:
                w.runner = original_runner
            assert second['marker']['game_unresolved'] is True
            assert second['marker']['escalations'] == ['initial']
            return []

        older = w.check('18:30', more_patches=[
            ('bts.contest_fetch.fetch_pending_predictions', older_pending),
        ])
        w.capture([RIGHT], f'{DAY}T230000Z')
        final = w.check('19:00', MATCH)
        for label, result in [('second finishes first', second), ('older fetch resumes', older), ('follow-up', final)]:
            dump(label, report(result, w))
        assert current_receipt(w, final)['qualification']['slots'][0]['state'] == 'unit_unverified'
        assert not (final['marker']['status'] == 'confirmed' or ALL_CLEAR in final['attempted_dm']), (
            'An overlapping no-entry run erased the wrong-game observation, then a batter match gave an all-clear.'
        )


if __name__ == '__main__':
    print_source()
    check_overlap()
