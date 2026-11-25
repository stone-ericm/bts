import pytest


@pytest.fixture(autouse=True)
def fail_unstubbed_game_status_request(request, monkeypatch):
    if 'test_real_lineup_skip_return_against_base[status_failure]' not in request.node.nodeid:
        return

    from bts import picks

    def offline(*args, **kwargs):
        raise OSError('synthetic offline game status')

    monkeypatch.setattr(picks, 'retry_urlopen', offline)
