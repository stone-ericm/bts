import pytest


@pytest.fixture(autouse=True)
def omit_network_backoff(monkeypatch):
    from bts import picks

    def offline(*args, **kwargs):
        raise OSError('synthetic offline game status')

    monkeypatch.setattr(picks, 'retry_urlopen', offline)
