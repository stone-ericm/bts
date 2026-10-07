"""C2 step 2a tests: every test starts with no current serving witness (code review r4: the witness is a context
variable, which would otherwise carry over between tests in one thread)."""
import pytest


@pytest.fixture(autouse=True)
def _no_current_witness():
    from bts import serving_witness as W
    token = W._CURRENT.set(None)
    yield
    W._CURRENT.reset(token)
