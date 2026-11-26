"""Keep the old text suite's unmocked status fallback offline."""

from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from bts import util


def unavailable(url, **_):
    print(f"Captured legacy status fallback for {url}", flush=True)
    raise HTTPError(url, 404, "Offline status fixture", None, None)


if __name__ == "__main__":
    with patch.object(util, "urlopen", unavailable):
        raise SystemExit(pytest.main(["-q", "-s", "--tb=short", "tests/test_wr_e_texts.py"]))
