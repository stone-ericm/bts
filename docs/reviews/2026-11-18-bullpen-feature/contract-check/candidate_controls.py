import sys

import pytest

from bts.features import compute
from bts.model import predict
from identity import make_candidate


candidate = make_candidate()['compute_all_features']
# Resolve globals from the live module so each test's lookup monkeypatch is seen.
candidate = type(candidate)(candidate.__code__, vars(compute), name=candidate.__name__)
compute.compute_all_features = candidate
predict.compute_all_features = candidate
raise SystemExit(pytest.main(sys.argv[1:]))
