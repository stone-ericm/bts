"""Run caller comparisons with one clustered check in the in-loop fixture."""

from types import SimpleNamespace
from unittest.mock import patch

import probe_callers

original = probe_callers.replay
world = probe_callers.world.__wrapped__


def replay(case, *args):
    def clustered(*fixture_args):
        result = world(*fixture_args)
        if case == "in_loop":
            result[0]["scheduler"]["cluster_min"] = 45
        return result

    with patch.object(probe_callers, "world", SimpleNamespace(__wrapped__=clustered)):
        return original(case, *args)


if __name__ == "__main__":
    with patch.object(probe_callers, "replay", replay):
        probe_callers.main()
