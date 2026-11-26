"""Pin timestamps in the comparison fixture's synthetic skip decision."""
from datetime import datetime
from unittest.mock import patch

from bts import daily_decision
import grid

original = daily_decision.write_decision


def fixed_decision(*args, **kwargs):
    kwargs.setdefault("now", datetime.fromisoformat("2027-03-24T20:00:00+00:00"))
    return original(*args, **kwargs)


if __name__ == "__main__":
    with patch.object(daily_decision, "write_decision", fixed_decision):
        grid.main()
