import datetime as dt
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from probe_support import DAY, HERE, end_of_day, flag, manual


logging.disable(logging.CRITICAL)


def inspect(picks, name):
    path = picks.parent / "health_state" / name
    return json.loads(path.read_text())


def run(picks, day=DAY.date(), wall=DAY, transport=None):
    transport = transport if transport is not None else Mock(return_value="sent")
    result = end_of_day(picks, day, wall, transport)
    return result, [call.args[1] for call in transport.call_args_list]


with TemporaryDirectory(dir=HERE, prefix="delivery-") as directory:
    root = Path(directory)
    day2 = DAY + dt.timedelta(days=1)
    after_midnight = DAY + dt.timedelta(hours=3)
    for initial, later in (("not_earned", "used"), ("used", "not_earned")):
        picks = root / initial / "picks"
        flag(picks, "active")
        manual(picks, state=initial)
        first, sent = run(picks)
        assert len(first) == len(sent) == 1
        manual(picks, state=later)
        second, suppressed = run(picks, wall=after_midnight)
        assert len(second) == 1 and not suppressed
        status = inspect(picks, "health_dm_delivery_status.json")
        print(initial, "then", later, "same processed day", status["sent_sources"],
              "second suppressed", bool(second) and not suppressed)
        manual(picks, now=day2, state=later)
        _, sent2 = run(picks, day=day2.date(), wall=day2)
        assert len(sent2) == 1 and "2nd consecutive day" in sent2[0]
        print("next day", sent2[0].splitlines()[0], "2nd consecutive day")

    picks = root / "historical" / "picks"
    flag(picks, "active")
    manual(picks)
    result, sent = run(picks, wall=after_midnight)
    print("historical DM", sent[0])
    assert len(result) == len(sent) == 1 and "Prior slate" in sent[0] and "`" not in sent[0]
    assert inspect(picks, "health_dm_delivery_status.json")["sent_day"] == str(DAY.date())
    assert not run(picks, wall=after_midnight)[1]
    manual(picks, now=day2)
    assert len(run(picks, day=day2.date(), wall=day2)[1]) == 1

    picks = root / "out of order" / "picks"
    flag(picks, "active")
    manual(picks, now=day2, state="not_earned")
    assert run(picks, day=day2.date(), wall=day2)[1]
    manual(picks)
    assert run(picks, wall=after_midnight)[1]
    manual(picks, now=day2, state="not_earned")
    assert run(picks, day=day2.date(), wall=day2)[1]
    print("D+1, D after midnight, D+1 again: all sent because status holds one date")

    picks = root / "failed" / "picks"
    flag(picks, "active")
    manual(picks)
    _, attempted = run(picks, transport=Mock(side_effect=RuntimeError("offline test")))
    assert attempted and inspect(picks, "health_dm_delivery_status.json")["status"] == "failed"
    _, retried = run(picks, transport=Mock(return_value=""))
    assert retried and inspect(picks, "health_dm_delivery_status.json")["status"] == "sent"
    assert not run(picks)[1]
    print("failed retry attempted, empty ID stored as sent")

    picks = root / "oom" / "picks"
    flag(picks, "active")
    manual(picks)
    result, sent = run(picks)
    print("oom directory", [(level, source) for level, source, _ in result])
    assert any(source == "analytics_job_oom" for _, source, _ in result)
    assert "CRITICAL" in sent[0].splitlines()[0]
