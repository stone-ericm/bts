from unittest.mock import patch

import probe_day as day
from bts import scheduler
from support import DAY, at


def observe_without_coverage(directory, count, kind, date):
    with patch.object(scheduler, '_print_morning_check') as journal:
        scheduler._morning_state_check(day.walk._config(directory), directory, str(date), at(date), at(date, 13))
    assert f'{kind} contest state accepted (streak {count})' in journal.call_args.args[0]
    for previous in (None, DAY):
        alerts = day.health.check(directory, now=at(date), expected=True, historical_day=previous)
        assert len(alerts) == 1 and alerts[0].level == 'CRITICAL', alerts
        assert 'coverage' in alerts[0].message and ('no source_date' in alerts[0].message or 'no coverage date' in alerts[0].message)
    return journal.call_args.args[0] + ' Health separately reports missing coverage.'


if __name__ == '__main__':
    day.typo('dated', 75, 75, (57, 57))
    day.typo('dated', 12, 157, (12, 57))
    day.typo('dated', 12, 157, (12, 12))
    day.typo('automatic', 5, 157, (5, 5))
    day.typo('dated', 75, 75, (0, 57))
    with patch.object(day, 'observe', observe_without_coverage):
        day.typo('automatic', 5, 400, (5, 5), coverage=False)
