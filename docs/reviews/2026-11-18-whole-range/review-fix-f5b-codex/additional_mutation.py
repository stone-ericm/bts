from pathlib import Path
import sys

from mutation_checks import CASES, CLI, check


CASES['manual-resets-live-tiers'] = ((CLI, '    if prior.get("status") == "confirmed":\n',
                                      '    if not cron:\n'
                                      '        prior.pop("escalations", None)\n'
                                      '        if prior.get("status") == "alerted":\n'
                                      '            prior = {}\n'
                                      '    if prior.get("status") == "confirmed":\n'),)
check('manual-resets-live-tiers', Path(sys.argv[1]))
