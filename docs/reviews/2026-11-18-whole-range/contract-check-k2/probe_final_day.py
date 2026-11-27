import json
from math import nextafter

from bts import strategy
from probe_policy_v2 import ARTIFACTS, STREAKS, row


def main():
    edges = ARTIFACTS["boundaries"]
    probabilities = [0.0, *edges, *(nextafter(edge, 0) for edge in edges), 1.0]
    data = []
    for d in ("2027-09-26", "2027-09-27"):
        for streak in STREAKS:
            for saver in (False, True):
                for probability in probabilities:
                    bests = [(streak, "trusted")]
                    if streak < 55:
                        bests += [(streak + 1, "trusted"), (streak + 2, "trusted"),
                                  (streak + 2, "untrusted"), (None, None)]
                    for best, status in bests:
                        result = row(d, streak, saver, probability, best, status)
                        bar = strategy.effective_pick_bar(streak, d, saver, ARTIFACTS, best, status)
                        data.append({"day": d, "saver": saver, "p": probability,
                                     "display_bar": bar, **result})
    print(json.dumps(data))


if __name__ == "__main__":
    main()
