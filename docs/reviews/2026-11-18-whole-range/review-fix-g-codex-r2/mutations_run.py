"""Use the exact source spacing for the raw-lookup substitution."""
import sys

from mutations import MUTATIONS, mutate

file, before, after, test = MUTATIONS["production_raw"]
MUTATIONS["production_raw"] = (
    file, before.replace("date)\n    return", "date)\n\n    return"), after, test,
)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        mutate(name)
