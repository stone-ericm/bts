"""Check every reachable bar, including each effective tail best."""
from collections import Counter
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys

import bts
from bts import strategy
from scripts.audit.season_walk import walk


work = Path(__file__).resolve().parent
base = work / "base/src"
head = work.parent.parent / "src"
source = {"base": base, "head": head}[sys.argv[1]]
assert Path(bts.__file__).is_relative_to(source), bts.__file__
artifact = strategy._load_mdp()
assert artifact["policy_table"] is not None and artifact["tail"] is not None
end = walk.LAST_GAME_DAY + timedelta(days=1)
counts = Counter()
fingerprint = sha256()
for days in range(artifact["season_length"] + 2):
    day = (end - timedelta(days=days)).isoformat()
    for streak in range(58):
        regime, _ = strategy._regime(artifact, streak, day)
        bests = [(None, None)]
        if regime == strategy.OBJECTIVE_TAIL:
            bests += [(best, "trusted") for best in range(streak, 58)]
            bests += [(57, "untrusted")]
        for saver in (False, True):
            for best, quality in bests:
                bar = strategy.effective_pick_bar(streak, day, saver, artifact, best, quality)
                counts[regime] += 1
                fingerprint.update((json.dumps([day, streak, saver, best, quality, bar]) + "\n").encode())
print("Imported:", bts.__file__)
print("Complete reachable reach-57 and tail grid:", dict(counts), "total:", counts.total())
print("SHA256:", fingerprint.hexdigest())
