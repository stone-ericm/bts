#!/bin/bash
# run_rest.sh <shard 0-3> <run-tag> [max]: run up to <max> (default 22) of shard <shard>'s labels that no earlier
# output file for that shard (shard-<i>-*.txt) has a verdict for, at f453283, in s<i>. Each run starts with its
# own clean baseline (mutation_sweep.main), so a resumed label is measured exactly like a first-pass one.
E=/Users/eric/projects/bts-w15-evidence
O=$E/out/f453283
T=docs/audit/2026-09-29-incident-register-evidence/tooling
i=$1; tag=$2; max=${3:-22}
L=$($E/tool/.venv/bin/python - "$O" "$i" "$max" <<'EOF'
import json, re, sys
from pathlib import Path
O, i, mx = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
labels = json.load(open(O / "shards.json"))[i]
done = set()
for f in O.glob(f"shard-{i}-*.txt"):
    for l in f.read_text().splitlines():
        m = re.match(r"^([A-Z]+[0-9]+) .*: (KILLED|SURVIVED|ERRORED|INCOMPLETE|SKIPPED|FAILED-OTHER|INVALID)", l)
        if m:
            done.add(m.group(1))
print(",".join([l for l in labels if l not in done][:mx]))
EOF
)
test -n "$L" || { echo "s$i: nothing left"; exit 0; }
test -e $O/shard-$i-$tag.txt && { echo "shard-$i-$tag.txt exists"; exit 3; }
for bak in $E/s$i/scripts/audit/incident_register/*.sweepbak $E/s$i/$T/*.sweepbak; do   # a run killed mid-mutant: restore
  [ -e "$bak" ] && mv "$bak" "${bak%.sweepbak}" && echo "restored leftover $(basename $bak)"
done
test "$(git -C $E/s$i rev-parse --short HEAD)" = f453283 || { echo "s$i not at f453283"; exit 3; }
test -z "$(git -C $E/s$i status --short)" || { echo "s$i dirty"; exit 3; }
echo "s$i run $tag: $(echo $L | tr ',' ' ' | wc -w | tr -d ' ') labels"
printf "pin %s\nworktree %s\nkeep %s/keep/shard-%s-%s\ncommand cd %s && nice -n 10 .venv/bin/python %s/mutation_sweep.py %s %s\nstarted %s\n" "$(git -C $E/s$i rev-parse HEAD)" "$E/s$i" "$O" "$i" "$tag" "$E/s$i" "$T" "$E/s$i" "$L" "$(date -u +%FT%TZ)" > $O/shard-$i-$tag.meta
export W15_SWEEP_KEEP=$O/keep/shard-$i-$tag     # every run's JUnit XML + exception-identity records (fresh review F4)
cd $E/s$i && nice -n 10 .venv/bin/python $T/mutation_sweep.py $E/s$i "$L" > $O/shard-$i-$tag.txt 2>&1
echo "exit=$?" >> $O/shard-$i-$tag.txt
grep -h -E "BASELINE|: [A-Z-]+( MUTANT)? \||^DONE|^exit=" $O/shard-$i-$tag.txt | cut -c1-110
echo "s$i dirty=$(git -C $E/s$i status --short | wc -l | tr -d ' ')"
