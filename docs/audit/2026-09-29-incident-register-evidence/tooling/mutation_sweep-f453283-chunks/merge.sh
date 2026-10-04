#!/bin/bash
# merge.sh: merge the final strict sweep at f453283 into the branch's evidence (run once every shard prints 'nothing left').
set -euo pipefail
E=/Users/eric/projects/bts-w15-evidence
O=$E/out/f453283
B=/Users/eric/projects/bts-w15/docs/audit/2026-09-29-incident-register-evidence/tooling
files=()
for i in 0 1 2 3; do for f in $(ls $O/shard-$i-*.txt | sort); do files+=("$f"); done; done
printf '%s\n' "${files[@]}" > $O/merge_order.txt
git -C /Users/eric/projects/bts-w15 mv $B/mutation_sweep.txt $B/mutation_sweep-c59ee58.txt
python3 $B/merge_shards.py f453283 $B/mutation_sweep.py $B/mutation_sweep.txt "${files[@]}" | tee $O/merge.log
mkdir -p $B/mutation_sweep-f453283-chunks
cp $O/shard-*.txt $O/shard-*.meta $O/run_rest.sh $O/merge_order.txt $O/merge.log $B/mutation_sweep-f453283-chunks/
cp -R $O/keep $B/mutation_sweep-f453283-chunks/keep
