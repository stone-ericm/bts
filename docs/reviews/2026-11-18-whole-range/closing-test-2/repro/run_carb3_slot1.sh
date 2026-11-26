#!/bin/zsh
# Runs Carbon 3's slot-request.txt lines one at a time (read and approved by bts-wr-fresh14), each bounded.
cd /Users/eric/projects/bts-work/wr2-carbon3 || exit 9
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
i=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  i=$((i+1))
  timeout 1200 zsh -c "$line"; echo "$(date '+%F %T') line $i rc=$?" >> .carbon-review/whole-range/logs/slot-run-1.rc.txt
done < .carbon-review/whole-range/slot-request.txt
echo "SLOT RUNS DONE $(date '+%F %T')" >> .carbon-review/whole-range/logs/slot-run-1.rc.txt
