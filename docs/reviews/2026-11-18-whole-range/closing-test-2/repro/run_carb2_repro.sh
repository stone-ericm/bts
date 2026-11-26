#!/bin/zsh
# Coordinator reproduction of Carbon 2's three probes (scripts read by bts-wr-fresh14), one at a time, bounded.
cd /Users/eric/projects/bts-work/wr2-carbon2 || exit 9
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
O=/Users/eric/projects/bts-work/wr2-carbon2/.carbon-review/whole-range/repro; mkdir -p $O
for p in probe_mid_late_and_streaks probe_deploy_and_rollback probe_cron; do
  timeout 1500 .venv/bin/python .carbon-review/whole-range/$p.py > $O/$p.log 2>&1
  rc=$?; echo "$p rc=$rc $(date '+%F %T')" >> $O/rc.txt
done
