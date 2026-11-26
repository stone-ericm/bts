#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p /tmp/task13/retrieve_logs

retrieve_one() {
  local seed=$1
  local ip=$2
  local log=/tmp/task13/retrieve_logs/seed${seed}.log
  echo "[box=$seed ip=$ip] start $(date -Iseconds)" > $log
  # No trailing slash on source glob — preserves directory structure (data/simulation_seedN/...)
  rsync -az --info=stats0 \
    "root@${ip}:/root/projects/bts/data/simulation_seed*" \
    "data/" >> $log 2>&1
  echo "[box=$seed ip=$ip] done $(date -Iseconds) rc=$?" >> $log
}
export -f retrieve_one

awk '{print $1, $3}' /tmp/task13/ips.tsv | xargs -P 17 -L 1 bash -c 'retrieve_one "$@"' _

echo "===== local seed dirs after retrieval ====="
ls -d data/simulation_seed*/ 2>/dev/null | sort -V | tee /tmp/task13/seed_dirs.txt
echo "Total: $(ls -d data/simulation_seed*/ 2>/dev/null | wc -l) dirs"
echo ""
echo "===== sample contents ====="
ls data/simulation_seed0/ 2>/dev/null
