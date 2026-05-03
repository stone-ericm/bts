#!/bin/bash
# v2.5_retrieve.sh — Pull harness output JSON files from all 4 Vultr boxes.
#
# Mirrors task13_retrieve.sh but for the v2.5 ablation cell outputs.
# Reads /tmp/v2.5/instances.tsv (cell_label TAB instance_id TAB ip).
# Pulls data/validation/falsification_harness_v2.5_cell*.json[/heatmap.json]
# into local data/validation/.
#
# Usage: bash scripts/v2.5_retrieve.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f /tmp/v2.5/instances.tsv ]; then
  echo "ERROR: /tmp/v2.5/instances.tsv not found. Run v2.5_provision.sh first." >&2
  exit 1
fi

mkdir -p data/validation /tmp/v2.5/retrieve_logs

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"

retrieve_one() {
  local cell=$1
  local ip=$2
  local log=/tmp/v2.5/retrieve_logs/cell${cell}.log
  echo "[cell=$cell ip=$ip] start $(date -Iseconds)" > "$log"

  # Pull main result JSON
  rsync -az --info=stats0 \
    -e "ssh $SSH_OPTS" \
    "root@${ip}:/root/projects/bts/data/validation/falsification_harness_v2.5_cell${cell}.json" \
    "data/validation/" >> "$log" 2>&1
  MAIN_RC=$?

  # Pull heatmap JSON (may not exist in all cells; tolerate missing)
  rsync -az --info=stats0 \
    -e "ssh $SSH_OPTS" \
    "root@${ip}:/root/projects/bts/data/validation/falsification_harness_v2.5_cell${cell}_heatmap.json" \
    "data/validation/" >> "$log" 2>&1 || true

  echo "[cell=$cell ip=$ip] done $(date -Iseconds) main_rc=$MAIN_RC" >> "$log"
  return $MAIN_RC
}
export -f retrieve_one
export SSH_OPTS

# Run retrieval in parallel (4 boxes)
awk '{print $1, $3}' /tmp/v2.5/instances.tsv | xargs -P 4 -L 1 bash -c 'retrieve_one "$@"' _

echo ""
echo "===== local v2.5 result files ====="
ls data/validation/falsification_harness_v2.5_cell*.json 2>/dev/null | sort | tee /tmp/v2.5/retrieved_files.txt
echo "Total: $(ls data/validation/falsification_harness_v2.5_cell*.json 2>/dev/null | wc -l | tr -d ' ') files"

# Quick verdict summary
echo ""
echo "===== quick verdict check ====="
for f in $(ls data/validation/falsification_harness_v2.5_cell*.json 2>/dev/null | sort); do
  CELL=$(basename "$f" | sed 's/falsification_harness_v2.5_cell//' | sed 's/\.json//')
  VERDICT=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('verdict', 'N/A'))" 2>/dev/null || echo "parse_error")
  echo "  cell $CELL: $VERDICT"
done
