#!/bin/bash
# v2.5_retrieve.sh — Pull harness output JSON files from all 4 Vultr boxes.
#
# Mirrors task13_retrieve.sh but for the v2.5 ablation cell outputs.
# Reads /tmp/v2.5/instances.tsv (cell_label TAB instance_id TAB ip).
# Pulls data/validation/falsification_harness_v2.5_cell*.json[/heatmap.json]
# into local data/validation/.
#
# Exit behaviour: exits 0 if AT LEAST 1 cell retrieved successfully; exits 1
# only if ALL cells fail. A per-cell status summary is always printed so a
# partial failure is clearly visible without the whole script aborting early.
#
# Usage: bash scripts/v2_5_retrieve.sh

set -uo pipefail
cd "$(dirname "$0")/.."

if [ ! -f /tmp/v2.5/instances.tsv ]; then
  echo "ERROR: /tmp/v2.5/instances.tsv not found. Run v2.5_provision.sh first." >&2
  exit 1
fi

mkdir -p data/validation /tmp/v2.5/retrieve_logs

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"
STATUS_DIR=/tmp/v2.5/retrieve_status
mkdir -p "$STATUS_DIR"
# Clear any stale per-cell status files from a previous run
rm -f "$STATUS_DIR"/cell_*.rc

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
  local main_rc=$?

  # Pull heatmap JSON (may not exist in all cells; tolerate missing)
  rsync -az --info=stats0 \
    -e "ssh $SSH_OPTS" \
    "root@${ip}:/root/projects/bts/data/validation/falsification_harness_v2.5_cell${cell}_heatmap.json" \
    "data/validation/" >> "$log" 2>&1 || true

  echo "[cell=$cell ip=$ip] done $(date -Iseconds) main_rc=$main_rc" >> "$log"
  # Write exit code to status file for the parent shell to collect
  echo "$main_rc" > "/tmp/v2.5/retrieve_status/cell_${cell}.rc"
  return $main_rc
}
export -f retrieve_one
export SSH_OPTS

# Run retrieval in parallel (4 boxes); set +e so a failed rsync doesn't abort
# xargs early and deprive us of the other cells' output.
set +e
awk '{print $1, $3}' /tmp/v2.5/instances.tsv | xargs -P 4 -L 1 bash -c 'retrieve_one "$@"' _
set -e

# ---- Per-cell status summary ----
echo ""
echo "===== per-cell retrieve status ====="
SUCCEEDED=0
FAILED=0
for rc_file in $(ls "$STATUS_DIR"/cell_*.rc 2>/dev/null | sort); do
  cell=$(basename "$rc_file" | sed 's/cell_//' | sed 's/\.rc//')
  rc=$(cat "$rc_file")
  if [ "$rc" -eq 0 ]; then
    echo "  cell $cell: OK (rc=0)"
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    echo "  cell $cell: FAILED (rc=$rc)  — log: /tmp/v2.5/retrieve_logs/cell${cell}.log"
    FAILED=$((FAILED + 1))
  fi
done
echo "  Succeeded: $SUCCEEDED / $((SUCCEEDED + FAILED))"

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

# Exit nonzero only if every single cell failed
if [ "$SUCCEEDED" -eq 0 ]; then
  echo "" >&2
  echo "ERROR: ALL $FAILED cell(s) failed to retrieve." >&2
  exit 1
fi
