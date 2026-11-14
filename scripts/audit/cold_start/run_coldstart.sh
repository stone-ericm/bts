#!/bin/zsh
# Cold-start rehearsal, three dates in sequence, from the scratch checkout. Each date gets its own log.
# Usage: run_coldstart.sh <opener> <day2> <next-season-opener>
cd ~/bts-runs/coldstart/repo || exit 1
export UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York
git log --oneline -1
for d in "$@"; do
  echo "=== $d start $(date '+%F %T')"
  .venv/bin/python ../rehearse.py "$d" > ../rehearse-$d.log 2>&1
  echo "=== $d rc=$? end $(date '+%F %T')"
  if [ "$d" = "$2" ]; then
    # After the two 2026 dates: add the full frozen 2026 season for the 2027 date.
    rm -f data/processed/pa_2026.parquet data/processed/.refreshed_2026_*
    cp -c ~/projects/bts/data/processed/pa_2026.parquet data/processed/pa_2026.parquet
    echo "=== swapped in the frozen pa_2026.parquet"
  fi
done
echo "ALL DONE $(date '+%F %T')"
