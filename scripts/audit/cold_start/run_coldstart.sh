#!/bin/zsh
# Cold-start rehearsal for three dates, in sequence, from the scratch checkout. Each date gets its own log.
#
#   run_coldstart.sh <opener> <day2> <next-season-date>
#
# <opener>: must start cold for its season (no season parquet, no refresh marker, no raw directory); predictions and
#           a selected pick are required.
# <day2>:   the day after, run on the state the opener left; predictions and a pick are required.
# <next-season-date>: run after the first season's full file is copied in; no expectation (diagnostic), but a stage
#           that raises still fails.
# Expects the checkout at ~/bts-runs/coldstart/repo, rehearse.py beside it, and the full season file for <opener>'s
# year at ~/projects/bts/data/processed/. Exits non-zero if any date failed or the file swap failed.
if [ $# -ne 3 ]; then
  echo "usage: run_coldstart.sh <opener> <day2> <next-season-date>" >&2
  exit 64
fi
cd ~/bts-runs/coldstart/repo || exit 1
export UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York
git log --oneline -1
shasum -a 256 ../rehearse.py
worst=0
run_date() {  # <date> <extra args...>
  local d=$1; shift
  echo "=== $d start $(date '+%F %T') args: $*"
  .venv/bin/python ../rehearse.py "$d" "$@" > ../rehearse-$d.log 2>&1
  local rc=$?
  echo "=== $d rc=$rc end $(date '+%F %T')"
  [ $rc -ne 0 ] && worst=$rc
  return 0
}
run_date "$1" --require-cold --expect rows
run_date "$2" --expect rows
year=${1%%-*}
src=~/projects/bts/data/processed/pa_$year.parquet
# After the two early dates: put the full season file in place for the next-season date.
if rm -f data/processed/pa_$year.parquet \
   && find data/processed -name ".refreshed_${year}_*" -delete \
   && cp -c "$src" data/processed/pa_$year.parquet; then
  echo "=== swapped in $src"
  shasum -a 256 "$src" data/processed/pa_$year.parquet
else
  echo "=== SWAP FAILED; not running $3"
  echo "ALL DONE rc=1 $(date '+%F %T')"
  exit 1
fi
run_date "$3"
echo "ALL DONE rc=$worst $(date '+%F %T')"
exit $worst
