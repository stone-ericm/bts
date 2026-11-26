#!/bin/zsh
# Part d jobs (closing test re-run), one wrapped call: RED (new tests vs main's walk.py), GREEN (both walk
# test files vs the extension), then the extended walk end to end on FINAL2's code and the shipped artifacts.
cd /Users/eric/projects/bts-work/wr2-walk || exit 9
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
OUT=$PWD/walk-out; mkdir -p $OUT
PY=$PWD/.venv/bin/python
W=scripts/audit/season_walk/walk.py
step() { local name=$1 secs=$2; shift 2; echo "$(date '+%F %T') start $name" >> $OUT/steps.txt
  timeout $secs "$@" > $OUT/$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $name rc=$rc" >> $OUT/steps.txt; }
cp $W $OUT/walk.py.extension
git show main:$W > $W
step red 900 env PYTHONPATH=$PWD/src $PY -m pytest -p no:cacheprovider -q tests/scripts/season_walk/test_season_walk_h2g.py -rfE
cp $OUT/walk.py.extension $W
step green 1800 env PYTHONPATH=$PWD/src $PY -m pytest -p no:cacheprovider -q tests/scripts/season_walk -rfE
step walk-extended 2400 env PYTHONPATH=$PWD/src $PY -m scripts.audit.season_walk.walk --extended --out $OUT/walk-final2-extended.json
echo "$(date '+%F %T') ALL STEPS DONE" >> $OUT/steps.txt
