#!/bin/zsh
# One wrapped job: run_all (out), gap (out2), batch3 part a (out3), then the merged head's season walk.
C=/Users/eric/bts-runs/fresh15/closing3
zsh $C/run_all.sh; zsh $C/gap.sh; zsh $C/batch3a.sh
cd /Users/eric/projects/bts-work/wr3-harness
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
mkdir -p closing-a/walk
timeout 2400 env PYTHONPATH=$PWD/src .venv/bin/python -m scripts.audit.season_walk.walk --out closing-a/walk/walk-final3.json > closing-a/walk/walk.log 2>&1; echo "walk rc=$?" >> closing-a/walk/walk.log
echo ALL-DONE >> closing-a/walk/walk.log
# The extended walk of CLOSING-TEST-2 part d (branch audit/season-walk-h2g's walk.py, not merged), on FINAL3's code:
mkdir -p closing-a/walk-ext/scripts/audit/season_walk
cp scripts/audit/season_walk/__init__.py closing-a/walk-ext/scripts/audit/season_walk/ 2>/dev/null; touch closing-a/walk-ext/scripts/__init__.py closing-a/walk-ext/scripts/audit/__init__.py closing-a/walk-ext/scripts/audit/season_walk/__init__.py
cp /Users/eric/bts-runs/fresh15/closing3/walk_h2g_ext.py closing-a/walk-ext/scripts/audit/season_walk/walk.py
ln -sfn /Users/eric/projects/bts-work/wr3-harness/data closing-a/walk-ext/data
(cd closing-a/walk-ext && timeout 2400 env PYTHONPATH=/Users/eric/projects/bts-work/wr3-harness/src:/Users/eric/projects/bts-work/wr3-harness/closing-a/walk-ext ../../.venv/bin/python -m scripts.audit.season_walk.walk --extended --out ../walk/walk-final3-extended.json > ../walk/walk-extended.log 2>&1; echo "walk-extended rc=$?" >> ../walk/walk-extended.log)
echo ALL-DONE-2 >> closing-a/walk/walk.log
