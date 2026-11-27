from pathlib import Path
from shutil import copytree
from tempfile import mkdtemp


root = Path(mkdtemp(prefix="wr-k2-r2-floor-", dir="/private/tmp"))
copytree("src/bts", root / "src/bts")
strategy = root / "src/bts/strategy.py"
text = strategy.read_text()
old = "longest = _regime(None, streak, date, min_days=1)[0] == OBJECTIVE_LONGEST"
new = "longest = _regime(None, streak, date)[0] == OBJECTIVE_LONGEST"
assert text.count(old) == 1
strategy.write_text(text.replace(old, new))
print(root)
