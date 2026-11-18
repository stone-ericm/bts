"""Check source identity against the locally archived base."""

import ast
from pathlib import Path

from day_probe import REVIEW, ROOT


def source_files(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*.py")}


def definitions(text):
    lines = text.splitlines(keepends=True)
    return {
        node.name: "".join(lines[min([node.lineno, *(d.lineno for d in node.decorator_list)]) - 1:node.end_lineno])
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


if __name__ == "__main__":
    base = source_files(REVIEW / "base" / "src")
    head = source_files(ROOT / "src")
    own = Path("bts/scheduler.py")
    assert base.keys() == head.keys()
    assert {name for name in base if base[name] != head[name]} == {own}
    before, after = [definitions(data[own].decode()) for data in (base, head)]
    assert {name for name, code in before.items() if code != after[name]} == {"run_day"}
    run_day = after["run_day"]
    start = run_day.index('    try:\n        if (delivery_mode in {"dm", "public"} and _contest_state_required(config)')
    end = run_day.index("    # 4. Main loop", start)
    assert (run_day[:start] + run_day[end:]) == before["run_day"]
    print(f"{len(head) - 1} other Python source files and {len(before) - 1} pre-existing scheduler definitions byte-identical")
    print("run_day body byte-identical with the inserted advisory removed")
