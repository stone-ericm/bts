"""Inventory changed clock/refusal lines and candidate operator instructions."""
import ast
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORBIDDEN = ("handover", "contract-check", "STOP-", "kickoff-")
CLOCK = re.compile(r"now\(|today\(|%Y|2026|2027|SEASON_|date\.fromisoformat")
REFUSAL = re.compile(r"\braise\b|\brefus(?:e|al|ed|ing)\b|sys\.exit\(|return False", re.I)
IMPERATIVE = re.compile(
    r"(?:^|[.!?;]\s|[“‘`\"]|\bthen\s|\bplease\s)"
    r"(?:run|do not|never|use|check|inspect|verify|compare|enter|keep|preserve|contact|"
    r"restart|stop|wait|review|restore|reseed|follow|archive|read|record|set|add|install|"
    r"send|refresh|mark|replace|recheck|ask|remove|change|show|initialize|rerun|"
    r"re-run|switch|escalate|prefer|select|give|leave|report|back off)\b",
    re.I,
)
COMMAND = re.compile(r"(?:\bbts\s+[-a-z][a-z-]*|\bsystemctl\b|\bcrontab\b|"
                     r"\bgit (?:push|revert|pull|checkout|reset)\b|\buv run\b)")


def diff_inventory():
    patch = subprocess.check_output(
        ["git", "diff", "619abda7", "HEAD", "--unified=0", "--",
         "src", "scripts/cron-setup-hetzner.sh", ".github/workflows/deploy.yml",
         "scripts/systemd", "config"], cwd=ROOT, text=True,
    )
    entries = {"clock": [], "refusal": []}
    location, line = "", 0
    for row in patch.splitlines():
        if row.startswith("+++ b/"):
            location = row[6:]
        elif row.startswith("@@"):
            line = int(re.search(r"\+(\d+)", row).group(1))
        elif row.startswith("+") and not row.startswith("+++"):
            if not any(part in location for part in FORBIDDEN):
                for name, pattern in (("clock", CLOCK), ("refusal", REFUSAL)):
                    if pattern.search(row[1:]):
                        entries[name].append([f"{location}:{line}", row[1:].strip()])
            line += 1
        elif row.startswith(" "):
            line += 1
    (HERE / "diff-inventory.json").write_text(json.dumps(entries, indent=2))
    return {key: len(lines) for key, lines in entries.items()}


def render(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(render(value) if isinstance(value, ast.Constant) else
                       "{" + ast.unparse(value.value) + "}" for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return render(node.left) + render(node.right)
    return ""


def source():
    for path in sorted((ROOT / "src").rglob("*.py")):
        if any(part in str(path.name) for part in FORBIDDEN):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
                continue
            parent = parents.get(id(node))
            if isinstance(parent, (ast.JoinedStr, ast.BinOp)):
                continue
            sentence = render(node).strip()
            if len(sentence) > 18 and (IMPERATIVE.search(sentence) or COMMAND.search(sentence)):
                yield str(path.relative_to(ROOT)), node.lineno, " ".join(sentence.split())


def docs():
    for filename in ("docs/ops/2027-season-start.md", "CLAUDE.md", "INCIDENT.md"):
        for index, row in enumerate((ROOT / filename).read_text().splitlines(), 1):
            # Preserve links as printed references, without opening their targets.
            for part in re.split(r"<(?:/)?li>|(?<=[.!?])\s+(?=[A-Z*])", row):
                text = part.strip()
                if text and (IMPERATIVE.search(text.strip("-*# >")) or COMMAND.search(text)):
                    yield filename, index, text


def main():
    entries = sorted(set((*source(), *docs())))
    # The hand-reviewed rows remain above this appendix on repeat runs.
    report = HERE / "operator-text.md"
    marker = "## Candidate instruction index"
    body = report.read_text().split(marker)[0].rstrip()
    appendix = [
        "", "", marker, "",
        "Text search and Python string extraction also surfaced these fragments and command examples. "
        "The table above records the tested or traced instructions. An index row without a matched assessment "
        "is marked unclear. Braces denote interpolated values.", "",
        "| Location | Sentence or template from source | Worst state tried | Result | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for path, line, text in entries:
        text = text.replace("|", "&#124;")
        appendix.append(f"| `{path}:{line}` | {text} | See assessed rows above for exercised cases | "
                        "Source inventory | unclear by index alone |")
    report.write_text(body + "\n".join(appendix) + "\n")
    print(json.dumps({"operator_candidates": len(entries), **diff_inventory()}))


if __name__ == "__main__":
    main()
