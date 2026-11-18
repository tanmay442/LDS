"""Check every notebook for valid JSON, sane structure and Python syntax errors.

Usage: python3 scripts/validate_notebooks.py
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def clean_source(source):
    if isinstance(source, list):
        source = "".join(source)
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate(path):
    problems = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"invalid JSON: {error}"], 0, 0

    if notebook.get("nbformat") != 4:
        problems.append("nbformat is not 4")

    cells = notebook.get("cells", [])
    if not cells:
        problems.append("no cells")

    if cells and cells[0].get("cell_type") != "markdown":
        problems.append("first cell is not markdown")

    all_text = " ".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else cell.get("source", "")
        for cell in cells
    )
    if "http" not in all_text:
        problems.append("no documentation links found")

    code_count = 0
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        code_count += 1
        source = cell.get("source", "")
        if not source:
            problems.append(f"cell {index}: empty code cell")
            continue
        try:
            ast.parse(clean_source(source))
        except SyntaxError as error:
            problems.append(f"cell {index}: SyntaxError: {error.msg} (line {error.lineno})")

    return problems, len(cells), code_count


def main():
    files = sorted(
        path for path in NOTEBOOKS.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )
    if not files:
        print("No notebooks found")
        return 1

    failures = 0
    for path in files:
        problems, cell_count, code_count = validate(path)
        status = "OK" if not problems else "PROBLEMS"
        print(f"{status:9} {str(path.relative_to(NOTEBOOKS)):45} {cell_count:3} cells ({code_count} code)")
        for problem in problems:
            print(f"          - {problem}")
        failures += len(problems)

    print(f"\n{len(files)} notebooks checked, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
