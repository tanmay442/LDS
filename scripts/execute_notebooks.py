"""Execute every notebook and save copies with outputs for verification.

Usage:
    .venv/bin/python scripts/execute_notebooks.py
    .venv/bin/python scripts/execute_notebooks.py 00_START_HERE.ipynb data-science/unit5_scikit_learn.ipynb
"""

import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import ROOT, apply

apply()
NOTEBOOKS = ROOT / "notebooks"
REPORTS = ROOT / "reports" / "executed"


def execute(name):
    source = NOTEBOOKS / name
    if not source.exists():
        return f"MISSING {name}"

    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(source.parent)}},
    )
    destination = REPORTS / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
        nbformat.write(notebook, destination)
        return f"OK      {name}"
    except Exception:
        failed = destination.with_name(f"{source.stem}_FAILED.ipynb")
        nbformat.write(notebook, failed)
        error = traceback.format_exc().strip().splitlines()[-1]
        return f"FAILED  {name}: {error} (see {failed})"


def main():
    names = sys.argv[1:] or [
        str(path.relative_to(NOTEBOOKS))
        for path in sorted(NOTEBOOKS.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in path.parts
    ]
    results = [execute(name) for name in names]
    print("\n".join(results))
    print(f"\n{sum(line.startswith('OK') for line in results)}/{len(results)} notebooks executed")
    return 0 if all(line.startswith("OK") for line in results) else 1


if __name__ == "__main__":
    sys.exit(main())
