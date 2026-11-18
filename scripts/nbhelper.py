"""Minimal stdlib-only helper for generating Jupyter notebooks in this project.

Use with the system Python: python3 scripts/build_<name>.py
No third-party packages are required to *create* a notebook.
"""

import json
from pathlib import Path


def md(text):
    """Create a markdown cell from a string."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    """Create an empty code cell from a string."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n"),
    }


def write_notebook(path, cells, kernel_name="python3"):
    """Write a list of cells to a .ipynb file."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (DS uv env)",
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {"name": "python", "version": "3.12.16"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {target} ({len(cells)} cells)")
