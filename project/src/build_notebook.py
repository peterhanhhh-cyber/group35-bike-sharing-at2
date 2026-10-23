"""Convert a 'percent-cell' Python script to a Jupyter notebook.

Cell markers:
    # %% [markdown]   -> markdown cell
    # %%              -> code cell

Usage:
    python build_notebook.py source.py output.ipynb [--execute] [--timeout 3600]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

_MD_PREFIX = re.compile(r"^# ?")


def strip_markdown_prefix(line: str) -> str:
    """Percent-cell markdown must be commented in .py; remove that comment."""
    return _MD_PREFIX.sub("", line, count=1)


def split_cells(text: str) -> list[tuple[str, str]]:
    cells = []
    current_type = None
    buffer: list[str] = []

    def flush():
        nonlocal buffer, current_type
        if current_type is not None:
            source = "\n".join(buffer).strip("\n")
            if source.strip():
                cells.append((current_type, source))
        buffer = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            current_type = "markdown" if "[markdown]" in stripped else "code"
        else:
            if current_type is None:
                current_type = "code"
            if current_type == "markdown":
                line = strip_markdown_prefix(line)
            buffer.append(line)
    flush()
    return cells


def build(source: Path, output: Path) -> nbformat.NotebookNode:
    cells = split_cells(source.read_text(encoding="utf-8"))
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "markdown"
        else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python (PyTorch CPU)",
        "language": "python",
        "name": "torch-cpu",
    }
    nb.metadata["language_info"] = {"name": "python"}
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output)
    return nb


def execute(nb: nbformat.NotebookNode, output: Path, timeout: int = 3600) -> None:
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="torch-cpu",
        resources={"metadata": {"path": str(output.parent)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(nb, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="percent-cell Python script")
    parser.add_argument("output", type=Path, help="notebook path to create")
    parser.add_argument(
        "--execute", action="store_true", help="execute the generated notebook"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="cell execution timeout in seconds (default: 3600)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    output = args.output
    nb = build(source, output)
    if args.execute:
        execute(nb, output, timeout=args.timeout)
        print(f"executed -> {output}")
    else:
        print(f"built -> {output}")


if __name__ == "__main__":
    main()
