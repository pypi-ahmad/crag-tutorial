"""Validate teaching structure; optionally execute and save both notebooks."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
for name in ("00_quick_check.ipynb", "01_crag_tutorial.ipynb", "02_naive_vs_crag_comparison.ipynb"):
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    count = 0
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        count += 1
        assert i and notebook.cells[i - 1].cell_type == "markdown", (name, i)
        for label in ("Motivation", "Paper mapping", "Next cell", "Failure", "Read the output"):
            assert label in notebook.cells[i - 1].source, (name, i, label)
        compile(cell.source, f"{name}:cell{i + 1}", "exec")
    print(f"{name}: {count} code cells; all teaching checks passed.", flush=True)
    if args.execute:
        def progress(cell_index, **kwargs):
            print(f"{name}: executing cell {cell_index + 1}", flush=True)
        client = NotebookClient(notebook, timeout=3600, kernel_name="crag-tutorial", resources={"metadata": {"path": str(ROOT / "notebooks")}}, on_cell_start=progress)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        print(f"{name}: execution complete; outputs saved.", flush=True)
