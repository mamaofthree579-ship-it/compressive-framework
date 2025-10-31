#!/usr/bin/env python3
import sys
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from pathlib import Path

# ---- Input notebook path ----
if len(sys.argv) < 2:
    print("Usage: execute_notebook.py <notebook_path>")
    sys.exit(1)

nb_path = Path(sys.argv[1])
if not nb_path.exists():
    print(f"❌ Notebook not found: {nb_path}")
    sys.exit(1)

print(f"Executing: {nb_path}")

# ---- Load notebook ----
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# ---- Ensure each cell has an ID ----
for cell in nb.cells:
    if "id" not in cell:
        cell["id"] = f"auto_{hash(cell['source']) & 0xffffffff:x}"

# ---- Execute notebook ----
try:
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
except CellExecutionError as e:
    print(f"❌ Error executing notebook: {e}")
    sys.exit(1)

# ---- Save executed notebook ----
output_path = nb_path.parent / f"{nb_path.stem}_executed.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Successfully executed: {output_path}")
