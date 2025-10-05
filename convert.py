import nbformat

with open("notebook/fine_tuning.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if "metadata" in cell and "widgets" in cell.metadata:
        if "state" not in cell.metadata["widgets"]:
            cell.metadata["widgets"]["state"] = {}

with open("notebook/fine_tuning.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
