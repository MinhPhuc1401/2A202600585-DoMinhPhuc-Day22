import json
from pathlib import Path

notebook_path = Path("colab/Lab22_DPO_T4.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Cell 51 source:")
print("".join(nb["cells"][51]["source"]))
