import json
from pathlib import Path

import torch

base = Path(__file__).resolve().parent
config = json.loads((base / "training_config.json").read_text())
model = torch.load(base / "final_model.pt", map_location="cpu")
print("Config keys:", sorted(config.keys()))
print("Model type:", type(model).__name__)
print("Device:", config.get("device"))
print("Reload verification complete.")
