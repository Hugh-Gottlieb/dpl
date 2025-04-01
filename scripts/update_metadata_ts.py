import os
import json

root = r"C:\Users\z5485746\Documents\test_flight_5B"

for acquisition in os.listdir(root):
    metadata_path = os.path.join(root, acquisition, acquisition + "_metadata.json")
    if not os.path.exists(metadata_path):
        continue
    print(metadata_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    for img in metadata["images"]:
        if img["time"] > 1e11:
            img["time"] = img["time"] * 1e-3
    for transition in metadata["transitions"]:
        if transition["time"] > 1e11:
            transition["time"] = transition["time"] * 1e-3
    with open(metadata_path, "w") as f:
        metadata = json.dump(metadata, f, indent=4, ensure_ascii=False, sort_keys=True)