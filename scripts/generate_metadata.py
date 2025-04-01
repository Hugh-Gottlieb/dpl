import os
import json

# NOTE: could also read gps / gimbal information from exif?
def generate_metadata(path, name):
    metadata_path = os.path.join(path, name + "_metadata.json")
    old_metadata_path = os.path.join(path, "metadata.csv")
    if not os.path.exists(old_metadata_path):
        print(f"Skipping {metadata_path}: couldn't find old metadata.csv file")
        return
    metadata = {} # omit gps, gimbal and transitions
    metadata["camera"] = {}
    metadata["images"] = []
    with open(old_metadata_path, "r") as f:
        lines = f.readlines()
    started_files = False
    for line in lines:
        line = line.strip()
        if line == "":
            started_files = True
            continue
        key, val = line.split(",")
        if started_files:
            name = key.split("/")[-1][:-4]
            name_without_number = "_".join(name.split("_")[:-1])
            name_number = int(name.split("_")[-1])
            name = f"{name_without_number}_{name_number:03}"
            metadata["images"].append({"name": name, "time": float(val), "pl_state": "unknown"})
        elif key == "exposure":
            metadata["camera"]["exposure_time"] = float(val)
        elif key == "fps":
            metadata["camera"]["fps"] = float(val)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, sort_keys=True)

root = r"C:\Users\z5485746\Documents\glenrowan"
for flight in os.listdir(root):
    flight_path = os.path.join(root, flight)
    for acquisition in os.listdir(flight_path):
        acquisition_path = os.path.join(flight_path, acquisition)
        if not os.path.isdir(acquisition_path):
            continue
        generate_metadata(acquisition_path, acquisition)