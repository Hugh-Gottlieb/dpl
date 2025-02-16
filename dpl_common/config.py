import json
import os

# Take a file's `__file__` and return the abs path to the "config.json" in the same directory
def get_config_path(file: str) -> str:
    return os.path.join(os.path.dirname(os.path.realpath(file)), "config.json")

class Config:

    def __init__(self, path: str):
        self.read(path)

    # NOTE - Qt can get in the way of this getting called correctly
    def __del__(self):
        self.write()

    def read(self, path: str):
        self.path = path
        with open(path) as f:
            self.config = json.load(f)

    def write(self):
        with open(self.path, "w") as f:
            json.dump(self.config, f, indent=4)

    def same_hash(self, other_config: dict) -> int:
        return self.__hash_config_dict(other_config) == self.__hash_config_dict(self.config)

    def __hash_config_dict(self, config: dict):
        return hash(json.dumps(config, sort_keys=True))

    def get_config(self) -> dict:
        return self.config

    def get(self, key: str) -> object:
        return self.config[key]

    def set(self, key: str, value: object):
        self.config[key] = value