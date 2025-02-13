import json

class Config:

    def __init__(self, path: str):
        self.read(path)

    def __del__(self):
        self.write(self.path)

    def read(self, path: str):
        self.path = path
        with open(path) as f:
            self.config = json.load(f)

    def write(self, path: str):
        with open(path, "w") as f:
            json.dump(self.config, f, indent=4)

    def hash(self) -> int:
        return hash(json.dumps(self.config, sort_keys=True))

    def get_config(self) -> dict:
        return self.config

    def get(self, key: str) -> object:
        return self.config[key]

    def set(self, key: str, value: object):
        self.config[key] = value