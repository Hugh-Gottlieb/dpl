import json

class Config:

    def __init__(self):
        self.config = {}

    def __init__(self, path):
        self.read(path)

    def read(self, path):
        with open(path) as f:
            self.config = json.load(f)

    def write(self, path):
        with open(path, "w") as f:
            json.dump(self.config, f)

    def hash(self):
        return hash(json.dumps(self.config, sort_keys=True))

    def get_config(self):
        return self.config

    def get(self, key):
        return self.config[key]

    def set(self, key, value):
        self.config[key] = value