import os

# Take a file's `__file__` and return the path to the "config.json" in the same directory
def get_config_path(file):
    return os.path.join(os.path.dirname(os.path.realpath(file)), "config.json")
