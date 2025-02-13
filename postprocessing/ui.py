from dpl_common.config import Config
from dpl_common.helpers import get_config_path

config = Config(get_config_path(__file__))
print(config.get_config())
print(config.hash())