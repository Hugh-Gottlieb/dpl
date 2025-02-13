from dpl_common.config import Config, get_config_path

config = Config(get_config_path(__file__))
print(config.get_config())