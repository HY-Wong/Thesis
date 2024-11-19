import yaml

from typing import Dict


class Config:
	def __init__(self, config: Dict):
		for key, value in config.items():
			if isinstance(value, dict):
				value = Config(value)
			setattr(self, key, value)


def load_config(config_path: str):
	"""
	Load the model and training configuration file in YAML format.
	"""
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)

	config = Config(config)
	return config