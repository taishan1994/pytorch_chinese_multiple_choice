import sys
sys.path.append("..")

from config.config_parser import create_config

config = create_config("../config/sfks.config")
print(config.sections())
print(config.options("train"))
print(config.get("train", "epoch"))
print(config.items("train"))
"""
    ['train']
    ['epoch']
    10
    [('epoch', '10')]
"""