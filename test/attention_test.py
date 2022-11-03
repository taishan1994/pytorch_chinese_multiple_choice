import sys
sys.path.append("..")

import torch

from models.layer.Attention import Attention
from config.config_parser import create_config

config = create_config("../config/sfks.config")
gpu_list = []
attention = Attention(config, gpu_list)

option = torch.randn((8, 16, 300))
question = torch.randn((8, 25, 300))
x_atten, y_atten, a_ = attention(option, question)
print(x_atten.shape)
print(y_atten.shape)
print(a_.shape)