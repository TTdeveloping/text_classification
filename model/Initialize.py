import torch
import numpy as np
import torch.nn as nn

def init_embedding(input_embedding,seed=666):
    """
    初始化embedding层的权重
    :param input_embedding:
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -scope, scope)

def init_linear(input_linear,seed=1337):
    """
    初始化全连接层权重
    :param input_linear:
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

