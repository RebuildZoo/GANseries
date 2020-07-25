import torch.nn as nn
import torch.nn.init as init


def init_xavier(pNet):
    for m in pNet.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def init_kaiming(pNet):
    for m in pNet.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

def init_norm(pNet):
    for m in pNet.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.normal_(m.weight, mean = 0.0, std = 0.02)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight, mean = 1.0, std = 0.02)
            init.constant_(m.bias, 0)


