import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import time

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=3, stride=1,
                            padding=0, bias=False),
                  nn.ReLU(),
                  nn.Conv2d(out_size, out_size, kernel_size=3, stride=1,
                            padding=0, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#up phase convtranspose2d
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUpConv(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UnetUpConv, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=3,
                            stride=1, padding=0, bias=False),
                  nn.ReLU(),
                  nn.Conv2d(out_size, out_size, kernel_size=3,
                            stride=1, padding=0, bias=False)
                  ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
def center_crop_concat(input_x, input_y):
    width = input_x.shape[2]-input_y.shape[2]
    start_point = int(width/2)
    x_crop = input_x[:, :, start_point:start_point +
                     input_y.shape[2], start_point:start_point+input_y.shape[2]]
    out = torch.cat((x_crop, input_y), 1)
    return out