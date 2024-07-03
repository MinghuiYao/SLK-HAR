import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from SLKblock import  SLKBlock
from matplotlib import pyplot as plt
from torchstat import stat
from sparse_core import Masking, CosineDecay

'''SLK-Net'''
class SLK_Net(nn.Module):
    def __init__(self, train_shape, category):
        super(SLK_Net, self).__init__()
       
        self.layer = nn.Sequential(
            nn.Conv2d(1,64, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            SLKBlock(64,128,(57,1),(5,1)),
            nn.ReLU(),

            SLKBlock(128,256,(57,1),(5,1)),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(256*train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


   
   