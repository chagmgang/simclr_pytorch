import torch
import torchvision

import torch.nn as nn
import numpy as np

class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        backbone = list(torchvision.models.resnet50(pretrained=True).children())[:-3]
        self.backbone = nn.ModuleList(backbone)

    def forward(self, x):
        for backbone in self.backbone:
            x = backbone(x)
        return x

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.backbone = Backbone()

        ### output shape of self.backbone = [batch_size, 1024, 7, 7]
        self.backbone_height = 7
        self.backbone_width = 7
        self.backbone_channel = 1024

        self.gap = nn.AvgPool2d((self.backbone_height, self.backbone_width))

        self.l1 = nn.Linear(self.backbone_channel, self.backbone_channel)
        self.l2 = nn.Linear(self.backbone_channel, self.backbone_channel)

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.backbone(x)
        h = self.gap(h)
        h = torch.reshape(h, (batch_size, -1))
        
        x = torch.relu(self.l1(h))
        x = self.l2(x)

        return h, x

if __name__ == '__main__':

    data = np.random.rand(32, 3, 112, 112)
    data = torch.as_tensor(data, dtype=torch.float32)
    backbone = Model()

    x = backbone(data)
    print(x.shape)