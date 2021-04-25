import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, inputlayer, hiddenlayer):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(inputlayer, hiddenlayer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenlayer, 1)

    def forward(self, x):

        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        y_pred = torch.sigmoid(out)

        return y_pred

model = NeuralNet(inputlayer=3, hiddenlayer=5)
loss = nn.BCELoss()

Y = torch.tensor([2])
y_pred = torch.tensor([[2,3,1]])

print(loss(Y, y_pred))



