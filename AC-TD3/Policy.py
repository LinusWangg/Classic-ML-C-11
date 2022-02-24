import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(n_features, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1)

   # 连续
    def forward(self, x, a_bound):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = torch.mul(x, a_bound)
        return actions

    #离散
    #def forward(self, x):
    #    x = self.fc1(x)
    #    x = F.relu(x)
    #    x = self.out(x)
    #    return F.sigmoid(x)