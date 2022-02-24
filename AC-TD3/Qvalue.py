import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Qvalue(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Qvalue, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_ACTIONS, 50)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        
    def forward(self, s, a):
        x = self.fc1(s)
        y = self.fc2(a)
        x = F.relu(x + y)
        action_value = self.out(x)
        return action_value