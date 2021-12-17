import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    ##  离散空间采用了 softmax policy 来参数化策略
    def __init__(self, n_features, n_actions):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(n_features, 30)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(30, n_actions)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x * 2
        return actions

class Actor(object):

    def __init__(self, n_features, n_actions, lr=0.001):
        self.policy = Policy(n_features, n_actions)
        self.target = Policy(n_features, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actions = self.policy.forward(state)
        return actions[0].detach()



