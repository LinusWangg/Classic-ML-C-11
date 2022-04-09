import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    ##  离散空间采用了 softmax policy 来参数化策略
    def __init__(self, n_features, n_actions):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(n_features, 50)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x
        return actions.cpu()

