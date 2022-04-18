import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):
    ## 学习者
    def __init__(self, n_features, n_actions):
        super(Learner, self).__init__()
        self.fc1 = nn.Linear(n_features, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = x.cuda()
        x = self.fc1(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x.cpu()