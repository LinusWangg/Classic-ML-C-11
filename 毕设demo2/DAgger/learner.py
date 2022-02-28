from tkinter.tix import Tree
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):
    ## 学习者
    def __init__(self, n_features, n_actions):
        super(Learner, self).__init__()
        self.layer1 = nn.Linear(n_features, 64)
        self.layer1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.layer2 = nn.Linear(64, n_actions)
        self.layer2.weight.data.normal_(0, 0.1) # initialization of FC1

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.softmax(x)
        return x