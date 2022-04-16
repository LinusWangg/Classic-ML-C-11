import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):
    ## 学习者
    def __init__(self, n_features, n_actions):
        super(Learner, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, x):
        x = x.cuda()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.cpu()