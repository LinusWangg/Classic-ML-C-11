import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from Settings import ParaExchange, GAMMA

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 30)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_ACTIONS, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        
    def forward(self, s, a):
        x = self.fc1(s)
        y = self.fc2(a)
        x = F.relu(x + y)
        action_value = self.out(x)
        return action_value
    
class Critic(object):

    def __init__(self, n_features, n_actions, lr=0.002):
        self.eval_net = Net(n_features, n_actions)
        self.target_net = Net(n_features, n_actions)
        self.iter = 0
        self.paraExchange = ParaExchange
        self.GAMMA = GAMMA
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()