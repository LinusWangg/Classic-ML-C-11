import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from Settings import ParaExchange, GAMMA

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        state_value = self.out(x)
        return state_value
    
class Critic(object):

    def __init__(self, n_features, n_actions, lr=1e-2):
        self.eval_net = Net(n_features, n_actions)
        self.target_net = Net(n_features, n_actions)
        self.iter = 0
        self.paraExchange = ParaExchange
        self.GAMMA = GAMMA
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()