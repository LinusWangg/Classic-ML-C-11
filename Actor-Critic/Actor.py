import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    ##  离散空间采用了 softmax policy 来参数化策略
    def __init__(self, n_features, n_actions):
        super(Policy,self).__init__()
        self.affline1 = nn.Linear(n_features,128)
        self.dropout = nn.Dropout(p=0.6)
        self.affline2 = nn.Linear(128,n_actions)  # 两种动作

        self.saved_log_probs = 0
        self.rewards = []

    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores,dim=1)

class Actor(object):

    def __init__(self, n_features, n_actions, lr=1e-2):
        self.policy = Policy(n_features, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs = m.log_prob(action)
        return action.item()

    def learn(self, TD_error):
        self.optimizer.zero_grad()
        policy_loss = -self.policy.saved_log_probs * TD_error
        policy_loss.backward(retain_graph=True)
        self.optimizer.step()



