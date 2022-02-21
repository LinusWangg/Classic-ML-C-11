from typing import AsyncIterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import threading

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 18)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(18, 18)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(18, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        
    def forward(self, x):
        x = x.cuda()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
    
class DQN(object):
    def __init__(self, MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS).cuda(), Net(N_STATES, N_ACTIONS).cuda()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, EPSILON, N_ACTIONS):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action
        
        return action

    def store_transition(self, s, a , r, s_, MEMORY_CAPACITY):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a.cuda())
        q_next = self.target_net(b_s_).detach()
        q_target = b_r.cuda() + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target).cuda()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def make_DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR):
    return DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR)