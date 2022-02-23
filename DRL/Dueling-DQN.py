from typing import AsyncIterable
from numpy.lib.function_base import _quantile_is_valid
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.outA = nn.Linear(50, N_ACTIONS)
        self.outA.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_STATES, 50)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.outV = nn.Linear(50, 1)
        self.outV.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        a_value = self.outA(x1)
        x2 = self.fc2(x)
        x2 = F.relu(x2)
        state_value = self.outV(x2)
        q_value = state_value + (a_value - torch.mean(a_value))
        return q_value
    
class DQN(object):
    def __init__(self, MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, EPSILON, ENV_A_SHAPE, N_ACTIONS):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < EPSILON:
            q_value = self.eval_net.forward(x)
            action = torch.max(q_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
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

        q_eval = self.eval_net.forward(b_s).gather(1, b_a)
        q_next = self.target_net.forward(b_s_).detach()
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

dqn = DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR)

for i in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        #env.render()
        a = dqn.choose_action(s, EPSILON, ENV_A_SHAPE, N_ACTIONS)

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_, MEMORY_CAPACITY)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
            if done:
                print('Ep: ', i,
                      '| Ep_r: ', round(ep_r, 2))
        
        if done:
            break
        s = s_