from os import times
from typing import AsyncIterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(N_STATES, 50, batch_first=True).to(self.device)
        self.fc = nn.Linear(50, N_ACTIONS).to(self.device)
        
    def forward(self, x, batch_size):
        out, (h_n, c_n) = self.lstm(x.to(self.device))
        h = out[:, :, :]
        h = h.reshape(out.shape[0]*out.shape[1], out.shape[2])
        actions_value = self.fc(h)
        return actions_value
    
class DQN(object):
    def __init__(self, MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.memory_counter = 0
        #self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2))
        self.memory = []
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.timestep = 8

    def choose_action(self, x, EPSILON, ENV_A_SHAPE, N_ACTIONS):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).reshape(1, 1, N_STATES)

        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x, 1).to("cpu")
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        return action

    def store_transition(self, s, a , r, s_, MEMORY_CAPACITY):
        transition = (s, [a, r], s_)
        index = self.memory_counter % MEMORY_CAPACITY
        if self.memory_counter < MEMORY_CAPACITY:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.memory_counter += 1

    def learn(self, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(len(self.memory), BATCH_SIZE)
        for i in sample_index:
            criteria = nn.MSELoss()
            b_memory = self.memory[i]
            size = len(b_memory[0])
            index = np.random.randint(0, size-self.timestep+1)

            b_s = torch.FloatTensor(b_memory[0][index:index+self.timestep]).reshape(1, self.timestep, N_STATES)
            b_a = torch.LongTensor(b_memory[1][0][index:index+self.timestep]).reshape(self.timestep, 1).to(self.device)
            b_r = torch.FloatTensor(b_memory[1][1][index:index+self.timestep]).reshape(self.timestep, 1).to(self.device)
            b_s_ = torch.FloatTensor(b_memory[2][index:index+self.timestep]).reshape(1, self.timestep, N_STATES)

            q_eval = self.eval_net.forward(b_s, 1).gather(1, b_a)
            q_next = self.target_net.forward(b_s_, 1).detach()
            q_target = b_r + GAMMA*q_next.max(1)[0].view(self.timestep, 1)
            loss = criteria(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 50
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

dqn = DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR)

for i in range(10000):
    s = env.reset()
    ep_r = 0
    s_episode = []
    a_episode = []
    r_episode = []
    s__episode = []
    while True:
        #env.render()
        s_episode.append(s)
        a = dqn.choose_action(s, EPSILON, ENV_A_SHAPE, N_ACTIONS)
        a_episode.append(a)

        s_, r, done, info = env.step(a)
        
        s__episode.append(s_)

        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        r_episode.append(r)


        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
            if done:
                print('Ep: ', i,
                      '| Ep_r: ', round(ep_r, 2))
        
        if done:
            dqn.store_transition(s_episode, a_episode, r_episode, s__episode, MEMORY_CAPACITY)
            break
        s = s_