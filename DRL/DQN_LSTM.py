from os import times
from typing import AsyncIterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import retro
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)) #220*316*5
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2)) #110*158*5
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=7)) #104*152*10
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2)) #52*76*10
        self.conv.add_module("relu_2", torch.nn.ReLU())
        self.conv.add_module("conv_3", torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)) #48*72*20
        self.conv.add_module("dropout_3", torch.nn.Dropout())
        self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2)) #24*36*20
        self.conv.add_module("relu_3", torch.nn.ReLU())
        self.conv.add_module("conv_4", torch.nn.Conv2d(in_channels=20, out_channels=24, kernel_size=5)) #20*32*24
        self.conv.add_module("dropout_4", torch.nn.Dropout())
        self.conv.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2)) #10*16*24
        self.conv.add_module("relu_4", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(10*16*24, 1000))
        self.fc.add_module("relu_5", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(1000, 100))
        self.fc.add_module("sigmoid2", torch.nn.Sigmoid())

    def forward(self, X):
        X = X.reshape(-1, 3, 224, 320)
        X = self.conv.forward(X)
        X = X.view(-1, 10*16*24)
        return self.fc.forward(X)

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cnn = ConvNet().to(self.device)
        self.lstm = nn.LSTM(N_STATES, 50, batch_first=True).to(self.device)
        self.fc = nn.Linear(50, N_ACTIONS).to(self.device)
        
    def forward(self, x, timestep = 1):
        x = torch.FloatTensor(x)
        x = self.cnn(x.to(self.device)).reshape(1, timestep, 100)
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

    def choose_action(self, x, EPSILON, N_ACTIONS):

        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x).to("cpu")
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
            actions = [0 for i in range(N_ACTIONS)]
            actions[action] = 1

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action
            actions = [0 for i in range(N_ACTIONS)]
            actions[action] = 1
        
        return action, actions

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

            b_s = torch.unsqueeze(torch.FloatTensor(b_memory[0][index:index+self.timestep]), 0)
            b_a = torch.LongTensor(b_memory[1][0][index:index+self.timestep]).reshape(self.timestep, 1).to(self.device)
            b_r = torch.FloatTensor(b_memory[1][1][index:index+self.timestep]).reshape(self.timestep, 1).to(self.device)
            b_s_ = torch.unsqueeze(torch.FloatTensor(b_memory[2][index:index+self.timestep]), 0)

            q_eval = self.eval_net.forward(b_s, self.timestep).gather(1, b_a)
            q_next = self.target_net.forward(b_s_, self.timestep).detach()
            q_target = b_r + GAMMA*q_next.max(1)[0].view(self.timestep, 1)
            loss = criteria(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

# Hyper Parameters
BATCH_SIZE = 4
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 10
MAX_EPS = 300
env = retro.make('Airstriker-Genesis')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = 100

dqn = DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR)

for i in range(10000):
    if(i%10==0):
        torch.save({'eval_cnn':dqn.eval_net.cnn.state_dict(), 'lstm':dqn.eval_net.lstm.state_dict(), 'fc':dqn.eval_net.fc.state_dict()}, 'parameters.pth.tar')
    s = env.reset()
    ep_r = 0
    s_episode = []
    a_episode = []
    r_episode = []
    s__episode = []
    for j in range(MAX_EPS):
        #env.render()
        s_episode.append(s)
        a, a_s = dqn.choose_action(s, EPSILON, N_ACTIONS)
        a_episode.append(a)

        s_, r, done, info = env.step(a_s)
        
        s__episode.append(s_)

        #x, x_dot, theta, theta_dot = s_
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #r = r1 + r2
        r_episode.append(r)


        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
            if done or j==MAX_EPS-1:
                print('Ep: ', i,
                      '| Ep_r: ', round(ep_r, 2))
        
        if done or j==MAX_EPS-1:
            print(i)
            dqn.store_transition(s_episode, a_episode, r_episode, s__episode, MEMORY_CAPACITY)
            break
        s = s_