from typing import AsyncIterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import cv2

class Net(nn.Module):
    def __init__(self, N_CHANNELS, N_ACTIONS):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(N_CHANNELS, 32, 8, 4)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(512, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        
    def forward(self, x):
        x = x.cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        actions_value = self.out(x)
        return actions_value.cpu()
    
class DDQN(object):
    def __init__(self, MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS).cuda(), Net(N_STATES, N_ACTIONS).cuda()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 4*80*80*2 + 3))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, EPSILON, ENV_A_SHAPE, N_ACTIONS):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        return action

    def store_transition(self, s, a , r, s_, done, MEMORY_CAPACITY):
        s = s.reshape(4*80*80)
        s_ = s_.reshape(4*80*80)
        transition = np.hstack((s, [a, r, done], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES].reshape(BATCH_SIZE, 4, 80, 80))
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_done = torch.FloatTensor(b_memory[:, N_STATES+2:N_STATES+3])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:].reshape(BATCH_SIZE, 4, 80, 80))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_maxa = self.eval_net(b_s_).detach().argmax(dim=-1).view(BATCH_SIZE, 1)
        q_next = self.target_net(b_s_).gather(1, q_maxa)
        q_target = b_r + (1 - b_done)*GAMMA*q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ColorMat2Binary(self, state):
        height = state.shape[0]
        width = state.shape[1]
        nchannel = state.shape[2]
 
        sHeight = int ( height * 0.5 )#210变成105
        sWidth = 80 #定义CNN输入的宽度是80

        state_gray = cv2.cvtColor( state, cv2.COLOR_BGR2GRAY)#将RGB转换成灰度图像
        _,state_binary = cv2.threshold( state_gray, 5, 255, cv2.THRESH_BINARY )
        state_binarySmall = cv2.resize( state_binary, (sWidth,sHeight), interpolation = cv2.INTER_AREA )
        cnn_inputImg = state_binarySmall[25:, :]
        cnn_inputImg = cnn_inputImg.reshape((80, 80))
        return cnn_inputImg

    def save_model(self):
        torch.save({'eval':self.eval_net.state_dict(),
        'target':self.target_net.state_dict()}, 'Breakout-v4_parameters.pth.tar')

# Hyper Parameters
BATCH_SIZE = 256
LR = 1e-3                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 20000
env = gym.make('Breakout-v4')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = 4*80*80
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

dqn = DDQN(MEMORY_CAPACITY, 4, N_ACTIONS, LR)

for i in range(400000):
    if i%60 == 0:
        dqn.save_model()
    s = env.reset()
    s = dqn.ColorMat2Binary(s)
    s_shadow = np.stack((s, s, s, s), axis=0)
    ep_r = 0
    while True:
        #env.render("human")
        a = dqn.choose_action(s_shadow, EPSILON, ENV_A_SHAPE, N_ACTIONS)

        s_, r, done, info = env.step(a)
        s_ = dqn.ColorMat2Binary(s_).reshape(1, 80, 80)
        #x, x_dot, theta, theta_dot = s_
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #r = r1 + r2
        next_s_shadow = np.append(s_, s_shadow[:3, :, :], axis=0)

        dqn.store_transition(s_shadow, a, r, next_s_shadow, done, MEMORY_CAPACITY)
        s_shadow = next_s_shadow
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
            if done:
                print('Ep: ', i,
                      '| Ep_r: ', round(ep_r, 2))
        
        if done:
            break
        s = s_