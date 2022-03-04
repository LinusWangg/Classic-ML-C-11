from gzip import READ
from select import select
import gym
from Settings import *
from expert import Expert
from learner import Learner
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from ExpirencePool import ExperiencePool

class DAgger_Pipeline(object):
    def __init__(self, n_features, n_actions, init_model, select_mode="Random", lr=0.02):
        self.n_features = n_features
        self.n_actions = n_actions
        self.expert = Expert(n_features, n_actions)
        parameters = torch.load("毕设demo2/parameters/"+game_name+"_parameters.pth.tar")
        self.expert.load_state_dict(parameters[game_name+'_Eval'])
        self.learner = Learner(n_features, n_actions)
        self.learner.load_state_dict(init_model.state_dict())
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.BCELoss()
        self.ExpPool = ExperiencePool(n_features, 3000, 3)
        self.select_mode = select_mode

    def train(self, batch_size):
        #states = torch.from_numpy(np.array(states))
        #actions = torch.from_numpy(np.array(actions))
        #dataDagger = TensorDataset(states, actions)
        #trainData = DataLoader(dataset=dataDagger, batch_size=batch_num, shuffle=True)
        total_loss = 0
        batch_data = self.ExpPool.sample(batch_size, self.learner, self.select_mode)
        states = torch.from_numpy(batch_data)
        for i in range(5):
            #for s, a in zip(states, actions):
            expert_a = self.expert_action(states)
            actions = self.learner.forward(states.float()).to(torch.float64)
            expert_a = expert_a.to(torch.float64)
            loss = self.loss(actions, expert_a)
            total_loss += loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return total_loss / 5

    def learner_action(self, state, EPSILON, N_ACTIONS):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.learner.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action
        
        return action

    def expert_action(self, state):
        state = torch.from_numpy(np.array(state)).float().squeeze(0)
        actions = self.expert.forward(state)
        actions = actions.detach()
        max_act = torch.max(actions, dim = 1).indices
        max = torch.zeros(actions.shape)
        for i in range(max.shape[0]):
            max[i, max_act[i]] = 1.
        return max

def main(select_mode, init_model):
    reward_log = []
    loss_log = []
    env = gym.make(game_name)
    env = env.unwrapped
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    n_maxstep = 200
    n_testtime = 5
    pipeline = DAgger_Pipeline(n_features, n_actions, init_model, select_mode)
    RENDER = False
    for epoch in range(epoch_num):
        mean_r = 0
        #if pipeline.ExpPool.is_build:
            #print(1)
        for time in range(n_testtime):
            #if pipeline.ExpPool.is_build:
                #print(2)
            s = env.reset()
            ep_r = 0
            done = False
            for j in range(n_maxstep):
                if RENDER:
                    env.render()
                a = pipeline.learner_action(s, EPSILON, n_actions)
                pipeline.ExpPool.add(s)

                s_, r, done, info = env.step(a)

                ep_r += r
                if done or j>=n_maxstep-1:
                    mean_r += ep_r
                    break
                
                s = s_
        mean_r /= n_testtime
        print('Mode: ', select_mode, 'Ep: ', epoch, '| Ep_r: ', round(mean_r, 2))
        mean_loss = 0
        if pipeline.ExpPool.is_build:
            #print(3)
            mean_loss = pipeline.train(batch_num)
        reward_log.append(mean_r)
        loss_log.append(mean_loss)
    
    del pipeline
    
    return {"reward_log":reward_log, "loss_log":loss_log}

def save_log(log_file, file_path):
    with open(file_path, "w") as f:
        f.write(str(log_file))

if __name__ == '__main__':
    np.random.seed(1)
    init_model = Learner(4, 2)
    select_mode = ["Random", "MaxEntropy", "Density-Weighted", "maxDis2Center"]
    log = {}
    for mode in select_mode:
        log[mode] = main(mode, init_model)
    save_log(log, "log.txt")
    
    

