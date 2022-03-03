from gzip import READ
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
    def __init__(self, n_features, n_actions, lr=0.02):
        self.n_features = n_features
        self.n_actions = n_actions
        self.expert = Expert(n_features, n_actions)
        parameters = torch.load("毕设demo2/parameters/"+game_name+"_parameters.pth.tar")
        self.expert.load_state_dict(parameters[game_name+'_Eval'])
        self.learner = Learner(n_features, n_actions)
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.BCELoss()
        self.ExpPool = ExperiencePool(n_features, 2000, 3)

    def train(self, batch_size):
        #states = torch.from_numpy(np.array(states))
        #actions = torch.from_numpy(np.array(actions))
        #dataDagger = TensorDataset(states, actions)
        #trainData = DataLoader(dataset=dataDagger, batch_size=batch_num, shuffle=True)
        total_loss = 0
        batch_data = self.ExpPool.sample(batch_size, self.learner, "MaxEntropy")
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
        return total_loss

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

def main():
    env = gym.make(game_name)
    env = env.unwrapped
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
    pipeline = DAgger_Pipeline(n_features, n_actions)
    RENDER = False
    for epoch in range(epoch_num):
        s = env.reset()
        ep_r = 0
        done = False
        step = 0
        while True:
            if step%50==0:
                print(epoch, step)
            if RENDER:
                env.render()
            a = pipeline.learner_action(s, EPSILON, n_actions)
            pipeline.ExpPool.add(s)

            s_, r, done, info = env.step(a)

            ep_r += r
            if done and pipeline.ExpPool.is_build:
                print('Ep: ', epoch,
                    '| Ep_r: ', round(ep_r, 2))
                #RENDER = True

            if done:
                break
            
            s = s_
            step += 1
        
        if pipeline.ExpPool.is_build:
            total_loss = pipeline.train(batch_num)
            #print(total_loss)


if __name__ == '__main__':
    main()
    

