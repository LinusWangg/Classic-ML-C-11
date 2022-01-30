import gym
from Settings import *
from expert import Expert
from learner import Learner
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class DAgger_Pipeline(object):
    def __init__(self, n_features, n_actions, lr=0.002):
        self.expert = Expert(n_features, n_actions)
        parameters = torch.load("毕设demo/parameters/"+game_name+"_parameters.pth.tar")
        self.expert.load_state_dict(parameters[game_name+'_ActorEval'])
        self.learner = Learner(n_features, n_actions)
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.MSELoss().requires_grad_()

    def train(self, states, actions):
        states = torch.from_numpy(np.array(states))
        actions = torch.from_numpy(np.array(actions))
        dataDagger = TensorDataset(states, actions)
        trainData = DataLoader(dataset=dataDagger, batch_size=batch_num, shuffle=True)
        for i in range(5):
            for s, a in trainData:
                expert_a = self.expert_action(s)
                a = a.to(torch.float64).requires_grad_()
                expert_a = expert_a.to(torch.float64).requires_grad_()
                loss = torch.mean((a-expert_a)**2)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def learner_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.learner.forward(state)
        return actions[0].detach()

    def expert_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.expert.forward(state)
        return actions[0].detach()

def main():
    env = gym.make(game_name)
    env = env.unwrapped
    n_actions = env.action_space.shape[0]
    n_features = env.observation_space.shape[0]
    pipeline = DAgger_Pipeline(n_features, n_actions)
    a_bound = env.action_space.high
    a_low_bound = env.action_space.low
    for epoch in range(epoch_num):
        states = []
        actions = []
        s = env.reset()
        ep_r = 0
        done = False
        for j in range(300):
            #env.render()
            a = pipeline.learner_action(s)
            a = np.clip(np.random.normal(a, 0.001), a_low_bound, a_bound)

            s_, r, done, info = env.step(a)

            ep_r += r
            states.append(s)
            actions.append(a)

            if j == 199 or done:
                print('Ep: ', epoch,
                    '| Ep_r: ', round(ep_r, 2))
            
            if done:
                break
            s = s_
        pipeline.train(states, actions)

if __name__ == '__main__':
    main()
    

