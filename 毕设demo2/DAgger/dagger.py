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
        parameters = torch.load("毕设demo2/parameters/"+game_name+"_parameters.pth.tar")
        self.expert.load_state_dict(parameters[game_name+'_Eval'])
        self.learner = Learner(n_features, n_actions)
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.MSELoss()

    def train(self, states, actions):
        states = torch.from_numpy(np.array(states))
        actions = torch.from_numpy(np.array(actions))
        dataDagger = TensorDataset(states, actions)
        trainData = DataLoader(dataset=dataDagger, batch_size=batch_num, shuffle=True)
        for i in range(5):
            for s, a in trainData:
                expert_a = self.expert_action(s)
                a = self.learner.forward(s).to(torch.float64)
                expert_a = expert_a.to(torch.float64).requires_grad_()
                loss = self.loss(a, expert_a)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def learner_action(self, state, EPSILON, ENV_A_SHAPE, N_ACTIONS):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.learner.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        return action

    def expert_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.expert.forward(state)
        return actions[0].detach()

def main():
    env = gym.make(game_name)
    env = env.unwrapped
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
    pipeline = DAgger_Pipeline(n_features, n_actions)
    for epoch in range(epoch_num):
        states = []
        actions = []
        s = env.reset()
        ep_r = 0
        done = False
        while True:
            env.render()
            a = pipeline.learner_action(s, EPSILON, ENV_A_SHAPE, n_actions)

            s_, r, done, info = env.step(a)

            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            states.append(s)
            actions.append(a)

            ep_r += r
            if done:
                print('Ep: ', epoch,
                    '| Ep_r: ', round(ep_r, 2))
                break
            
            s = s_
        pipeline.train(states, actions)

if __name__ == '__main__':
    main()
    

