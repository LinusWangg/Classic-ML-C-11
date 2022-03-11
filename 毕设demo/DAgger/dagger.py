from asyncio.windows_utils import pipe
from gzip import READ
from select import select
import gym
from transformers import pipeline
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
    
    def __init__(self, n_features, n_actions, init_model, select_mode="Random", low_bound=-2, high_bound=2, gap=0.5, lr=0.002):
        self.n_features = n_features
        self.n_actions = n_actions
        self.expert = Expert(n_features, n_actions)
        parameters = torch.load("毕设demo/parameters/model.pk1")
        self.expert.load_state_dict(parameters['actor_eval'])
        self.learner = Learner(n_features, n_actions)
        self.learner.load_state_dict(init_model.state_dict())
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.MSELoss()
        self.ExpPool = ExperiencePool(n_features, 10000, 3)
        self.select_mode = select_mode
        self.gap = gap
        self.low_bound = low_bound - gap/2
        self.high_bound = high_bound + gap/2

    def train(self, batch_size):
        #states = torch.from_numpy(np.array(states))
        #actions = torch.from_numpy(np.array(actions))
        #dataDagger = TensorDataset(states, actions)
        #trainData = DataLoader(dataset=dataDagger, batch_size=batch_num, shuffle=True)
        actNet_loss = 0
        selectNet_loss = 0
        batch_data = self.ExpPool.sample(batch_size, self.learner, self.select_mode)
        states = torch.from_numpy(batch_data)
        for i in range(5):
            #for s, a in zip(states, actions):
            expert_a = self.expert_action(states).to(torch.float64)
            actions = self.learner.forward(states.float()).to(torch.float64)
            #expert_a = expert_a.to(torch.float64)
            loss = self.loss(actions, expert_a)
            actNet_loss += loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.select_mode == "LossPredict":
                yhat_loss = []
                for s in states:
                    ex_a = self.expert_action(s).detach().to(torch.float64).item()
                    low = self.low_bound

                    ex_ = 0 #替死
                    while low < ex_a:
                        low += self.gap
                        if low > ex_a:
                            break
                        ex_ += 1
                    ex_a_ = torch.zeros(1)
                    ex_a_[0] = ex_
                    ex_a_ = ex_a_.to(torch.long)

                    lr_a = self.learner(s.float()).detach().to(torch.float64).item()
                    lr_a_ = []
                    low = self.low_bound
                    dir = 1.1
                    step = 1
                    while low < self.high_bound:
                        low += self.gap
                        if low > lr_a and lr_a > low-self.gap:
                            lr_a_.append(5)
                            dir = 0.9
                            step *= dir
                            continue
                        lr_a_.append(step)
                        step *= dir
                    lr_a_ = torch.Tensor(lr_a_)
                    lr_a_ = torch.softmax(lr_a_, dim=0)
                    lr_a_ = lr_a_.unsqueeze(0)
                    loss = nn.CrossEntropyLoss()
                    y_hat = loss(lr_a_, ex_a_).item()
                    yhat_loss.append([y_hat])
                selectNet_loss += self.ExpPool.LossPredTrain(batch_data, torch.FloatTensor(yhat_loss))

        return actNet_loss / 5, selectNet_loss / 5

    def learner_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.learner.forward(state)
        return actions[0].detach()

    def expert_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.expert.forward(state)
        return actions[0].detach()

def main(select_mode, init_model):
    reward_log = []
    loss_log = []
    env = gym.make(game_name)
    env = env.unwrapped
    n_actions = env.action_space.shape[0]
    n_features = env.observation_space.shape[0]
    a_low_bound = env.action_space.low
    a_bound = env.action_space.high
    var = 0.5
    n_maxstep = 500
    n_testtime = 5
    pipeline = DAgger_Pipeline(n_features, n_actions, init_model, select_mode)
    pipeline.low_bound = a_low_bound[0] - pipeline.gap / 2
    pipeline.high_bound = a_bound[0] + pipeline.gap / 2
    RENDER = False
    start = 0
    for epoch in range(epoch_num):
        mean_r = 0
        for time in range(n_testtime):
            s = env.reset()
            ep_r = 0
            done = False
            for j in range(n_maxstep):
                if RENDER:
                    env.render()
                a = pipeline.learner_action(s)
                a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
                pipeline.ExpPool.add(s)

                s_, r, done, info = env.step(a)

                ep_r += r
                if done or j>=n_maxstep-1:
                    mean_r += ep_r
                    break
                
                s = s_
        mean_r /= n_testtime
        print('Mode: ', select_mode, 'Ep: ', epoch, '| Ep_r: ', round(mean_r, 2))
        actNet_loss = 0
        selectNet_loss = 0
        if pipeline.ExpPool.is_build:
            actNet_loss, selectNet_loss = pipeline.train(batch_num)
            var *= 0.9995
            if start == 0:
                start = epoch
            WRITER.add_scalar('Reward/'+select_mode, mean_r, epoch-start)
            WRITER.add_scalar('actNetLoss/'+select_mode, actNet_loss, epoch-start)
            WRITER.add_scalar('selectNetLoss/'+select_mode, selectNet_loss, epoch-start)
            WRITER.flush()
        reward_log.append(mean_r)
        loss_log.append(actNet_loss)
    
    del pipeline
    
    return {"reward_log":reward_log, "loss_log":loss_log}

def save_log(log_file, file_path):
    with open(file_path, "w") as f:
        f.write(str(log_file))

if __name__ == '__main__':
    np.random.seed(1)
    init_model = Learner(3, 1)
    select_mode = ["LossPredict", "Random"]
    log = {}
    for mode in select_mode:
        log[mode] = main(mode, init_model)
    save_log(log, "log.txt")
    
    

