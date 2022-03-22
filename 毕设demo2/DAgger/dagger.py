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
import itertools

from ExpirencePool import ExperiencePool

class DAgger_Pipeline(object):
    def __init__(self, n_features, n_actions, init_model, select_mode="Random", n_clusters=1, lr=0.02):
        self.n_features = n_features
        self.n_actions = n_actions
        self.expert = Expert(n_features, n_actions)
        parameters = torch.load("毕设demo2/parameters/"+game_name+"_parameters.pth.tar")
        self.expert.load_state_dict(parameters[game_name+'_Eval'])
        self.learner = Learner(n_features, n_actions)
        self.learner.load_state_dict(init_model.state_dict())
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.CrossEntropyLoss()
        self.ExpPool = ExperiencePool(n_features, 10000, n_clusters, select_mode)
        self.select_mode = select_mode
        self.lamda = 0.15
        self.lr = lr

    def train(self, batch_size):
        #states = torch.from_numpy(np.array(states))
        #actions = torch.from_numpy(np.array(actions))
        #dataDagger = TensorDataset(states, actions)
        #trainData = DataLoader(dataset=dataDagger, batch_size=batch_num, shuffle=True)
        actNet_loss = 0
        selectNet_loss = 0
        batch_data = []
        idxs = []
        weight = []
        if self.select_mode != "LossPER":
            batch_data = self.ExpPool.sample(batch_size, self.learner)
        elif self.select_mode == "LossPER":
            batch_data, idxs, weight = self.ExpPool.sample(batch_size, self.learner)
        states = torch.from_numpy(batch_data).to(torch.float32)
        for i in range(5):
            #for s, a in zip(states, actions):
            expert_a = self.expert_action(states)
            actions = self.learner.forward(states.float()).to(torch.float64)
            #expert_a = expert_a.to(torch.float64)
            if self.select_mode == "LossPredict":
                total_loss = 0
                loss1 = self.loss(actions, expert_a)
                actNet_loss += loss1.item()
                l_hat = self.ExpPool.LossPred.pred(states)
                loss2 = []
                for i in range(0, len(expert_a), 2):
                    j = i+1
                    loss = nn.CrossEntropyLoss()
                    loss_i = loss(torch.unsqueeze(actions[i], 0), torch.unsqueeze(expert_a[i], 0))
                    loss_j = loss(torch.unsqueeze(actions[j], 0), torch.unsqueeze(expert_a[j], 0))
                    loss2.append(max(0, -torch.sign(loss_i-loss_j))*(l_hat[i]-l_hat[j]+1e-6))
                loss2 = torch.mean(torch.FloatTensor(loss2))
                selectNet_loss += loss2.item()
                total_loss = loss1 + self.lamda * loss2
                optim = torch.optim.Adam(itertools.chain(self.learner.parameters(), self.ExpPool.LossPred.lossNet.parameters()), lr=self.lr)
                optim.zero_grad()
                total_loss.backward()
                optim.step()
            
            elif self.select_mode == "LossPER":
                loss = self.loss(actions, expert_a)
                actNet_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                yhat_loss = []
                for s in states:
                    ex_a = self.expert_action(s.reshape(1, 1, s.shape[0])).detach()
                    lr_a = torch.unsqueeze(self.learner(s.float()).to(torch.float64), 0)
                    loss = nn.CrossEntropyLoss()
                    y_hat = loss(lr_a, ex_a).item()
                    yhat_loss.append([y_hat])
                selectNet_loss += self.ExpPool.LossPredTrain(batch_data, torch.FloatTensor(yhat_loss))
                
                self.ExpPool.update_SumTree(batch_num, idxs, states)
                
            else:
                loss = self.loss(actions, expert_a)
                actNet_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        return actNet_loss / 5, selectNet_loss / 5

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
        return max_act

def main(select_mode, init_model):
    reward_log = []
    loss_log = []
    env = gym.make(game_name)
    env = env.unwrapped
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    n_maxstep = 1000
    n_testtime = 5
    n_clusters = 8
    if select_mode == "LossPredict":
        n_clusters = 1
    pipeline = DAgger_Pipeline(n_features, n_actions, init_model, select_mode, n_clusters)
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
        actNet_loss = 0
        selectNet_loss = 0

        if pipeline.ExpPool.is_build:
            print("Updating", end=" ")
            actNet_loss, selectNet_loss = pipeline.train(batch_num)
            if start == 0:
                start = epoch
            WRITER.add_scalar(game_name+'/Reward/'+select_mode, mean_r, epoch-start)
            WRITER.add_scalar(game_name+'/actNetLoss/'+select_mode, actNet_loss, epoch-start)
            WRITER.add_scalar(game_name+'/selectNetLoss/'+select_mode, selectNet_loss, epoch-start)
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
    init_model = Learner(4, 2)
    select_mode = ["LossPER", "LossPredict", "Random", "MaxEntropy", "Density-Weighted"]
    log = {}
    for mode in select_mode:
        log[mode] = main(mode, init_model)
    save_log(log, "log-"+game_name+".json")
    
    

