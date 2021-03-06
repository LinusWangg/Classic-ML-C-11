import gym
from Settings import *
from expert import Expert
from learner import Learner
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import itertools
import os

from ExpirencePool import ExperiencePool

class DAgger_Pipeline(object):
    
    def __init__(self, n_features, n_actions, a_bound, init_model, select_mode="Random", lr=1e-3):
        self.n_features = n_features
        self.n_actions = n_actions
        self.a_bound = torch.Tensor(a_bound)
        self.expert = Expert(n_features, n_actions).cuda()
        parameters = torch.load("毕设demo/parameters/Ant-v2_parameters.pth.tar")
        self.expert.load_state_dict(parameters['actor_eval'])
        self.learner = Learner(n_features, n_actions).cuda()
        self.learner.load_state_dict(init_model.state_dict())
        self.optim = torch.optim.Adam(self.learner.parameters(), lr)
        self.loss = nn.MSELoss()
        self.select_mode = select_mode
        self.lamda = 0.15
        self.lr = lr
        self.ExpPool = ExperiencePool(n_features, 1000, 7, select_mode)

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
            batch_data = self.ExpPool.sample2Dagger(batch_size, self.learner)
        elif self.select_mode == "LossPER":
            batch_data, idxs, weight = self.ExpPool.sample2Dagger(batch_size, self.learner)
        states = torch.from_numpy(batch_data).to(torch.float32)
        #self.ExpPool.toDaggerMem(batch_data, self.expert_action(states).numpy())
        for i in range(1):
            #for s, a in zip(states, actions):
            #batch_data, expert_a = self.ExpPool.sample(batch_size)
            #expert_a = torch.from_numpy(expert_a).to(torch.float64)
            states = torch.from_numpy(batch_data).to(torch.float32)
            expert_a = self.expert_action(states).to(torch.float64)
            actions = self.learner_action(states.float()).to(torch.float64)
            #expert_a = expert_a.to(torch.float64)
            if self.select_mode == "DisWeightSample":
                loss = self.loss(actions, expert_a)
                actNet_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                yhat_loss = []
                for s in states:
                    ex_a = self.expert_action(s).detach().to(torch.float64)
                    lr_a = self.learner(s.float()).detach().to(torch.float64)
                    loss = nn.MSELoss()
                    y_hat = loss(lr_a, ex_a).item()
                    yhat_loss.append([y_hat])
                selectNet_loss += self.ExpPool.LossPredTrain(batch_data, torch.FloatTensor(yhat_loss))

            elif self.select_mode == "LossPredict":
                total_loss = 0
                loss1 = self.loss(actions, expert_a)
                actNet_loss += loss1.item()
                l_hat = self.ExpPool.LossPred.pred(states)
                loss2 = []
                for i in range(0, len(expert_a), 2):
                    j = i+1
                    loss = nn.MSELoss()
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
                    ex_a = self.expert_action(s).detach().to(torch.float64)
                    lr_a = self.learner(s.float()).detach().to(torch.float64)
                    loss = nn.MSELoss()
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

    def learner_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.learner.forward(state)
        actions = torch.mul(actions, self.a_bound)
        return actions[0]

    def expert_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        actions = self.expert.forward(state)
        actions = torch.mul(actions, self.a_bound)
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
    var = 0.1
    n_maxstep = 1000
    n_testtime = 1
    n_testtime2 = 5
    pipeline = DAgger_Pipeline(n_features, n_actions, a_bound, init_model, select_mode)
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
                a = np.clip(np.random.normal(a.detach().numpy(), var), a_low_bound, a_bound)

                pipeline.ExpPool.add(s)

                s_, r, done, info = env.step(a)

                ep_r += r
                if done or j==n_maxstep-1:
                    mean_r += ep_r
                    break
                
                s = s_
        mean_r /= n_testtime
        if epoch % 5 == 0:
            path = 'models2/'+game_name+'/'+select_mode
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(pipeline.learner.state_dict(), 'models2/'+game_name+'/'+select_mode+'/'+'Dagger_'+str(epoch)+'.pth')
        print('Mode: ', select_mode, 'Ep: ', epoch, '| Ep_r: ', round(mean_r, 2))
        actNet_loss = 0
        selectNet_loss = 0
        
        if pipeline.ExpPool.is_build:
            print("Updating", end=" ")
            actNet_loss, selectNet_loss = pipeline.train(batch_num)
            var *= 0.9995
            #rewards = []
            if start == 0:
                start = epoch
            
            #for i in range(n_testtime2):
            #    s = env.reset()
            #    ep_r = 0
            #    done = False
            #    for j in range(n_maxstep):
            #        if RENDER:
            #            env.render()
            #        a = pipeline.learner_action(s)
            #        a = a.numpy()

            #        s_, r, done, info = env.step(a)

            #        ep_r += r
            #        if done or j>=n_maxstep-1:
            #            rewards.append(ep_r)
            #            break
                    
            #        s = s_
            
            WRITER.add_scalar(game_name+'/Reward/'+select_mode, mean_r, epoch-start)
            WRITER.add_scalar(game_name+'/actNetLoss/'+select_mode, actNet_loss, epoch-start)
            WRITER.add_scalar(game_name+'/selectNetLoss/'+select_mode, selectNet_loss, epoch-start)
            WRITER.flush()
            reward_log.append(mean_r)
            #print(np.mean(rewards), end=" ")
            loss_log.append(actNet_loss)
    
    del pipeline
    
    return {"reward_log":reward_log}

def save_log(log_file, file_path):
    with open(file_path, "w") as f:
        f.write(str(log_file))

if __name__ == '__main__':
    np.random.seed(1)
    init_model = Learner(111, 8)
    select_mode = ["DisWeightSample", "Random", "LossPER", "DisSample", "MaxDisSample", "LossPredict"]
    log = {}
    for mode in select_mode:
        log[mode] = main(mode, init_model)
    save_log(log, "log5-"+game_name+".json")
    
    

