import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

class Actor(nn.Module):
    def __init__(self, n_features, n_actions, action_bound):
        super(Actor,self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(n_features, 128)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.mu = nn.Linear(128, n_actions)
        self.mu.weight.data.normal_(0, 0.1) # initilizaiton of OUT
        self.sigma = nn.Linear(128, n_actions)
        self.sigma.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.mu(x)
        mu = self.action_bound * torch.tanh(mu)
        sigma = self.sigma(x)
        sigma = F.softplus(sigma)
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, n_features):
        super(Critic, self).__init__()
        self.n_features = n_features

        self.fc1 = nn.Linear(n_features, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(128, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x

class AC_PPO(object):
    def __init__(self, n_features, n_actions, action_bound, lr=0.0001, gamma=0.9):
        super(AC_PPO, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.Actor_old = Actor(n_features, n_actions, action_bound)
        self.Actor_now = Actor(n_features, n_actions, action_bound)
        self.Critic = Critic(n_features)
        self.lr = lr
        self.gamma = gamma
        self.eps = 0.2

        self.Actor_optim = torch.optim.Adam(self.Actor_now.parameters(), lr=lr)
        self.Critic_optim = torch.optim.Adam(self.Critic.parameters(), lr=lr)

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        mu, sigma = self.Actor_now.forward(s)
        distribution = torch.distributions.Normal(mu, sigma)
        action = distribution.sample()
        return np.clip(action, -self.action_bound, self.action_bound)

    def learn(self, b_s, b_a, b_r):
        discount_reward = []
        Advantage = 0
        for r in b_r[::-1]:
            Advantage = r + self.gamma * Advantage
            discount_reward.append(Advantage)
        bs = torch.FloatTensor(b_s)
        b_vs = self.Critic(bs)
        discount_reward.reverse()
        discount_reward = torch.FloatTensor(discount_reward)
        
        mu, sigma = self.Actor_now(bs)
        pi = torch.distributions.Normal(mu, sigma)
        mu_old, sigma_old = self.Actor_old(bs)
        pi_old = torch.distributions.Normal(mu_old, sigma_old)
        ratio = torch.exp(pi.log_prob(torch.FloatTensor(b_a)) - pi_old.log_prob(torch.FloatTensor(b_a)))
        loss1 = ratio * discount_reward
        actor_loss = -torch.mean(torch.min(loss1, torch.clamp(ratio, 1-self.eps, 1+self.eps)*(discount_reward-b_vs)))
        
        self.Actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.Actor_optim.step()

        critic_loss = torch.mean(discount_reward ** 2)
        critic_loss = nn.MSELoss()
        loss = critic_loss(discount_reward, b_vs)
        self.Critic_optim.zero_grad()
        loss.backward()
        self.Critic_optim.step()


env = gym.make('Pendulum-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
BATCH_SIZE = 32
a_bound = env.action_space.high
ppo = AC_PPO(N_STATES, N_ACTIONS, 2)
EP_STEPS = 200
RENDER = False

for i in range(5000):
    b_s = []
    b_r = []
    b_a = []
    s = env.reset()
    ep_r = 0
    for j in range(200):
        if RENDER: 
            env.render()
        b_s.append(s)
        a = ppo.choose_action(s)
        b_a.append(a)
        s_, r, done, info = env.step(a)
        b_r.append((r+8)/8)
        #if done:
        #    r = -20
        #x, x_dot, theta, theta_dot = s_
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #r = r1 + r2

        ep_r += r
        
        if j % BATCH_SIZE == 0 or done or j == 199:
            b_s = np.array(b_s)
            b_r = np.array(b_r)
            b_a = np.array(b_a)
            for k in range(10):
                ppo.learn(b_s, b_a, b_r)
            b_s = []
            b_r = []
            b_a = []

        if j == 199 or done:
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r.item(), 2))
            if ep_r > -600:
                RENDER = True
            break
        
        s = s_