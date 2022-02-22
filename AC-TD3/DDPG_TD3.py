import gym
from numpy import dtype
from Policy import *
from Qvalue import *

env = gym.make('Pendulum-v1')
env = env.unwrapped
#连续
N_ACTIONS = env.action_space.shape[0]
#离散
#N_ACTIONS = 1
N_STATES = env.observation_space.shape[0]
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
TAU = 0.01
LR = 1e-3
GAMMA = 0.99

class DDPG(object):
    def __init__(self):
        self.actor_eval = Policy(N_STATES, N_ACTIONS)
        self.actor_target = Policy(N_STATES, N_ACTIONS)
        self.critic_eval = Qvalue(N_STATES, N_ACTIONS)
        self.critic_target = Qvalue(N_STATES, N_ACTIONS)
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2), dtype=np.float32)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor_eval.parameters(), LR)
        self.critic_opt = torch.optim.Adam(self.critic_eval.parameters(), LR)
        self.critic_loss = nn.MSELoss()

    def store_transition(self, s, a , r, s_, MEMORY_CAPACITY):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def select_action(self, s, a_bound):
        #连续
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions = self.actor_eval.forward(s, a_bound)
        return actions[0].detach()
        #离散
        #s = torch.from_numpy(s).float().unsqueeze(0)
        #act = self.actor_eval.forward(s)
        #action = np.clip(np.random.normal(act.detach().numpy(), VAR), 0, 1)
        #return act, round(action.item())

    def learn(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES+1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        #b_amat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        #index = b_a.long()
        #for i in range(BATCH_SIZE):
        #    b_amat[i, index[i]] = 1.

        # 这边还要算一遍action是为了构造出梯度
        a = self.actor_eval(b_s, a_bound)
        q_eval = self.critic_eval(b_s, a)
        a_loss = -torch.mean(q_eval)
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        a_next = self.actor_target(b_s_, a_bound)
        #argmax_a = torch.argmax(a_next, axis=1)
        #a_next_mat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        #for i in range(BATCH_SIZE):
        #    a_next_mat[i, argmax_a[i]] = 1.
        q_next = self.critic_target(b_s_, a_next)
        q_target = b_r + GAMMA * q_next
        q_eval = self.critic_eval(b_s, b_a)
        q_loss = self.critic_loss(q_target, q_eval)
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')


ddpg = DDPG()
var = 3
a_bound = env.action_space.high
a_low_bound = env.action_space.low
EP_STEPS = 200
RENDER = False

for i in range(5000):
    s = env.reset()
    ep_r = 0
    for j in range(200):
        if RENDER: 
            env.render()
        a = ddpg.select_action(s, a_bound)
        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)

        s_, r, done, info = env.step(a)

        #if done:
        #    r = -20
        #x, x_dot, theta, theta_dot = s_
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #r = r1 + r2
        ddpg.store_transition(s, a, r/10, s_, MEMORY_CAPACITY)

        ep_r += r
        
        if ddpg.memory_counter > MEMORY_CAPACITY:
            var *= 0.9995
            ddpg.learn()

        if done or j==199:
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r, 2))
            if ep_r > -600:
                #RENDER = True
                pass
        
        if done:
            break
        s = s_