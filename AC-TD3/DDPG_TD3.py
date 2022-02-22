import gym
from numpy import dtype
from Policy import *
from Qvalue import *

env = gym.make('CartPole-v0')
env = env.unwrapped
#连续
#N_ACTIONS = env.action_space.shape[0]
#离散
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MEMORY_CAPACITY = 3000
BATCH_SIZE = 32
TAU = 0.01
LR = 1e-3
GAMMA = 0.9

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

    def select_action(self, s):
        #连续
        #s = torch.from_numpy(s).float().unsqueeze(0)
        #actions = self.actor_eval.forward(s, a_bound)
        #return actions[0].detach()
        #离散
        s = torch.from_numpy(s).float().unsqueeze(0)
        probs = self.actor_eval.forward(s)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def learn(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES+1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        b_amat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        index = torch.Tensor(b_a, dtype = int)
        b_amat[index] = 1.

        q_eval = self.critic_eval(b_s, b_a)
        a_loss = -torch.mean(q_eval)
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        a_next = self.actor_target(b_s_)
        a_next = np.array(a_next)
        argmax_a = np.argmax(a_next, axis=0)
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
#连续
#var = 3
#a_bound = env.action_space.high
#a_low_bound = env.action_space.low
EP_STEPS = 200
RENDER = False

for i in range(5000):
    s = env.reset()
    ep_r = 0
    for j in range(200):
        if RENDER: 
            env.render()
        #连续
        #a = ddpg.select_action(s, a_bound)
        #a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
        #离散
        a = ddpg.select_action(s)

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
            ddpg.learn()

        if j == 199 or done:
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r, 2))
            if ep_r > -600:
                #RENDER = True
                pass
        
        if done:
            break
        s = s_