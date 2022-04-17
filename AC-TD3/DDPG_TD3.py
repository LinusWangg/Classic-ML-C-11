import gym
from numpy import dtype
from Policy import *
from Qvalue import *

env = gym.make('Ant-v2')
env = env.unwrapped
#连续
N_ACTIONS = env.action_space.shape[0]
#离散
#N_ACTIONS = 1
N_STATES = env.observation_space.shape[0]
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
TAU = 0.005
LR = 1e-4
GAMMA = 0.99

class DDPG(object):
    def __init__(self, a_bound):
        self.actor_eval = Policy(N_STATES, N_ACTIONS).cuda()
        self.actor_target = Policy(N_STATES, N_ACTIONS).cuda()
        self.critic_eval1 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.critic_eval2 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.critic_target1 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.critic_target2 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + N_ACTIONS + 2), dtype=np.float32)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target1.load_state_dict(self.critic_eval1.state_dict())
        self.critic_target2.load_state_dict(self.critic_eval2.state_dict())
        #param = torch.load("Ant-v2_parameters.pth.tar")
        #self.actor_eval.load_state_dict(param['actor_eval'])
        #self.actor_target.load_state_dict(param['actor_target'])
        #self.critic_eval1.load_state_dict(param['critic_eval1'])
        #self.critic_eval2.load_state_dict(param['critic_eval2'])
        #self.critic_target1.load_state_dict(param['critic_target1'])
        #self.critic_target2.load_state_dict(param['critic_target2'])
        self.actor_opt = torch.optim.Adam(self.actor_eval.parameters(), LR)
        self.critic_opt1 = torch.optim.Adam(self.critic_eval1.parameters(), LR)
        self.critic_opt2 = torch.optim.Adam(self.critic_eval2.parameters(), LR)
        self.critic_loss1 = nn.MSELoss()
        self.critic_loss2 = nn.MSELoss()
        self.critic_update = 0
        self.policy_delay = 3
        self.a_bound = a_bound
        self.policy_noise = 0.2*a_bound
        self.noise_clip = 0.5*a_bound

    def store_transition(self, s, a, r, done, s_, MEMORY_CAPACITY):
        transition = np.hstack((s, a, r, done, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def select_action(self, s, a_bound):
        #连续
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions = self.actor_eval.forward(s, a_bound)
        return actions.cpu().detach().numpy()
        #离散
        #s = torch.from_numpy(s).float().unsqueeze(0)
        #act = self.actor_eval.forward(s)
        #action = np.clip(np.random.normal(act.detach().numpy(), VAR), 0, 1)
        #return act, round(action.item())

    def learn(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES+N_ACTIONS])
        b_r = torch.FloatTensor(b_memory[:, N_STATES+N_ACTIONS:N_STATES+N_ACTIONS+1])
        b_done = torch.FloatTensor(b_memory[:, N_STATES+N_ACTIONS+1:N_STATES+N_ACTIONS+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        #b_amat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        #index = b_a.long()
        #for i in range(BATCH_SIZE):
        #    b_amat[i, index[i]] = 1.

        # Target Policy Smoothing Regularization
        noise = (torch.randn_like(b_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        a_next1 = self.actor_target(b_s_, self.a_bound)
        a_next1 = (
            a_next1 + noise
        ).clamp(-self.a_bound, self.a_bound)

        #a_next1 = torch.clip(torch.normal(a_next1, 0.2*a_bound), a_low_bound, a_bound)
        #argmax_a = torch.argmax(a_next, axis=1)
        #a_next_mat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        #for i in range(BATCH_SIZE):
        #    a_next_mat[i, argmax_a[i]] = 1.
        q_next1 = self.critic_target1(b_s_, a_next1)
        q_next2 = self.critic_target2(b_s_, a_next1)
        q_target1 = b_r + (1 - b_done) * GAMMA * torch.min(q_next1, q_next2)
        q_eval1 = self.critic_eval1(b_s, b_a)
        q_eval2 = self.critic_eval2(b_s, b_a)
        q_loss1 = self.critic_loss1(q_target1, torch.min(q_eval1, q_eval2))
        self.critic_opt1.zero_grad()
        q_loss1.backward()
        self.critic_opt1.step()

        #a_next2 = self.actor_target(b_s_, a_bound)
        #a_next2 = torch.clip(torch.normal(a_next2, 0.1), a_low_bound.item(), a_bound.item())
        a_next2 = a_next1.detach()
        q_next1 = self.critic_target1(b_s_, a_next2)
        q_next2 = self.critic_target2(b_s_, a_next2)
        q_target2 = b_r + (1 - b_done) * GAMMA * torch.min(q_next1, q_next2)
        q_eval1 = self.critic_eval1(b_s, b_a)
        q_eval2 = self.critic_eval2(b_s, b_a)
        q_loss2 = self.critic_loss2(q_target2, torch.min(q_eval1, q_eval2))
        self.critic_opt2.zero_grad()
        q_loss2.backward()
        self.critic_opt2.step()
        self.critic_update += 1

        # Actor_Delay
        if self.critic_update % self.policy_delay == 0:
            # 这边还要算一遍action是为了构造出梯度
            a = self.actor_eval(b_s, a_bound)
            q_eval1 = self.critic_eval1(b_s, a)
            q_eval2 = self.critic_eval2(b_s, a)
            a_loss = -torch.mean(torch.min(q_eval1, q_eval2))
            self.actor_opt.zero_grad()
            a_loss.backward()
            self.actor_opt.step()

            for x in self.actor_target.state_dict().keys():
                eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
                eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
            for x in self.critic_target1.state_dict().keys():
                eval('self.critic_target1.' + x + '.data.mul_((1-TAU))')
                eval('self.critic_target1.' + x + '.data.add_(TAU*self.critic_eval1.' + x + '.data)')
            for x in self.critic_target2.state_dict().keys():
                eval('self.critic_target2.' + x + '.data.mul_((1-TAU))')
                eval('self.critic_target2.' + x + '.data.add_(TAU*self.critic_eval2.' + x + '.data)')



var = 0.15
a_bound = torch.Tensor(env.action_space.high)
a_low_bound = torch.Tensor(env.action_space.low)
ddpg = DDPG(a_bound)
EP_STEPS = 200
RENDER = False

for i in range(1000000):
    s = env.reset()
    ep_r = 0
    if i % 60 == 0:
        torch.save({'actor_eval':ddpg.actor_eval.state_dict(),
                    'actor_target':ddpg.actor_target.state_dict(),
                    'critic_eval1':ddpg.critic_eval1.state_dict(),
                    'critic_eval2':ddpg.critic_eval2.state_dict(),
                    'critic_target1':ddpg.critic_target1.state_dict(),
                    'critic_target2':ddpg.critic_target2.state_dict()}, "Ant-v2_parameters.pth.tar")
    if ddpg.memory_counter > MEMORY_CAPACITY:
        print("Updating", end=" ")
    for j in range(1000):
        #env.render()
        a = ddpg.select_action(s, a_bound)
        a = np.random.normal(a, var)
        a = np.clip(a, a_low_bound, a_bound).detach().numpy()[0]

        s_, r, done, info = env.step(a)

        #if done:
        #    r = -20
        #x, x_dot, theta, theta_dot = s_
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #r = r1 + r2
        #if r <= -100:
        #    done = True
        #    r = -1
        ddpg.store_transition(s, a, r, done, s_, MEMORY_CAPACITY)

        ep_r += r
        
        if ddpg.memory_counter > MEMORY_CAPACITY:
            ddpg.learn()

        if done or j==999:
            var *= 0.999
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r, 2))
            if i > 50:
                #RENDER = True
                pass
            break
        s = s_