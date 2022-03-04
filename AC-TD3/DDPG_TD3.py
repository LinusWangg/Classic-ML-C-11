import gym
from numpy import dtype
from Policy import *
from Qvalue import *

env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
#连续
N_ACTIONS = env.action_space.shape[0]
#离散
#N_ACTIONS = 1
N_STATES = env.observation_space.shape[0]
MEMORY_CAPACITY = 3000
BATCH_SIZE = 32
TAU = 0.01
LR = 1e-4
GAMMA = 0.99

class DDPG(object):
    def __init__(self):
        self.actor_eval = Policy(N_STATES, N_ACTIONS).cuda()
        self.actor_target = Policy(N_STATES, N_ACTIONS).cuda()
        self.critic_eval1 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.critic_eval2 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.critic_target1 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.critic_target2 = Qvalue(N_STATES, N_ACTIONS).cuda()
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + N_ACTIONS + 1), dtype=np.float32)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target1.load_state_dict(self.critic_eval1.state_dict())
        self.critic_target2.load_state_dict(self.critic_eval2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor_eval.parameters(), LR)
        self.critic_opt1 = torch.optim.Adam(self.critic_eval1.parameters(), LR)
        self.critic_opt2 = torch.optim.Adam(self.critic_eval2.parameters(), LR)
        self.critic_loss1 = nn.MSELoss().cuda()
        self.critic_loss2 = nn.MSELoss().cuda()
        self.critic_update = 0
        self.policy_delay = 3

    def store_transition(self, s, a , r, s_, MEMORY_CAPACITY):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def select_action(self, s, a_bound):
        #连续
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions = self.actor_eval.forward(s.cuda(), a_bound.cuda())
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
        b_r = torch.FloatTensor(b_memory[:, N_STATES+N_ACTIONS:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        #b_amat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        #index = b_a.long()
        #for i in range(BATCH_SIZE):
        #    b_amat[i, index[i]] = 1.

        # Target Policy Smoothing Regularization
        a_next1 = self.actor_target(b_s_.cuda(), a_bound.cuda())
        a_next1 = torch.clip(torch.normal(a_next1, 0.1), a_low_bound.cuda(), a_bound.cuda())
        #argmax_a = torch.argmax(a_next, axis=1)
        #a_next_mat = torch.FloatTensor(torch.zeros(BATCH_SIZE, N_ACTIONS))
        #for i in range(BATCH_SIZE):
        #    a_next_mat[i, argmax_a[i]] = 1.
        q_next1 = self.critic_target1(b_s_.cuda(), a_next1)
        q_next2 = self.critic_target2(b_s_.cuda(), a_next1)
        q_target1 = b_r.cuda() + GAMMA * torch.min(q_next1, q_next2)
        q_eval1 = self.critic_eval1(b_s.cuda(), b_a.cuda())
        q_eval2 = self.critic_eval2(b_s.cuda(), b_a.cuda())
        q_loss1 = self.critic_loss1(q_target1, torch.min(q_eval1, q_eval2))
        self.critic_opt1.zero_grad()
        q_loss1.backward()
        self.critic_opt1.step()

        #a_next2 = self.actor_target(b_s_, a_bound)
        #a_next2 = torch.clip(torch.normal(a_next2, 0.1), a_low_bound.item(), a_bound.item())
        a_next2 = a_next1.detach()
        q_next1 = self.critic_target1(b_s_.cuda(), a_next2)
        q_next2 = self.critic_target2(b_s_.cuda(), a_next2)
        q_target2 = b_r.cuda() + GAMMA * torch.min(q_next1, q_next2)
        q_eval1 = self.critic_eval1(b_s.cuda(), b_a.cuda())
        q_eval2 = self.critic_eval2(b_s.cuda(), b_a.cuda())
        q_loss2 = self.critic_loss2(q_target2, torch.min(q_eval1, q_eval2))
        self.critic_opt2.zero_grad()
        q_loss2.backward()
        self.critic_opt2.step()
        self.critic_update += 1

        # Actor_Delay
        if self.critic_update % self.policy_delay == 0:
            # 这边还要算一遍action是为了构造出梯度
            a = self.actor_eval(b_s.cuda(), a_bound.cuda())
            q_eval1 = self.critic_eval1(b_s.cuda(), a.cuda())
            q_eval2 = self.critic_eval2(b_s.cuda(), a.cuda())
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


ddpg = DDPG()
var = 2.0
a_bound = torch.Tensor(env.action_space.high)
a_low_bound = torch.Tensor(env.action_space.low)
EP_STEPS = 200
RENDER = False

for i in range(500000):
    s = env.reset()
    ep_r = 0
    if i%100 == 0:
        torch.save({'actor_eval':ddpg.actor_eval.state_dict(),
                    'actor_target':ddpg.actor_target.state_dict(),
                    'critic_eval1':ddpg.critic_eval1.state_dict(),
                    'critic_eval2':ddpg.critic_eval2.state_dict(),
                    'critic_target1':ddpg.critic_target1.state_dict(),
                    'critic_target2':ddpg.critic_target2.state_dict()}, "model.pk1")
    for j in range(400):
        if RENDER:
            env.render()
        a = ddpg.select_action(s, a_bound)
        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound).detach().numpy()[0]

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
        ddpg.store_transition(s, a, r, s_, MEMORY_CAPACITY)

        ep_r += r
        
        if ddpg.memory_counter > MEMORY_CAPACITY:
            #var *= 0.9995
            ddpg.learn()

        if done or j==399:
            var *= 0.995
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r, 2))
            if i > 500:
                RENDER = True
                #pass
        
        if done:
            break
        s = s_