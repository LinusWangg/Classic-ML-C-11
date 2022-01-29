import gym
from Actor import *
from Critic import *

env = gym.make('Pendulum-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01

class DDPG(object):
    def __init__(self):
        self.actor = Actor(N_STATES, N_ACTIONS)
        self.critic = Critic(N_STATES, N_ACTIONS)
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2), dtype=np.float32)
        self.actor.target.load_state_dict(self.actor.policy.state_dict())
        self.critic.target_net.load_state_dict(self.critic.eval_net.state_dict())

    def store_transition(self, s, a , r, s_, MEMORY_CAPACITY):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES+1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        a = self.actor.policy(b_s)
        q_eval = self.critic.eval_net(b_s, a)
        a_loss = -torch.mean(q_eval)
        self.actor.optimizer.zero_grad()
        a_loss.backward()
        self.actor.optimizer.step()

        a_next = self.actor.target(b_s_)
        q_next = self.critic.target_net(b_s_, a_next)
        q_target = b_r + 0.9 * q_next
        q_eval = self.critic.eval_net(b_s, b_a)
        q_loss = self.critic.loss_func(q_target, q_eval)
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()

        for x in self.actor.target.state_dict().keys():
            eval('self.actor.target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor.target.' + x + '.data.add_(TAU*self.actor.policy.' + x + '.data)')
        for x in self.critic.target_net.state_dict().keys():
            eval('self.critic.target_net.' + x + '.data.mul_((1-TAU))')
            eval('self.critic.target_net.' + x + '.data.add_(TAU*self.critic.eval_net.' + x + '.data)')

    def save_model(self):
        torch.save({'Pendulum_ActorEval':self.actor.policy.state_dict(),
        'Pendulum_ActorTarget':self.actor.target.state_dict(),
        'Pendulum_CriticEval':self.critic.eval_net.state_dict(),
        'Pendulum_CriticTarget':self.critic.target_net.state_dict()}, '\parameters.pth.tar')




ddpg = DDPG()
var = 3
a_bound = env.action_space.high
a_low_bound = env.action_space.low
EP_STEPS = 200
RENDER = False

for i in range(100):
    s = env.reset()
    ep_r = 0
    for j in range(200):
        if RENDER: 
            env.render()
        a = ddpg.actor.select_action(s)
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
            var *= 0.9995 # decay the exploration controller factor
            ddpg.learn()

        if j == 199 or done:
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r, 2))
        
        if done:
            break
        s = s_

ddpg.save_model()