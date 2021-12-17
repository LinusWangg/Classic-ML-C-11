import gym
from Actor import *
from Critic import *

env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
TAU = 0.01

class DDPG(object):
    def __init__(self):
        self.actor = Actor(N_STATES, N_ACTIONS)
        self.critic = Critic(N_STATES, N_ACTIONS)
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2))
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
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.critic.eval_net(b_s).gather(1, b_a)
        a_loss = -torch.mean(q_eval)
        a_next = self.actor.target(b_s_)
        m = torch.distributions.Categorical(a_next)
        a_next = m.sample()
        a_next = a_next.reshape((BATCH_SIZE, 1))
        q_next = self.critic.target_net(b_s_).gather(1, a_next)
        q_target = b_r + GAMMA * q_next
        q_loss = self.critic.loss_func(q_eval, q_target)

        self.actor.optimizer.zero_grad()
        a_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()

        for x in self.actor.target.state_dict().keys():
            eval('self.actor.target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor.target.' + x + '.data.add_(TAU*self.actor.policy.' + x + '.data)')
        for x in self.critic.target_net.state_dict().keys():
            eval('self.critic.target_net.' + x + '.data.mul_((1-TAU))')
            eval('self.critic.target_net.' + x + '.data.add_(TAU*self.critic.eval_net.' + x + '.data)')



ddpg = DDPG()

for i in range(5000):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = ddpg.actor.select_action(s)

        s_, r, done, info = env.step(a)

        #if done:
        #    r = -20
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        ddpg.store_transition(s, a, r, s_, MEMORY_CAPACITY)

        ep_r += r
        
        if ddpg.memory_counter > MEMORY_CAPACITY:
            ddpg.learn()
            if done:
                print('Ep: ', i,
                      '| Ep_r: ', round(ep_r, 2))
        
        if done:
            break
        s = s_