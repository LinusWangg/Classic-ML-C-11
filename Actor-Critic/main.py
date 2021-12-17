import gym
from Actor import *
from Critic import *

env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
actor = Actor(N_STATES, N_ACTIONS)
critic = Critic(N_STATES, N_ACTIONS)

critic.target_net.load_state_dict(critic.eval_net.state_dict())
actor.target.load_state_dict(actor.policy.state_dict())

for i in range(5000):
    s = env.reset()
    ep_r = 0
    critic.iter += 1
    while True:
        env.render()
        a = actor.select_action(s)

        s_, r, done, info = env.step(a)

        if done:
            r = -20
        #x, x_dot, theta, theta_dot = s_
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #r = r1 + r2

        V_eval = critic.eval_net(s)
        V_target = r + critic.GAMMA * critic.target_net(s_)
        TD_error = V_target - V_eval

        ep_r += r
        actor.learn(TD_error)
        critic.learn(V_eval, V_target)
        if done:
            print('Ep: ', i,
                '| Ep_r: ', round(ep_r, 2))
        
        if done:
            break
        s = s_