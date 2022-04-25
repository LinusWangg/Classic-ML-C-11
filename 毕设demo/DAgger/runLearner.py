import gym
from learner import Learner
import torch
import numpy as np

def learner_action(learner, state):
    state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
    actions = learner.forward(state)
    actions = torch.mul(actions, a_bound)
    return actions[0]

def save_log(log_file, file_path):
    with open(file_path, "w") as f:
        f.write(str(log_file))

if __name__ == "__main__":
    game_name = ['LunarLanderContinuous-v2', 'MoutainCarContinuous-v0', 'Pendulum-v0']
    run_epoch = [100, 1000, 1000]
    mode = ["DisWeightSample", "Random", "LossPER", "DisSample", "MaxDisSample", "LossPredict"]
    eps = [i for i in range(0, 1000, 5)]
    rewards_log = {}
    for i in range(len(game_name)):
        env = gym.make(game_name[i])
        env = env.unwrapped
        n_actions = env.action_space.shape[0]
        n_features = env.observation_space.shape[0]
        a_low_bound = env.action_space.low
        a_bound = torch.Tensor(env.action_space.high)
        a_high = env.action_space.high
        n_maxstep = run_epoch[i]
        n_testtime = 5
        rewards_log[game_name[i]] = {}
        agent = Learner(n_features, n_actions).cuda()
        for m in mode:
            rewards_log[game_name[i]][m] = []
            for ep in eps:
                reward = []
                path = "models/"+game_name[i]+"/"+m+"/Dagger_"+str(ep)+".pth"
                param = torch.load(path)
                agent.load_state_dict(param)
                for j in range(n_testtime):
                    s = env.reset()
                    ep_r = 0
                    done = False
                    for k in range(n_maxstep):
                        a = learner_action(agent, s)
                        a = np.clip(a.detach().numpy(), a_low_bound, a_high)

                        s_, r, done, info = env.step(a)

                        ep_r += r
                        if done or k==n_maxstep-1:
                            reward.append(ep_r)
                            break
                        
                        s = s_

                    rewards_log[game_name[i]][m].append(reward)
                print(game_name[i], " ", m, " ", ep, " MEAN_R=", np.mean(reward))

        save_log(rewards_log, "log2-rewards.json")
        
