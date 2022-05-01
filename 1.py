import matplotlib.pyplot as plt
import numpy as np
import json

def draw_pic(log_path, game_name, eps):
    with open(log_path, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
        plt.title('LunarLanderContinuous-v2环境不同算法得到的智能体性能比较')
        plt.rcParams['font.sans-serif'] = ['STSong']
        plt.xlabel('更新步长')
        plt.ylabel('累计奖赏')
        x = [i for i in range(200)]
        color = 0
        legend = []
        rewards = json_data[game_name]
        for key, data in rewards.items():
            reward_log = []
            legend.append(key)
            reward = data
            for j in range(len(data)):
                reward = np.mean(data[j])
                reward_log.append(reward)
            plt.plot(x, reward_log)

        plt.legend(legend)
        plt.show()
            
draw_pic("log8-rewards.json", "LunarLanderContinuous-v2", 0)