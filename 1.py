import matplotlib.pyplot as plt
import numpy as np
import json

def draw_pic(log_path, game_name, eps):
    with open(log_path+game_name+'.json', 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
        fig = plt.figure(figsize=(16, 9))
        plt.title(game_name+'环境不同算法的学习速率比较', fontsize = 20)
        plt.tick_params(labelsize=18)
        plt.rcParams['font.sans-serif'] = ['STSong']
        plt.rcParams.update({'font.size': 18})
        plt.xlabel('使用样本量', fontsize = 20)
        plt.ylabel('累计奖赏', fontsize = 20)
        x = [i*32*5 for i in range(1, 801)]
        color = 0
        legend = []
        rewards = json_data
        t = 0
        for key, data in rewards.items():
            reward_log = []
            legend.append(key)
            reward = data['reward_log']
            r = reward[0]
            for j in range(800):
                r = r*eps + (1-eps)*reward[j]
                reward_log.append(r)
            plt.plot(x, reward_log)

        plt.legend(legend)
        #plt.show()
        fig.savefig("Pool/learn/"+game_name+".png")
            
draw_pic("log3-", "Ant-v2", 0.9)