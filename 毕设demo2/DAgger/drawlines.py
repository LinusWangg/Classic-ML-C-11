import os
from pyecharts.charts import Line
import json
from Settings import *
from pyecharts import options as opts

def draw_pic(log_path, eps):
    with open(log_path, 'r', encoding='utf-8') as fp:
        line = Line(init_opts=opts.InitOpts(width='850px', height='650px'))
        json_data = json.load(fp)
        x = [i for i in range(1000)]
        color = ['#33CCCC', '#1A2D3E', '#87CEFA', '#111111', '#00FF7F', '#FFFF00', '#FF4500', '#FF3030', '#00FF00', '#800000']
        i = 0
        line.add_xaxis(x)
        for key, data in json_data.items():
            reward_log = []
            reward = data['reward_log'][0]
            for j in range(len(data['reward_log'])):
                reward = eps*reward + (1-eps)*data['reward_log'][j]
                reward_log.append(reward)
            line.add_yaxis(key, reward_log, 
                is_connect_nones=True,# 连接空值
                is_symbol_show=False,  # 默认显示
                symbol='pin',  # 标记形状
                symbol_size=10,
                is_smooth=False,  #是否平滑曲线
                color=color[i],  # 系列颜色
                linestyle_opts=opts.LineStyleOpts(width=2,type_='solid')
            )
            i += 1
        line.render('log-'+game_name+'.html')


if __name__ == '__main__':
    draw_pic("log-"+game_name+".json", 0.99)
