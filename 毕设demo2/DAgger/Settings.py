from tensorboardX import SummaryWriter
WRITER = SummaryWriter("./log-离散")
# eval to net iteration
ParaExchange = 100
# Critic GAMMA
GAMMA = 0.9
# GAME NAME
game_name = 'LunarLander-v2'
# Exp Pool Size
Exp_Size = 1000
# Epoch
epoch_num = 1000
# batchsize
batch_num = 32
# eps-action
EPSILON = 0.9  
# select-size
select_size = 64