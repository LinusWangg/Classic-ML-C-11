from tensorboardX import SummaryWriter
WRITER = SummaryWriter("./log-LSTM离散")
# eval to net iteration
ParaExchange = 100
# Critic GAMMA
GAMMA = 0.9
# GAME NAME
game_name = "CartPole-v0"
# Exp Pool Size
Exp_Size = 1000
# Epoch
epoch_num = 1000
# batchsize
batch_num = 32
# eps-action
EPSILON = 0.9  
# 