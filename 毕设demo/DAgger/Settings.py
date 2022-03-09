from tensorboardX import SummaryWriter
WRITER = SummaryWriter("./log")
# eval to net iteration
ParaExchange = 100
# Critic GAMMA
GAMMA = 0.9
# GAME NAME
game_name = "Pendulum-v1"
# Exp Pool Size
Exp_Size = 1000
# Epoch
epoch_num = 1000
# batchsize
batch_num = 32