from tensorboardX import SummaryWriter
WRITER = SummaryWriter("./log-连续")
# eval to net iteration
ParaExchange = 100
# Critic GAMMA
GAMMA = 0.9
# GAME NAME
game_name = "InvertedDoublePendulum-v2"
# Exp Pool Size
Exp_Size = 1000
# Epoch
epoch_num = 3000
# batchsize
batch_num = 32
# select_size
select_size = 64