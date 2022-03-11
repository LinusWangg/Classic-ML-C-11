import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):
    ## 学习者
    def __init__(self, n_features, n_actions):
        super(Learner, self).__init__()
        self.lstm = nn.LSTM(n_features, 64, batch_first=True)
        self.layer2 = nn.Linear(64, n_actions)
        self.layer2.weight.data.normal_(0, 0.1) # initialization of FC1

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h = out[:, :, :]
        h = h.reshape(out.shape[0]*out.shape[1], out.shape[2])
        x = self.layer2(h)
        x = F.softmax(x)
        return x