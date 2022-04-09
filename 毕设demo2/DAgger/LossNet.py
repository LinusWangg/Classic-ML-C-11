import torch
import torch.nn as nn
import torch.nn.functional as F

class LossNet(nn.Module):
    ## 学习者
    def __init__(self, n_features):
        super(LossNet, self).__init__()
        self.layer1 = nn.Linear(n_features, 64)
        self.layer1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.layer2 = nn.Linear(64, 1)
        self.layer2.weight.data.normal_(0, 0.1) # initialization of FC1

    def forward(self, x):
<<<<<<< HEAD
=======
        x = x.cuda()
>>>>>>> 2149808199f7913f93c29d38dd98773d7fc6018f
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

class LossPred(object):

    def __init__(self, n_features, lr=0.002):
        super(LossPred, self).__init__()
        self.n_features = n_features
        self.lossNet = LossNet(self.n_features)
        self.lossfunc = nn.MSELoss()
        self.optim = torch.optim.Adam(self.lossNet.parameters(), lr)

    def pred(self, data):
        return self.lossNet.forward(data)
    
    def train(self, data, yhat_loss):
        mean_loss = 0
        for i in range(10):
            pred_res = self.pred(data.to(torch.float32))
            loss = self.lossfunc(pred_res, yhat_loss)
            mean_loss += loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        mean_loss /= 10
        return mean_loss
        
