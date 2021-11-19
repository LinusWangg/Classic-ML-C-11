from typing import ForwardRef
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.autograd import Variable
from torch import optim

np.random.seed(1)

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)) #60*60*10
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2)) #30*30*10
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)) #26*26*20
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2)) #13*13*20
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(20*13*13, 1000))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(1000, 100))
        self.fc.add_module("relu_4", torch.nn.ReLU())
        self.fc.add_module("dropout_4", torch.nn.Dropout())
        self.fc.add_module("fc3", torch.nn.Linear(100, 6))

    def forward(self, X):
        X = X.cuda()
        X = self.conv.forward(X)
        X = X.view(-1, 20*13*13)
        return self.fc.forward(X)

    def train(self, loss, optimizer, x_val, y_val):
        x_val = x_val.cuda()
        y_val = y_val.cuda()
        X = Variable(x_val, requires_grad = False)

        optimizer.zero_grad()

        fx = self.forward(X)
        output = loss.forward(fx, y_val)

        output.backward()

        optimizer.step()

        return output.item()

    def predict(self, x_val):
        X = Variable(x_val, requires_grad = False)
        output = self.forward(X)
        return output.cpu().data.numpy().argmax(axis=1)


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def main():
    train_X, train_Y, test_X, test_Y, classes = load_dataset()
    train_X = train_X.reshape(-1, 3, 64, 64)
    test_X = test_X.reshape(-1, 3, 64, 64)
    train_Y = train_Y.reshape(-1, 1)

    train_X = torch.from_numpy(train_X / 255).float()
    test_X = torch.from_numpy(test_X / 255).float()
    train_Y = torch.from_numpy(train_Y.squeeze()).long()

    n_samples = len(train_X)
    n_classes = 6
    model = ConvNet().cuda()

    loss = torch.nn.CrossEntropyLoss(size_average=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    batch_size = 50
    epochs = 500
    for i in range(epochs):
        cost = 0
        num_batches = n_samples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k+1) * batch_size
            cost += model.train(loss, optimizer, train_X[start:end], train_Y[start:end])
        
        predY = model.predict(test_X)
        print("Epoch %d, cost = %f, acc = %.2f%%"
            % (i+1, cost / num_batches, 100. * np.mean(predY == test_Y)))

    
main()



