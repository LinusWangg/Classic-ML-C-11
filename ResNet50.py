import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import torch
from torch import nn,optim
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)

class BottleNeck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsampling=False):
        super(BottleNeck, self).__init__()
        self.downsampling = downsampling
        self.stride = stride

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, 4*planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes*4)
            )
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet50结构 7*7,64,stride2 3*3,maxpool,stride2 64-64-256*3 128-128-512*4
# 256-256-1024*6 512*512*2048*3
class ResNet(nn.Module):
    def __init__(self, num_Layers, num_classes):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_Layer(inplanes=64, planes=64, stride=1, num_Layer=num_Layers[0])
        self.layer2 = self.make_Layer(inplanes=256, planes=128, stride=2, num_Layer=num_Layers[1])
        self.layer3 = self.make_Layer(inplanes=512, planes=256, stride=2, num_Layer=num_Layers[2])
        self.layer4 = self.make_Layer(inplanes=1024, planes=512, stride=2, num_Layer=num_Layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(2048, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)      

    def make_Layer(self, inplanes, planes, stride, num_Layer):
        layers = []
        layers.append(BottleNeck(inplanes, planes, stride, downsampling=True))
        for i in range(1, num_Layer):
            layers.append(BottleNeck(planes*4, planes)) #此时上面一层已经输出了4*planes了，所以这里in_channels = planes*4

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        X = Variable(x_val.cuda(), requires_grad = False)
        output = self.forward(X)
        return output.cpu().data.numpy().argmax(axis=1)

def ResNet50():
    return ResNet(num_Layers=[3, 4, 6, 3], num_classes=6).cuda()

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
    model = ResNet50()
    print(model)
    loss = torch.nn.CrossEntropyLoss(size_average=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    batch_size = 32
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
