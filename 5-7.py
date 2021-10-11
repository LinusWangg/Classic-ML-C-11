import numpy as np
from matplotlib import pyplot as plt
X = np.random.randn(1,100)
Y = X*4 + 1

def sigmoid(x):
    if x.all()<0:
        return 0
    else:
        return x

def back_sigmoid(x):
    if x.all()<0:
        return 0
    else:
        return 1

class Net:
    def __init__(self, dim):
        np.random.seed(10)
        self.epoch = 0
        self.Layer = len(dim) - 1
        self.W = [0,0,0,0]
        self.B = [0,0,0,0]
        self.Z = [0,0,0,0]
        self.A = [0,0,0,0]
        self.dW = [0,0,0,0]
        self.dB = [0,0,0,0]
        self.dZ = [0,0,0,0]
        self.dA = [0,0,0,0]
        for i in range(1, self.Layer+1):
            self.W[i] = np.array(np.random.randn(dim[i],dim[i-1]))/np.sqrt(dim[i])
            self.B[i] = np.array(np.random.randn(dim[i],1))/np.sqrt(dim[i])
        print("Initial Net complete")

    def forward(self, X):
        self.Z[0] = np.array(X)
        self.A[0] = np.array(X)
        for i in range(1, self.Layer+1):
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            self.A[i] = sigmoid(self.Z[i])
        #print("result:{0}".format(self.A[self.Layer]))
    
    def calculateCost(self, Y):
        self.epoch += 1
        self.m = Y.shape[1]
        cost = np.multiply(self.A[self.Layer]-Y, self.A[self.Layer]-Y)
        cost = 0.5 * np.sum(cost) / self.m
        print("epoch:{0},loss:{1}".format(self.epoch, cost))
        return (self.epoch, cost)

    def backward(self):
        self.dA[self.Layer] = self.A[self.Layer]-Y
        i = self.Layer
        while i > 0:
            self.dZ[i] = np.multiply(self.dA[i], back_sigmoid(self.Z[i]))
            self.dA[i-1] = np.dot(self.W[i].T, self.dZ[i])
            self.dW[i] = np.dot(self.dZ[i], self.A[i-1].T) / self.m
            self.dB[i] = np.sum(self.dZ[i], axis=1, keepdims=True) / self.m
            i -= 1

    def GD(self, lr):
        for i in range(1, self.Layer+1):
            self.W[i] -= lr * self.dW[i]
            self.B[i] -= lr * self.dB[i]
    
    def predict(self, input):
        Z = [0,0,0,0]
        A = [0,0,0,0]
        Z[0] = np.array(input)
        A[0] = np.array(input)
        for i in range(1, self.Layer+1):
            Z[i] = np.dot(self.W[i], A[i-1]) + self.B[i]
            A[i] = sigmoid(Z[i])
        return A[self.Layer]
    
net = Net([1,1])
while True:
    net.forward(X)
    #print("Y:{0}".format(Y))
    cost = net.calculateCost(Y)[1]
    if cost <= 0.00005:
        break
    net.backward()
    net.GD(0.008)

print(net.predict([[2]]))