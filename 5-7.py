import numpy as np
from matplotlib import pyplot as plt
X = np.random.randn(1,100)
np.append(X, np.random.randn(1,100) + 3)
np.append(X, np.random.randn(1,100) - 3)
Y = X ** 2 + 1

def relu(x):
    if x.all()<0:
        return 0
    else:
        return x

def back_relu(x):
    if x.all()<0:
        return 0
    else:
        return 1

def sigmoid(x):
    if x.all() >= 0:
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x) / (1+np.exp(x))

def back_sigmoid(x):
    return x*(1-x)

def tanh(x):
    return np.tanh(x)

def back_tanh(x):
    return 1 - np.multiply(np.tanh(x), np.tanh(x))

class Net:
    def __init__(self, dim, act):
        np.random.seed(10)
        self.epoch = 0
        self.Layer = len(dim) - 1
        self.dim = dim
        self.act = act
        self.W = [0,0,0,0]
        self.B = [0,0,0,0]
        self.Z = [0,0,0,0]
        self.A = [0,0,0,0]
        self.dW = [0,0,0,0]
        self.dB = [0,0,0,0]
        self.dZ = [0,0,0,0]
        self.dA = [0,0,0,0]
        self.dW_Squar = [0,0,0,0]
        self.dB_Squar = [0,0,0,0]
        for i in range(1, self.Layer+1):
            self.W[i] = np.array(np.random.randn(dim[i],dim[i-1]))/np.sqrt(dim[i])
            self.B[i] = np.array(np.random.randn(dim[i],1))/np.sqrt(dim[i])
        print("Initial Net complete")
    
    def activate(self, L, Z):
        if self.act[L] == "relu":
            return relu(Z)
        elif self.act[L] == "sigmoid":
            return sigmoid(Z)
        elif self.act[L] == "tanh":
            return tanh(Z)

    def back_activate(self, L, Z):
        if self.act[L] == "relu":
            return back_relu(Z)
        elif self.act[L] == "sigmoid":
            return back_sigmoid(Z)
        elif self.act[L] == "tanh":
            return back_tanh(Z)

    def forward(self, X):
        self.Z[0] = np.array(X)
        self.A[0] = np.array(X)
        for i in range(1, self.Layer+1):
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            self.A[i] = self.activate(i, self.Z[i])
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
            self.dZ[i] = np.multiply(self.dA[i], self.back_activate(i, self.Z[i]))
            self.dA[i-1] = np.dot(self.W[i].T, self.dZ[i])
            self.dW[i] = np.dot(self.dZ[i], self.A[i-1].T) / self.m
            self.dB[i] = np.sum(self.dZ[i], axis=1, keepdims=True) / self.m
            i -= 1

    def GD(self, lr, eps):
        for i in range(1, self.Layer+1):
            dW_Squar = np.multiply(self.dW[i], self.dW[i])
            dB_Squar = np.multiply(self.dB[i], self.dB[i])
            self.dW_Squar[i] = 0.9*self.dW_Squar[i] + 0.1*dW_Squar
            self.dB_Squar[i] = 0.9*self.dB_Squar[i] + 0.1*dB_Squar
            self.W[i] -= lr * (1 / (np.sqrt(self.dW_Squar[i]) + eps)) * self.dW[i]
            self.B[i] -= lr * (1 / (np.sqrt(self.dB_Squar[i]) + eps)) * self.dB[i]
    
    def predict(self, input):
        Z = [0,0,0,0]
        A = [0,0,0,0]
        Z[0] = np.array(input)
        A[0] = np.array(input)
        for i in range(1, self.Layer+1):
            Z[i] = np.dot(self.W[i], A[i-1]) + self.B[i]
            A[i] = self.activate(i, Z[i])
        return A[self.Layer]
    
    
net = Net([1,5,1],["","tanh","relu"])
while True:
    net.forward(X)
    #print("Y:{0}".format(Y))
    cost = net.calculateCost(Y)[1]
    if cost <= 0.005:
        break
    net.backward()
    net.GD(0.01, 0.00)

#test_X = np.random.rand(1,100)
#test_Y = np.sin(test_X) + 1
#pre_X = net.predict(test_X)
#print(test_Y - pre_X)


x = np.array([np.arange(-3, 3, 0.1)])
 
y = net.predict(x)

x = x.transpose()
y = y.transpose()

plt.title("一元二次函数")
plt.plot(x, y)
plt.show()