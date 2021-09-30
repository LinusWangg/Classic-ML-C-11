import numpy as np
import matplotlib.pyplot as plt

features=np.array([
    [1,2,2,1,0,1,2,2,2,1,0,0,1,0,2,0,1],
    [2,2,2,2,2,1,1,1,1,0,0,2,1,1,1,2,2],
    [1,0,1,0,1,1,1,1,0,2,2,1,1,0,1,1,0],
    [0,0,0,0,0,0,1,0,1,0,2,2,1,1,0,2,1],
    [2,2,2,2,2,1,1,1,1,0,0,0,2,2,1,0,1],
    [1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,1],
    [0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
    [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]
])
labels=np.array([
    [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
])


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

class Net():
    def __init__(self, input_cell = 8, hidden_cell = 10, output_cell = 1):
        self.W1 = np.random.randn(input_cell, hidden_cell)
        self.e1 = np.zeros(hidden_cell).reshape(-1,1)
        self.W2 = np.random.randn(hidden_cell, output_cell)
        self.e2 = np.zeros(output_cell).reshape(-1,1)

        self.o1 = np.zeros(hidden_cell).reshape(-1,1)
        self.o2 = np.zeros(output_cell).reshape(-1,1)

        self.dW1 = np.zeros(self.W1.shape)
        self.de1 = np.zeros(self.e1.shape)
        self.dW2 = np.zeros(self.W2.shape)
        self.de2 = np.zeros(self.e2.shape)

    def forward(self, X):
        self.o1 = sigmoid(np.dot(self.W1.T, X) + self.e1) #(10,8)(8,?)=(10,?)
        self.o2 = sigmoid(np.dot(self.W2.T, self.o1) + self.e2) #(1,10)(10,?)=(1,?)
        return self.o2

    def standardBP(self, X, rate, label):
        # hid -> output
        self.dW2 = rate * self.o2 * (1 - self.o2) * (label - self.o2) * self.o1
        self.de2 = -1 * rate * self.o2 * (1 - self.o2) * (label - self.o2)
        # input -> hid
        self.dW1 = rate * self.o1 * (1 - self.o1) * (self.W2 * self.o1 * (1 - self.o1) * (label - self.o1)) * X
        self.de1 = -1 * rate * self.o1 * (1 - self.o1) * (self.W2 * self.o1 * (1 - self.o1) * (label - self.o1))
        self.dW1 = self.dW1.T

        self.W1 += self.dW1
        self.W2 += self.dW2
        self.e1 += self.de1
        self.e2 += self.de2

    def accumlateBP(self, X, rate, label):
        # hid -> output
        self.dW2 = rate * np.dot(self.o2 * (1 - self.o2) * (label - self.o2), self.o1.T)
        self.de2 = -1 * rate * (self.o2 * (1 - self.o2) * (label - self.o2)).sum(axis=1).reshape(-1,1)
        self.dW2 = self.dW2.T
        # input -> hid
        self.dW1 = rate * np.dot(self.o1 * (1 - self.o1) * (self.W2 * self.o1 * (1 - self.o1) * (label - self.o1)), X.T)
        self.de1 = -1 * rate * (self.o1 * (1 - self.o1) * (self.W2 * self.o1 * (1 - self.o1) * (label - self.o1))).sum(axis = 1).reshape(-1,1)
        self.dW1 = self.dW1.T

        self.W1 += self.dW1
        self.W2 += self.dW2
        self.e1 += self.de1
        self.e2 += self.de2

def trainStandardBP(features, labels, rate):
    net = Net()
    epoch = 0
    loss = 1
    all_loss = []
    while loss > 0.1:
        for i in range(features.shape[1]):
            X=features[:,i]
            Y=labels[0,i]
            net.forward(X.reshape(-1,1))
            net.standardBP(X, rate, Y)
        output=net.forward(features)
        loss=0.5*((output-labels)**2).sum()
        epoch+=1
        all_loss.append(loss)
        print("标准BP","学习率：",rate,"\n终止epoch：",epoch,"loss: ",loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(all_loss)
    plt.show()
    return net

def trainAccumlateBP(features, labels, rate):
    net = Net()
    epoch = 0
    loss = 1
    all_loss = []
    while loss > 0.02:
        net.forward(features)
        net.accumlateBP(features, rate, labels)
        output=net.forward(features)
        loss=0.5*((output-labels)**2).sum()/labels.shape[1]
        epoch+=1
        all_loss.append(loss)
        print("标准BP","学习率：",rate,"\n终止epoch：",epoch,"loss: ",loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(all_loss)
    plt.show()
    return net
    


net = trainAccumlateBP(features,labels,0.2)
X=features[:,-1]
print(net.forward(X.reshape(-1,1)))
