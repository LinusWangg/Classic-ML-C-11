import numpy as np
import math
import matplotlib.pyplot as plt

data_x = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
          [0.481, 0.149], [0.437, 0.211],[0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], 
          [0.639, 0.161], [0.657, 0.198],[0.360, 0.370], [0.593, 0.042], [0.719, 0.103]
         ]
data_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def combine(beta, x):
    x = np.mat(x + [1.]).T
    return beta.T * x

def predict(beta, x):
    return 1 / (1 + math.exp(-combine(beta, x)))

def p1(beta, x):
    return math.exp(combine(beta, x)) / (1 + math.exp(combine(beta, x)))

beta = np.mat([0.] * 3).T

steps = 50

for step in range(steps):
    param_1 = np.zeros((3, 1))
    for i in range(len(data_x)):
        x = np.mat(data_x[i] + [1.]).T
        param_1 = param_1 - x * (data_y[i] - p1(beta, data_x[i]))
    param_2 = np.zeros((3, 3))
    for i in range(len(data_x)):
        x = np.mat(data_x[i] + [1.]).T
        param_2 = param_2 + x * x.T * p1(beta, data_x[i]) * (1 - p1(beta, data_x[i]))
    last_beta = beta
    beta = last_beta - param_2.I * param_1
    if np.linalg.norm(last_beta.T - beta.T) < 1e-6:
        print(step)
        break

for i in range(len(data_x)):
    if data_y[i] == 1:
        plt.plot(data_x[i][0], data_x[i][1], 'ob')
    else:
        plt.plot(data_x[i][0], data_x[i][1], '^g')
w_0 = beta[0, 0]
w_1 = beta[1, 0]
b = beta[2, 0]
print(w_0, w_1, b)
x_0 = -b / w_0 #(x_0, 0)
x_1 = -b / w_1 #(0, x_1)
plt.plot([x_0, 0], [0, x_1])
plt.show()
