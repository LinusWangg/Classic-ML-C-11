import numpy as np
import math
import matplotlib.pyplot as plt

data_x = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
          [0.481, 0.149], [0.437, 0.211],[0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], 
          [0.639, 0.161], [0.657, 0.198],[0.360, 0.370], [0.593, 0.042], [0.719, 0.103]
         ]
data_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data_good = list()
data_bad = list()

for i in range(len(data_x)):
    if data_y[i] == 1:
        data_good.append(data_x[i])
    else:
        data_bad.append(data_x[i])

miu0 = np.zeros((2,1))
miu1 = np.zeros((2,1))
Sigma0 = np.zeros((1,1))
Sigma1 = np.zeros((1,1))
Sigmaw = np.zeros((1,1))

for data in data_good:
    miu0 = miu0 + np.mat(data).T
miu0 = miu0 / len(data_good)

for data in data_bad:
    miu1 = miu1 + np.mat(data).T
miu1 = miu1 / len(data_bad)

for data in data_good:
    Sigma0 = Sigma0 + (np.mat(data).T-miu0)*(np.mat(data).T-miu0).T

for data in data_bad:
    Sigma1 = Sigma1 + (np.mat(data).T-miu1)*(np.mat(data).T-miu1).T

Sigmaw = Sigma0 + Sigma1

w = Sigmaw.I * (miu0 - miu1)

print(w)

w0 = w[0,0]
w1 = w[1,0]
sin = w1 / math.sqrt(w0**2+w1**2)
cos = w0 / math.sqrt(w0**2+w1**2)
tan = w1 / w0

plt.plot(miu0[0, 0], miu0[1, 0], "ro")
plt.plot(miu1[0, 0], miu1[1, 0], "r^")
plt.plot([-0.1, 0.1], [-0.1 * tan, 0.1 * tan])

for i in range(len(data_x)):
    elem = w.T * np.mat(data_x[i]).T
    if data_y[i] == 1:
        plt.plot(elem*cos, elem*sin, 'ob')
    else:
        plt.plot(elem*cos, elem*sin, '^g')

for i in range(len(data_x)):
    if data_y[i] == 1:
        plt.plot(data_x[i][0], data_x[i][1], 'ob')
    else:
        plt.plot(data_x[i][0], data_x[i][1], '^g')
plt.show()

