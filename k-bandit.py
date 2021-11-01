import numpy as np
from matplotlib import pyplot as plt

def findbest(q):
    maxn = -114517
    action = -1
    for i in range(len(q)):
        if q[i] > maxn:
            maxn = q[i]
            action = i
    return maxn, action

def addGauss(q):
    q += 0.1 * np.random.randn(10)

def main():
    q1 = np.zeros((10))
    n1 = np.zeros((10))
    q2 = np.zeros((10))
    n2 = np.zeros((10))
    k = 10
    eps1_avg = []
    eps2_avg = []
    eps1_total = 0
    eps2_total = 0
    eps1_acc = []
    eps2_acc = []
    eps1_accT = 0
    eps2_accT = 0
    step = 40000
    for i in range(step):
        prob = np.random.uniform(0.0, 1.0)
        if prob > 0.1:
            maxn, maxnindex = findbest(q1)
        else:
            choice = np.random.randint(0, k-1)
            maxn = q1[choice]
            maxnindex = choice
        addGauss(q1)
        maxn2, maxnindex2 = findbest(q1)
        if maxnindex2 == maxnindex:
            eps1_accT += 1
        n1[maxnindex] += 1
        q1[maxnindex] = q1[maxnindex] + (1 / n1[maxnindex])*(maxn - q1[maxnindex])
        eps1_total += maxn
        eps1_avg.append(eps1_total / (i + 1))
        eps1_acc.append(eps1_accT / (i + 1))

    for i in range(step):
        prob = np.random.uniform(0.0, 1.0)
        if prob > 0.1:
            maxn, maxnindex = findbest(q2)
        else:
            choice = np.random.randint(0, k-1)
            maxn = q2[choice]
            maxnindex = choice
        addGauss(q2)
        maxn2, maxnindex2 = findbest(q2)
        if maxnindex2 == maxnindex:
            eps2_accT += 1
        n2[maxnindex] += 1
        q2[maxnindex] = q2[maxnindex] + 0.1*(maxn - q2[maxnindex])
        eps2_total += maxn
        eps2_avg.append(eps2_total / (i + 1))
        eps2_acc.append(eps2_accT / (i + 1))

    return eps1_acc, eps1_avg, eps2_acc, eps2_avg

acc1, avg1, acc2, avg2 = main()

x = np.array([np.arange(0,40000,1)])
x = x.transpose()

plt.plot(x, acc1, 'b')
plt.plot(x, acc2, 'g')
plt.show()

