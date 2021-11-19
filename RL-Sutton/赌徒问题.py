import matplotlib.pyplot as plt
import numpy as np

def calculate_values(values, states, ph, gamma):
    delta = 0
    for s in states:
        v = values[s]
        values_new = np.zeros((min(s, 100-s)+1))
        action = [x for x in range(1, min(s, 100-s)+1)]
        for a in action:
            if a+s == 100:
                values_new[a] = ph*1+(1-ph)*(gamma*values[s-a])
            else:
                values_new[a] = ph*(gamma*values[a+s])+(1-ph)*(gamma*values[s-a])
        values[s] = np.max(values_new)
        delta = max(delta, abs(v-values[s]))
    return delta

def calculate_action(values, states, ph, gamma):
    pi = [0 for i in range(101)]
    for s in states:
        values_new = np.zeros((min(s, 100-s)+1))
        action = [x for x in range(1, min(s, 100-s)+1)]
        for a in action:
            if a+s == 100:
                values_new[a] = ph+(1-ph)*(gamma*values[s-a])
            else:
                values_new[a] = ph*(gamma*values[a+s])+(1-ph)*(gamma*values[s-a])
        pi[s] = np.argmax(values_new)
    return pi


values = np.random.randn(101)*0.1
values[100] = 1
values[0] = 0
states = [x for x in range(1, 100)]

while True:
    delta = calculate_values(values, states, 0.4, 1)
    if delta < 1e-20:
        break

print(values)

pi = calculate_action(values, states, 0.4, 1)
print(pi)

S = np.linspace(0, 100, num=101, endpoint=True)
plt.figure()
plt.plot(S, values)
plt.show()
plt.figure()
plt.bar(S, pi)
plt.show()