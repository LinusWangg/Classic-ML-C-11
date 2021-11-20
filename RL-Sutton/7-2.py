import numpy as np
import matplotlib.pyplot as plt

True_value = [i/20 for i in range(-20, 21, 2)]

value = [0 for i in range(21)]

def TD_N(n, value):
    gamma = 0.9
    alpha = 0.2
    for i in range(10000):
        if i%100==0:
            print(i)
        S_List = []
        Act = [-1, 1]
        R_List = [0]
        now_state = 10
        T = np.inf
        t = 0
        S_List.append(now_state)
        while True:
            if now_state != 0 and now_state != 20:
                action = np.choose(np.random.randint(0, 2), Act)
                next_state = now_state + action
                if next_state == 0:
                    R_List.append(-1)
                elif next_state == 20:
                    R_List.append(1)
                else:
                    R_List.append(0)
                if next_state == 0 or next_state == 20:
                    T = t+1
                now_state = next_state
                S_List.append(now_state)

            tao = t-n+1
            if tao >= 0 and tao != np.inf:
                G = 0
                for i in range(tao+1, min(tao+n, T)+1):
                    G += np.power(gamma, i-tao-1)*R_List[i]
                if tao+n < T:
                    G += np.power(gamma, n)*value[S_List[tao+n]]
                value[S_List[tao]] = value[S_List[tao]] + alpha*(G-value[S_List[tao]])
            if tao == T-1:
                break
            t += 1

TD_N(8, value)
#for i in range(1,20):
#    sum += (True_value[i]-value[i])**2
#loss = np.sqrt(sum/19)
print(value)
            
