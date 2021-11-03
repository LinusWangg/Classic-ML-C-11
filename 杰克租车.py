import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from scipy.stats import poisson

def calculate_poisson(n, lamda):
    # n - 借车数量
    # lamda - 均值
    return poisson.pmf(n, lamda)

def calculate_evaluate(stat1, stat2, car_move, values):
    return_value = 0.0
    return_value -= abs(car_move) * 2

    cars_remain1 = min(stat1-car_move, 20)
    cars_remain2 = min(stat2+car_move, 20)

    prob_renttotal = 0.0
    for rent1 in range(cars_remain1+1):
        for rent2 in range(cars_remain2+1):
            prob_renttotal += pmf[rent1][3] * pmf[rent2][4]

    for rent1 in range(cars_remain1+1):
        for rent2 in range(cars_remain2+1):
            prob_rent = pmf[rent1][3] * pmf[rent2][4] / prob_renttotal

            reward = (rent1 + rent2) * 10
            cars_remain1_ = cars_remain1 - rent1
            cars_remain2_ = cars_remain2 - rent2

            prob_backtotal = 0.0
            for back1 in range(21-cars_remain1_):
                for back2 in range(21-cars_remain2_):
                     prob_backtotal += pmf[back1][3] * pmf[back2][2]

            for back1 in range(21-cars_remain1_):
                for back2 in range(21-cars_remain2_):
                    prob_back = pmf[back1][3] * pmf[back2][2] / prob_backtotal
            
                    cars_remain1__ = cars_remain1_ + back1
                    cars_remain2__ = cars_remain2_ + back2

                    prob = prob_back * prob_rent

                    return_value += prob * (reward + 0.9 * values[cars_remain1__][cars_remain2__])
    
    return return_value

def calculate_improve(policy, values):
    policy_stable = True
    for i in range(21):
        for j in range(21):
            old_actions = policy[i][j]
            actions = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
            actions_return = []
            for action in actions:
                if (0<=action<=i) or (-j<=action<=0):
                    actions_return.append(calculate_evaluate(i, j, action, values))
                else:
                    actions_return.append(-np.inf)
                
            new_actions = actions[np.argmax(actions_return)]
            policy[i][j] = new_actions

            if old_actions != new_actions:
                policy_stable = False

    return policy_stable

values = np.zeros((21, 21))
policy = np.zeros((21, 21), dtype=np.int)
pmf = np.zeros((21, 5))

for i in range(21):
    for j in range(2,5):
        pmf[i][j] = calculate_poisson(i, j)

iterations = 0

while True:
    
    while True:
        old_values = values.copy()
        for i in range(21):
            for j in range(21):
                new_value = calculate_evaluate(i, j, policy[i][j], values)
                values[i][j] = new_value

        max_value_change = abs(old_values - values).max()
        print(max_value_change)
        if max_value_change < 1e-4:
            break
    
    policy_stable = calculate_improve(policy, values)

    if policy_stable:
        break
    
    iterations += 1

for i in range(21):
    for j in range(21):
        print(policy[i][j],end=" ")
    print("\n")