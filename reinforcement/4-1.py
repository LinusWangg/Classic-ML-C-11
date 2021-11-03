import numpy as numpy

value = [0.0 for i in range(17)]

def calculatenext(x, values):
    if x == 16:
        return values[12] + values[13] + values[14] + values[16]
    elif x // 4 == 0 and x % 4 == 3:
        return values[x-1] + values[x+4] + values[x]*2
    elif x // 4 == 3 and x % 4 == 0:
        return values[x+1] + values[x-4] + values[x]*2
    elif x // 4 == 0:
        return values[x-1] + values[x+4] + values[x+1] + values[x]
    elif x // 4 == 3:
        return values[x-1] + values[x-4] + values[x+1] + values[x]
    elif x % 4 == 0:
        return values[x-4] + values[x+4] + values[x+1] + values[x]
    elif x % 4 == 3:
        return values[x-4] + values[x+4] + values[x-1] + values[x]
    else:
        return values[x-4] + values[x+4] + values[x-1] + values[x+1]


def calculate(values):
    delta = 0
    for i in range(17):
        if i == 0 or i == 15:
            continue
        v = values[i]
        values[i] = -1 + 0.25 * calculatenext(i, values)
        delta = max(delta, abs(values[i]-v))
    return delta

delta = 1
while delta > 1e-10:
    delta = calculate(value)

for i in range(16):
    print("%.2f" % (value[i]),end=" ")
    if i%4 == 3:
        print("\n")
print("          %.2f" % (value[16]))



