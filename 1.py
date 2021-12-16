import numpy as np
import matplotlib.pyplot as plt

size = 1e+07
z = np.random.normal(loc = 1.4,scale = 1.2, size = int(size))
sigma = 1.2
qz = 1/(np.sqrt(2*np.pi)*sigma**2)*np.exp(-0.5*(z-1.4)**2/sigma**2)
k = 3
u = np.random.uniform(low = 0, high = k*qz, size = int(size))

pz =  0.3*np.exp(-(z-0.3)**2) + 0.7* np.exp(-(z-2.)**2/0.3)
sample = z[pz >= u]
plt.hist(sample,bins=150,normed = True)