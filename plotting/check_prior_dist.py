import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

n = 10000
for i in np.linspace(-10, 10, 9):
    #intercept = stats.lognorm(s=1).rvs(size=n)
    intercept = 1 * np.ones(n)
    #slope = stats.norm(loc=0, scale=1).rvs(size=n)
    slope = i * np.ones(n)

    def param(gmt):
        return intercept / (1 + np.exp(-1 * (slope * gmt)))

    #gmt = np.zeros(n)
    gmt = np.linspace(0,1,n)
    param_sample = param(gmt)

    plt.plot(gmt, param_sample, label=f'slope = {i}')
plt.legend()
plt.show()
