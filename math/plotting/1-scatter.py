#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, c='magenta', s=10)
plt.title('Men\'s Height vs Weight')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()