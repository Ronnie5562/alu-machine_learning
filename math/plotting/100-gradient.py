#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))


plt.scatter(x, y, c=z)
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title('Mountain Elevation')
cbar = plt.colorbar()
cbar.set_label('elevation (m)')
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()