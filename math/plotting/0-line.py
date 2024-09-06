#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, 'r-')
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()