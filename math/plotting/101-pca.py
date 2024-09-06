#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load("./datasets/pca.npz/data.npy")
labels = np.load("./datasets/pca.npz/labels.npy")


data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    pca_data[:, 2],
    c=labels,
    label=labels,
    cmap=plt.get_cmap('plasma'),
)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title("PCA of Iris Dataset")

plt.tight_layout()
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
