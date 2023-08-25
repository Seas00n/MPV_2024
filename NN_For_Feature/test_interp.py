import numpy as np
import matplotlib.pyplot as plt

pcd_new = np.load("point2d_example.npy")

idx = np.arange(np.shape(pcd_new)[0])
plt.scatter(idx, pcd_new[:, 1])
idx_diff = np.arange(np.shape(pcd_new)[0]-1)
plt.scatter(idx_diff, np.diff(pcd_new[:, 1]))
idx_discontinuous = np.where(abs(np.diff(pcd_new[:,1]))>0.05)[0]
plt.scatter(idx[idx_discontinuous[0]-1], pcd_new[idx_discontinuous[0]-1,1])
plt.scatter(idx[idx_discontinuous[0]+1], pcd_new[idx_discontinuous[0]+1,1])
plt.show()
