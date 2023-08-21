import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

imu_buffer = np.memmap("imu_buffer.npy", dtype='float32', mode='r',
                       shape=(12,))

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
plt.ion()

try:
    while True:
        eular = [imu_buffer[6], imu_buffer[7], imu_buffer[8]]
        plt.cla()
        r_mat = R.from_euler('xyz', [eular[0], eular[1], eular[2]], degrees=True).as_matrix()
        x_ = r_mat[:, 0]
        y_ = r_mat[:, 1]
        z_ = r_mat[:, 2]
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.plot3D([0, x_[0]], [0, x_[1]], [0, x_[2]], color='r')
        ax.plot3D([0, y_[0]], [0, y_[1]], [0, y_[2]], color='green')
        ax.plot3D([0, z_[0]], [0, z_[1]], [0, z_[2]], color='blue')
        plt.draw()
        plt.pause(0.05)

except KeyboardInterrupt:
    pass
