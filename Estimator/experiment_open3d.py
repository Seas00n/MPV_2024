import sys

import matplotlib.pyplot as plt

from Environment.Plot_ import *
from Environment.alignment_open3d import open3d_alignment
from Environment.Environment import *
from Environment.feature_extra_new import *

experiment_idx = 4
moca_align_file_path = "/home/yuxuan/Project/StairFeatureExtraction/MocaDataProcess/align_data/"
moca_data = np.load(moca_align_file_path + "Moca_align{}.npy".format(experiment_idx), allow_pickle=True)
idx_align = np.load(moca_align_file_path + "idx_align{}.npy".format(experiment_idx), allow_pickle=True)
file_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(experiment_idx)

env = Environment()

pcd_os_buffer = [[], []]

if __name__ == "__main__":

    start_idx = idx_align[1]
    end_idx = idx_align[3]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.ion()
    plt.show(block=False)

    for i in range(start_idx, end_idx):
        pcd_data = np.load(file_path + "{}_pcd.npy".format(i), allow_pickle=True)
        pcd_data = pcd_data[0:-1, :]
        imu_data = np.load(file_path + "{}_imu.npy".format(i), allow_pickle=True)
        eular_angle = imu_data[7:10]
        env.pcd_to_binary_image(pcd_data, eular_angle)
        env.thin()

        plt.cla()
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 3)
        ax.set_zlim(-1, 2)
        ax.view_init(elev=20, azim=-45)

        if i == start_idx:
            pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
            pcd_os_buffer[-1] = pcd_os
        else:
            pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_opreator_system
            pcd_pre = pcd_pre_os.pcd_new
            pcd_new, pcd_new_os = env.pcd_thin, pcd_opreator_system(env.pcd_thin)

            # pcd_pre = pcd_new + np.array([0.2, 0.2]).reshape((-1, 2))

            pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
            regis = open3d_alignment(pcd_s=pcd_new,
                                     pcd_t=pcd_pre)
            trans = regis.alignment_new()
            pcd3d_new = pcd2d_to_3d(pcd_new, num_rows=5)
            pcd3d_pre = pcd2d_to_3d(pcd_pre, num_rows=5)

            print(trans[:, 3])
            xmove_o3d = -trans[1, 3]
            ymove_o3d = -trans[2, 3]
            print("xmoveo3d = {}, ymoveo3d = {}".format(xmove_o3d, ymove_o3d))

            xx = pcd3d_new[:, 0]
            yy = pcd3d_new[:, 1]
            zz = pcd3d_new[:, 2]
            ax.plot3D(xx[0:-1:51],
                      yy[0:-1:51],
                      zz[0:-1:51],
                      linewidth=1,
                      color='red')

            xx = pcd3d_pre[:, 0]
            yy = pcd3d_pre[:, 1] + xmove_o3d
            zz = pcd3d_pre[:, 2] + ymove_o3d
            ax.plot3D(xx[0:-1:51],
                      yy[0:-1:51],
                      zz[0:-1:51],
                      linewidth=1,
                      color='blue')
            plt.draw()
            plt.pause(0.01)
