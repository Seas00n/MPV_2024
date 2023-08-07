import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from my_feature import *
from alignment_open3d import *
import cv2


def fifo(buffer, data):
    buffer[0] = buffer[1]
    buffer[1] = data
    return buffer



if __name__ == "__main__":
    data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4_OPEN3D/"
    traj_x = np.load(data_save_path + "traj_x.npy")
    traj_y = np.load(data_save_path + "traj_y.npy")
    env_type = np.load(data_save_path + "env_type_buffer.npy")

    file_list = os.listdir(data_save_path)
    num_frames = len(file_list) - 3
    fea_A_buffer = [np.zeros((0, 2)), np.zeros((0, 2))]
    fea_B_buffer = [np.zeros((0, 2)), np.zeros((0, 2))]
    fea_C_buffer = [np.zeros((0, 2)), np.zeros((0, 2))]
    camera_dx_buffer = []
    camera_dy_buffer = []
    camera_x_buffer = []
    camera_y_buffer = []
    corner_buffer = []
    corner_pre = 0
    corner_new = 0
    fig = plt.figure(figsize=(5, 5))
    plt.ion()

    for i in range(num_frames):
        pcd_new = np.load(data_save_path + "{}_pcd2d.npy".format(i))
        plt.cla()
        print("------------{}______________".format(i))
        if i < 1:
            pcd_pre = pcd_new
            fea_A_new, fea_B_new, fea_C_new, corner_pre = get_fea_sa(pcd_new)
            camera_dx_buffer.append(0)
            camera_dy_buffer.append(0)
            camera_x_buffer.append(0)
            camera_y_buffer.append(0)
            fea_A_buffer = fifo(fea_A_buffer, fea_A_new)
            fea_B_buffer = fifo(fea_B_buffer, fea_B_new)
            fea_C_buffer = fifo(fea_C_buffer, fea_C_new)
            fea_A_get_pre = False
            if np.shape(fea_A_new)[0] > 10:
                fea_A_get_pre = True
            fea_B_get_pre = False
            if np.shape(fea_B_new)[0] > 10:
                fea_B_get_pre = True
            fea_C_get_pre = False
            if np.shape(fea_C_new)[0] > 10:
                fea_C_get_pre = True
        else:
            fea_A_new, fea_B_new, fea_C_new, corner_new = get_fea_sa(pcd_new)
            fea_A_buffer = fifo(fea_A_buffer, fea_A_new)
            fea_B_buffer = fifo(fea_B_buffer, fea_B_new)
            fea_C_buffer = fifo(fea_C_buffer, fea_C_new)
            fea_A_pre = fea_A_buffer[0]
            fea_B_pre = fea_B_buffer[0]
            fea_C_pre = fea_C_buffer[0]

            fea_A_get_new = False
            if np.shape(fea_A_new)[0] > 10:
                fea_A_get_new = True
            fea_B_get_new = False
            if np.shape(fea_B_new)[0] > 10:
                fea_B_get_new = True
            fea_C_get_new = False
            if np.shape(fea_C_new)[0] > 10:
                fea_C_get_new = True


            pcd_to_align_new, pcd_to_align_pre, pcd_component_new, pcd_component_pre = align_rule(
                fea_A=[fea_A_new, fea_A_pre],
                fea_A_get=[fea_A_get_new, fea_A_get_pre],
                fea_B=[fea_B_new, fea_B_pre],
                fea_B_get=[fea_B_get_new, fea_B_get_pre],
                fea_C=[fea_C_new, fea_C_pre],
                fea_C_get=[fea_C_get_new, fea_C_get_pre],
                corner_situation=[corner_new, corner_pre]
            )
            if np.shape(pcd_to_align_pre)[0] == np.shape(pcd_to_align_new)[0] == 0:
                idx = np.arange(0, min(np.shape(pcd_new)[0], np.shape(pcd_pre)[0]))
                np.random.shuffle(idx)
                idx_chosen = np.random.choice(idx, size=600)
                idx_chosen = np.sort(idx_chosen)
                pcd_to_align_new = pcd_new[idx_chosen, :]
                pcd_to_align_pre = pcd_pre[idx_chosen, :]
                pcd_component_new = [200, 200, 200]
                pcd_component_pre = [200, 200, 200]
            # reg = open3d_alignment(pcd_s=pcd_to_align_pre, pcd_t=pcd_to_align_new)
            # trans = reg.alignment()
            # dx = -trans[1, 3]
            # dy = -trans[2, 3]
            dx, dy = icp(pcd_s=pcd_to_align_pre, pcd_t=pcd_to_align_new,
                         pcd_s_component=pcd_component_pre,
                         pcd_t_component=pcd_component_new)
            print(dx)
            print(dy)

            plt.scatter(pcd_new[0:-1:20, 0], pcd_new[0:-1:20, 1], color='red')
            plt.scatter(pcd_pre[0:-1:20, 0] - dx, pcd_pre[0:-1:20, 1] - dy, color='blue')
            if np.shape(fea_A_new)[0] > 10:
                plt.plot(fea_A_new[:, 0], fea_A_new[:, 1], ".:g", linewidth=10)
            if np.shape(fea_B_new)[0] > 10:
                plt.plot(fea_B_new[:, 0], fea_B_new[:, 1], ".:y", linewidth=10)
            if np.shape(fea_C_new)[0] > 10:
                plt.plot(fea_C_new[:, 0], fea_C_new[:, 1], ".:m", linewidth=10)
            if np.shape(fea_A_pre)[0] > 10:
                plt.plot(fea_A_pre[:, 0] - dx, fea_A_pre[:, 1] - dy, "*:g", linewidth=6)
            if np.shape(fea_B_pre)[0] > 10:
                plt.plot(fea_B_pre[:, 0] - dx, fea_B_pre[:, 1] - dy, "*:y", linewidth=6)
            if np.shape(fea_C_pre)[0] > 10:
                plt.plot(fea_C_pre[:, 0] - dx, fea_C_pre[:, 1] - dy, "*:m", linewidth=6)
            pcd_pre = pcd_new
            fea_A_get_pre = fea_A_get_new
            fea_B_get_pre = fea_B_get_new
            fea_C_get_pre = fea_C_get_new
            plt.text(0.5, -0.6, "corner_pre:{},id:{}".format(corner_pre, i - 1))
            plt.text(0.5, -0.7, "corner_new:{},id:{}".format(corner_new, i))
            corner_pre = corner_new
        plt.xlim([0, 1])
        plt.ylim([-1, 0])
        plt.draw()
        if i == 1:
            plt.pause(5)
        plt.pause(0.1)
