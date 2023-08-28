import datetime
import random
import sys

import torch

sys.path.append("/home/yuxuan/Project/MPV_2024/")
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import lcm
import time
import PIL
import os
from scipy import interpolate
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *

from model.pointnet2d import *

env = Environment()
down_sample_rate = 5

model = PointNetDenseCls2d(k=7)
model.cuda()
model.load_state_dict(torch.load("model.pth"))
device = torch.device("cuda")


def data_in_one(inputdata, data_min, data_max):
    outputdata = (inputdata - data_min) / (data_max - data_min)
    return outputdata


def pre_process(pcd):
    voxel = np.copy(pcd)
    abs_voxel = np.abs(voxel[:, 0] - voxel[-1, 0]) + np.abs(voxel[:, 1] - voxel[-1, 1])
    idx_arange = np.argsort(abs_voxel, kind='quicksort')
    voxel = voxel[idx_arange, :]
    idx_discontinuous = np.where(np.sum(np.abs(np.diff(voxel, axis=0)), axis=1) > 0.05)[0]
    y_list = np.split(voxel[:, 1], idx_discontinuous)
    x_list = np.split(voxel[:, 0], idx_discontinuous)
    new_x = x_list[0][1:-1:2]
    new_y = y_list[0][1:-1:2]
    for i in range(len(idx_discontinuous)):
        interp_y = np.linspace(voxel[idx_discontinuous[i], 1], voxel[idx_discontinuous[i] + 1, 1], 11)
        interp_x = np.ones(11) * np.mean([voxel[idx_discontinuous[i], 0], voxel[idx_discontinuous[i] + 1, 0]])
        new_y = np.hstack([new_y, interp_y, y_list[i + 1][1:-1:3]])
        new_x = np.hstack([new_x, interp_x, x_list[i + 1][1:-1:3]])
    idx_pre = np.arange(np.shape(new_x)[0])
    fx = interpolate.interp1d(idx_pre, new_x, "zero")
    fy = interpolate.interp1d(idx_pre, new_y, "zero")
    xx_new = np.linspace(0, idx_pre[-1], 100)
    pcd_new = np.vstack([fx(xx_new), fy(xx_new)]).T
    voxel = np.vstack([fx(xx_new), fy(xx_new)]).T
    min_voxel_x = min(voxel[:, 0])
    max_voxel_x = max(voxel[:, 0])
    min_voxel_y = min(voxel[:, 1])
    max_voxel_y = max(voxel[:, 1])
    voxel[:, 0] = data_in_one(voxel[:, 0], min_voxel_x, max_voxel_x)
    voxel[:, 1] = data_in_one(voxel[:, 1], min_voxel_y, max_voxel_y)

    return voxel, pcd_new


def plot_mask(ax, pcd_new, label):
    ax.scatter(pcd_new[:, 0], pcd_new[:, 1]+0.2, color='b', alpha=0.1)
    idx_B = np.where(label == 2)[0]
    ax.scatter(pcd_new[idx_B, 0], pcd_new[idx_B, 1]+0.2, color='g', linewidths=2)
    idx_C = np.where(label == 3)[0]
    ax.scatter(pcd_new[idx_C, 0], pcd_new[idx_C, 1]+0.2, color='y', linewidths=2)

str = input("按回车开始")
if __name__ == "__main__":
    imu_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
    num_points = int(38528 / down_sample_rate) + 1
    pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(num_points, 3))

    ax = plt.axes()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.ion()
    num_frame = 2500
    try:
        for i in range(num_frame):
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            plt.cla()
            pcd_data_temp = pcd_buffer[:]
            imu_data = imu_buffer[:]
            eular_angle = imu_data[7:10]
            env.pcd_to_binary_image(pcd_data_temp, eular_angle)
            env.thin()
            env.classification_from_img()

            pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
            pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
            line_theta = pcd_os.env_rotate

            voxel, pcd_new = pre_process(env.pcd_thin)
            voxel = torch.from_numpy(voxel).to(torch.float32)

            pcd_os.show_(ax, pcd_color='r', id=int(i), downsample=1)
            if pcd_os.env_type == 1 or 3:
                if 8 >= pcd_os.corner_situation > 0:
                    voxel = torch.reshape(voxel, (1, -1, 2))
                    voxel = torch.vstack([voxel, voxel])
                    voxel = voxel.transpose(2, 1)
                    voxel = voxel.to(device)
                    label = model(voxel)
                    label = torch.max(label[0], dim=1)[1]
                    label = label[0]
                    label = label.cpu().detach().numpy()
                    plot_mask(ax, pcd_new=pcd_new, label=label)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.draw()
            plt.pause(0.025)

    except KeyboardInterrupt:
        pass
