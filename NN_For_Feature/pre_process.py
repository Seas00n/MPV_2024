import datetime
import random
import sys

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

env = Environment()
down_sample_rate = 5


class open3d_voxelpipeline(object):
    def __init__(self, pcd):
        self.pcd2d = pcd
        self.pcd2d_to_3d(num_rows=1)
        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd3d)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd_o3d, voxel_size=0.01)
        voxel_all = voxel_grid.get_voxels()
        voxel_x = []
        voxel_y = []
        for voxel in voxel_all:
            voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            voxel_x.append(voxel_center[1])
            voxel_y.append(voxel_center[2])
        voxel_x = np.array(voxel_x)
        voxel_y = np.array(voxel_y)
        abs_voxel = np.abs(voxel_x - voxel_x[-1]) + np.abs(voxel_y - voxel_y[-1])
        idx_arange = np.argsort(abs_voxel, kind='quicksort')
        voxel_x = voxel_x[idx_arange]
        voxel_y = voxel_y[idx_arange]
        idx_pre = np.arange(np.shape(voxel_x)[0])
        fx = interpolate.interp1d(idx_pre, voxel_x, "zero")
        fy = interpolate.interp1d(idx_pre, voxel_y, "zero")
        xx_new = np.linspace(0, idx_pre[-1], 100)
        self.voxel_center_x = fx(xx_new)
        self.voxel_center_y = fy(xx_new)

    def pcd2d_to_3d(self, num_rows=5):
        num_points = np.shape(self.pcd2d)[0]
        self.pcd3d = np.zeros((num_points * num_rows, 3))
        self.pcd3d[:, 1:] = np.repeat(self.pcd2d, num_rows, axis=0)
        x = np.linspace(-0.2, 0.2, num_rows).reshape((-1, 1))
        xx = np.repeat(x, num_points, axis=1)
        # weights_diag = np.diag(np.linspace(0.0001, -0.0001, num_rows))
        weights_diag = np.diag(np.linspace(0, 0, num_rows))
        idx = np.arange(num_points)
        idx_m = np.repeat(idx.reshape((-1, 1)).T, num_rows, axis=0)
        xx = xx + np.matmul(weights_diag, idx_m)
        self.pcd3d[:, 0] = np.reshape(xx.T, (-1,))


def normalize_rotate(point, origin, theta):
    point_new = np.zeros_like(point)
    x = point[0]-origin[0]
    y = point[1]-origin[1]
    point_new[0] = np.cos(theta) * x + np.sin(theta) * y
    point_new[1] = -np.sin(theta) * x + np.cos(theta) * y
    return point_new

def get_fea_all(pcd_os: pcd_opreator_system, origin_x, origin_y, random_theta=0):
    fea_vec = np.zeros((13,))
    fea_vec[0] = pcd_os.corner_situation
    origin = np.array([origin_x, origin_y])
    if pcd_os.is_fea_A_gotten:
        Anew = normalize_rotate(pcd_os.Acenter, origin, random_theta)
        fea_vec[1:3] = Anew
    if pcd_os.is_fea_B_gotten:
        Bnew = normalize_rotate(pcd_os.Bcenter, origin, random_theta)
        fea_vec[3:5] = Bnew
    if pcd_os.is_fea_C_gotten:
        Cnew = normalize_rotate(pcd_os.Ccenter, origin, random_theta)
        fea_vec[5:7] = Cnew
    if pcd_os.is_fea_D_gotten:
        Dnew = normalize_rotate(pcd_os.Dcenter, origin, random_theta)
        fea_vec[7:9] = Dnew
    if pcd_os.is_fea_E_gotten:
        Enew = normalize_rotate(pcd_os.Ecenter, origin, random_theta)
        fea_vec[9:11] = Enew
    if pcd_os.is_fea_F_gotten:
        Fnew = normalize_rotate(pcd_os.Fcenter, origin, random_theta)
        fea_vec[11:-1] = Fnew
    return fea_vec


fea_save = []
pcd_voxel_save = []
fea_save_path = "/media/yuxuan/SSD/ENV_Fea_Train/fea/fea_trainset_3.npy"
pcd_voxel_save_path = "/media/yuxuan/SSD/ENV_Fea_Train/voxel/voxel_trainset3.npy"




str = input("按回车开始")
if __name__ == "__main__":
    imu_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
    num_points = int(38528 / down_sample_rate) + 1
    pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(num_points, 3))

    ax = plt.axes()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.ion()
    num_frame = 100

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
            o3d_voxel = open3d_voxelpipeline(pcd=env.pcd_2d)
            pcd2d_new = np.array([o3d_voxel.voxel_center_x, o3d_voxel.voxel_center_y]).T
            pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
            pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
            origin_x = np.min(pcd_os.pcd_new[:, 0])
            origin_y = pcd_os.pcd_new[np.argmin(pcd_os.pcd_new[:, 0]), 1]

            line_theta = pcd_os.env_rotate
            x = pcd2d_new[0:, 0]
            y = pcd2d_new[0:, 1]
            pcd2d_new[0:, 0] = np.cos(line_theta) * x + np.sin(line_theta) * y
            pcd2d_new[0:, 1] = -np.sin(line_theta) * x + np.cos(line_theta) * y
            pcd2d_new[0:, 0] = pcd2d_new[0:, 0]-origin_x
            pcd2d_new[0:, 1] = pcd2d_new[0:, 1]-origin_y

            # random_theta = random.uniform(-0.15, 0.15)
            random_theta = 0
            x = pcd2d_new[0:, 0]
            y = pcd2d_new[0:, 1]
            pcd2d_new[0:, 0] = np.cos(random_theta) * x + np.sin(random_theta) * y
            pcd2d_new[0:, 1] = -np.sin(random_theta) * x + np.cos(random_theta) * y

            pcd_os.show_(ax, pcd_color='r', id=int(i), downsample=1)
            if pcd_os.env_type == 1 or 3:
                if 8 >= pcd_os.corner_situation > 0:
                    fea_vec = get_fea_all(pcd_os, origin_x, origin_y, random_theta=random_theta)
                    fea_save.append(fea_vec)
                    pcd_voxel_save.append(pcd2d_new)
                    if pcd_os.is_fea_B_gotten:
                        B = fea_vec[3:5]
                        ax.scatter(B[0], B[1], color='g', linewidths=2)
                        ax.scatter(pcd2d_new[:, 0], pcd2d_new[:, 1], color='b', alpha=0.3)
                        print(np.max(pcd2d_new[:, 1]))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            print(np.shape(pcd2d_new))
            print(np.shape(env.pcd_thin))
            plt.draw()
            plt.pause(0.025)

    except KeyboardInterrupt:
        pass
    fea_save = np.array(fea_save)
    print(np.shape(fea_save))
    pcd_voxel_save = np.array(pcd_voxel_save)
    np.save(fea_save_path, fea_save)
    np.save(pcd_voxel_save_path, pcd_voxel_save)
