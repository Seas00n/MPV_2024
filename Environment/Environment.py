import io
import statistics

import cv2
import numpy as np
import torch
from enum import Enum
from scipy.spatial.transform import Rotation
from Utils.IO import *
import scipy
from Utils.Algo import *
from sklearn.neighbors import NearestNeighbors


def pcd2d_to_3d(pcd_2d, num_rows=5):
    num_points = np.shape(pcd_2d)[0]
    pcd_3d = np.zeros((num_points * num_rows, 3))
    pcd_3d[:, 1:] = np.repeat(pcd_2d, num_rows, axis=0)
    x = np.linspace(-0.2, 0.2, num_rows).reshape((-1, 1))
    xx = np.repeat(x, num_points, axis=1)
    # weights_diag = np.diag(np.linspace(0.0001, -0.0001, num_rows))
    weights_diag = np.diag(np.linspace(0, 0, num_rows))
    idx = np.arange(num_points)
    idx_m = np.repeat(idx.reshape((-1, 1)).T, num_rows, axis=0)
    xx = xx + np.matmul(weights_diag, idx_m)
    pcd_3d[:, 0] = np.reshape(xx.T, (-1,))
    return pcd_3d


class Env_Type(Enum):
    Levelground = 0
    Upstair = 1
    DownStair = 2
    Upslope = 3
    Downslope = 4
    Obstacle = 5
    Unknown = 6


class Environment:
    def __init__(self):
        self.classification_model = torch.load('/home/yuxuan/Project/CCH_Model/realworld_model_epoch_29.pt',
                                               map_location=torch.device('cpu'))
        self.type_pred_from_nn = Env_Type.Levelground
        self.type_pred_buffer = np.zeros(3, dtype=np.uint64)
        self.pcd_2d = np.zeros([0, 2])
        self.pcd_thin = np.zeros([0, 2])
        self.img_binary = np.zeros((100, 100)).astype('uint8')
        self.R_world_imu = np.identity(3)
        self.R_world_camera = np.identity(3)
        self.R_world_body = np.identity(3)
        self.R_imu_camera = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()

    def pcd_to_binary_image(self, pcd, imu):
        eular = imu[0:3]
        # imu在世界坐标系下的位姿
        self.R_world_imu = Rotation.from_euler('xyz', [eular[0], eular[1], eular[2]],
                                               degrees=True).as_matrix()
        # camera 在世界坐标系下的位姿
        self.R_world_camera = np.matmul(self.R_world_imu, self.R_imu_camera)

        # body 在世界坐标系下的位姿
        R_body_imu = Rotation.from_euler('xyz', [eular[0] - 90, 0, 180], degrees=True).as_matrix()
        self.R_world_body = np.matmul(self.R_world_imu, R_body_imu.T)

        # pcd在body坐标系下的表示
        R_body_camera = np.matmul(R_body_imu, self.R_imu_camera)

        # 相机安装在机器左测，实际地形取相机偏右侧的部分对应机器正中间
        chosen_idx = np.logical_and(pcd[:, 0] < 0.05, pcd[:, 0] > 0.02)
        pcd_chosen = pcd[chosen_idx, :]
        pcd_chosen_in_body = np.matmul(R_body_camera, pcd_chosen.T).T
        # 选取z0.1-2距离的点
        chosen_idx = np.logical_and(pcd_chosen_in_body[:, 1] < 2.5, pcd_chosen_in_body[:, 1] > 0.01)
        chosen_y = pcd_chosen_in_body[chosen_idx, 1]
        chosen_z = pcd_chosen_in_body[chosen_idx, 2]
        self.img_binary = np.zeros((100, 100)).astype('uint8')
        if np.any(chosen_y):
            self.pcd_2d = pcd_chosen_in_body[chosen_idx, 2:0:-1]
            self.pcd_2d[:, 1] = -self.pcd_2d[:, 1]
            y_max = np.max(chosen_y)
            z_min = np.min(chosen_z)
            chosen_y = pcd_chosen_in_body[chosen_idx, 1]
            chosen_z = pcd_chosen_in_body[chosen_idx, 2]
            # 和z=0,y=1对齐
            y_max = np.max(chosen_y)
            z_min = np.min(chosen_z)
            chosen_y = chosen_y + (0.99 - y_max)
            chosen_z = chosen_z + (0.01 - z_min)
            # 只取出最前方1m^2内的点
            chosen_idx = np.logical_and(chosen_y > 0, chosen_z < 1)
            chosen_y = chosen_y[chosen_idx]
            chosen_z = chosen_z[chosen_idx]
            pixel_y = np.floor(100 * chosen_y).astype('int')
            pixel_z = np.floor(100 * chosen_z).astype('int')
            self.img_binary = np.zeros((100, 100)).astype('uint8')
            for i in range(np.size(pixel_y)):
                self.img_binary[pixel_y[i], pixel_z[i]] = 255
        else:
            self.pcd_2d = np.zeros([len(chosen_y), 2])

    def classification_from_img(self):
        img_input = self.img_binary.reshape(1, 1, 100, 100).astype('uint8')
        img_input = torch.tensor(img_input, dtype=torch.float)
        with torch.no_grad():
            output = self.classification_model(img_input)
        pred = output.data.max(1)[1].cpu().numpy()
        self.type_pred_buffer = fifo_data_vec(self.type_pred_buffer, pred)
        self.type_pred_from_nn = statistics.mode(self.type_pred_buffer)

    def elegant_img(self):
        img = np.zeros((500, 500, 3)).astype('uint8')
        img[:, :, 0] = cv2.resize(self.img_binary, (500, 500))
        img[:, :, 1] = cv2.resize(self.img_binary, (500, 500))
        img[:, :, 2] = cv2.resize(self.img_binary, (500, 500))
        return img

    def thin(self):
        nb1 = NearestNeighbors(n_neighbors=20, algorithm='auto')
        nb1.fit(self.pcd_2d)
        _, idx = nb1.kneighbors(self.pcd_2d)
        self.pcd_thin = np.mean(self.pcd_2d[idx, :], axis=1)

        xmin = np.min(self.pcd_thin[:, 0])
        idx_chosen = np.where(self.pcd_thin[:, 0] - xmin < 1)[0]
        self.pcd_thin = self.pcd_thin[idx_chosen, :]

        if self.pcd_thin[np.argmin(self.pcd_thin[:, 0]), 1] > self.pcd_thin[np.argmax(self.pcd_thin[:, 0]), 1] + 0.1:
            idx_chosen = np.where(self.pcd_thin[:, 1] < -0.4)[0]  # 下楼处理邻近噪声
        else:
            idx_chosen = np.where(self.pcd_thin[:, 1] < -0.1)[0]  # 上楼处理邻近噪声
        self.pcd_thin = self.pcd_thin[idx_chosen, :]

        mean_x = np.mean(self.pcd_thin[:, 0])
        sigma_x = np.std(self.pcd_thin[:, 0])
        mean_y = np.mean(self.pcd_thin[:, 1])
        sigma_y = np.std(self.pcd_thin[:, 1])
        idx_remove = np.logical_and(np.abs(self.pcd_thin[:, 0] - mean_x) > 3 * sigma_x,
                                    np.abs(self.pcd_thin[:, 1] - mean_y) > 3 * sigma_y)
        self.pcd_thin = np.delete(self.pcd_thin, idx_remove, axis=0)

        ymax = np.max(self.pcd_thin[:, 1])
        idx_remove = np.where(ymax - self.pcd_thin[:, 1] < 0.02)[0]
        line_max = self.pcd_thin[idx_remove, :]
        if np.max(line_max[:, 0]) - np.min(line_max[:, 0]) < 0.02:
            self.pcd_thin = np.delete(self.pcd_thin, idx_remove, axis=0)

        ymin = np.min(self.pcd_thin[:, 1])
        idx_remove = np.where(self.pcd_thin[:, 1] - ymin < 0.02)[0]
        line_min = self.pcd_thin[idx_remove, :]
        if np.max(line_min[:, 0]) - np.min(line_min[:, 0]) < 0.02:
            self.pcd_thin = np.delete(self.pcd_thin, idx_remove, axis=0)
        self.pcd_thin = self.pcd_thin[0:-1:2, :]

    def get_fea_sa(self):
        xc = yc = w = h = 0
        X = self.pcd_thin[:, 0]
        Y = self.pcd_thin[:, 1]
        if np.max(Y) - np.min(Y) < 0.1:
            print('点云高度差过小，当前地形应该是平地')
            return xc, yc, w, h
        th = 0.05
        X0 = X[Y - np.min(Y) < 0.25].reshape((-1, 1))
        Y0 = Y[Y - np.min(Y) < 0.25].reshape((-1, 1))
        try:
            inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
            mean_y1 = np.mean(Y0[inlier_mask1])
            idx1 = np.where(abs(Y0 - mean_y1) < 0.08)[0]  # 0.08
            x1 = X0[idx1, :]
            y1 = Y0[idx1, :]
            X1 = np.delete(X0, idx1).reshape((-1, 1))
            Y1 = np.delete(Y0, idx1).reshape((-1, 1))
        except:
            print("第一次拟合失败")
            return xc, yc, w, h
        try:
            inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X1, Y1, th)
            mean_y2 = np.mean(Y1[inlier_mask2])
            idx2 = np.where(abs(Y1 - mean_y2) < 0.08)[0]  # 0.08
            x2 = X1[idx2, :]
            y2 = Y1[idx2, :]
        except:
            print("第二次拟合错误，相机视角有问题或被阻挡")
            left_down_conner_y = np.min(Y1)
            right_up_conner_y = np.max(Y1)
            height_from_left_conner = mean_y1 - left_down_conner_y
            height_to_right_conner = right_up_conner_y - mean_y1
            # -----------------------#
            #      %>>>>>>          #
            #      ?                #
            #      ?                #
            #      ?                #
            # -----------------------#
            if height_from_left_conner > 0.05 and np.min(x1) > -0.05:
                xc = np.min(x1)
                h = height_from_left_conner
                yc = np.mean(y1)
                w = 0
            # -----------------------#
            #           %            #
            #           ?            #
            #   ?>>>>>>>?            #
            #   ?                    #
            # -----------------------#
            elif height_to_right_conner > 0.05:
                xc = np.max(x1)
                h = height_to_right_conner
                yc = np.max(Y1)
                w = 0
            else:
                xc = 0
                yc = 0
                h = 0
            return xc, yc, w, h
        if len(x1) * len(x2) == 0:
            return xc, yc, w, h
        if mean_y1 > mean_y2:
            stair_high_x, stair_high_y = x1, y1
            stair_low_x, stair_low_y = x2, y2
        else:
            stair_high_x, stair_high_y = x2, y2
            stair_low_x, stair_low_y = x1, y1
        w = np.max([np.max(stair_high_x) - np.max(stair_low_x),
                    np.min(stair_high_x) - np.min(stair_low_x)])
        h = np.mean(stair_high_y) - np.mean(stair_low_y)
        if np.mean(stair_low_y) - np.min(Y1) > 0.05 and np.min(stair_low_x) > -0.05:
            xc = np.min(stair_low_x)
            yc = np.mean(stair_low_y)
        else:
            xc = np.min(stair_high_x)
            yc = np.mean(stair_high_y)
        if w > 0.35 and np.max(stair_low_x) - np.min(stair_low_x) > 0.35:
            print("第一节台阶")
            w = np.max([np.max(stair_high_x) - np.max(stair_low_x),
                        np.max(stair_high_x) - np.min(stair_high_x)])
        elif w > 0.35 and np.max(stair_high_x) - np.min(stair_high_x) > 0.35:
            print("最后一个台阶")
            w = np.max([np.min(stair_high_x) - np.min(stair_low_x),
                        np.max(stair_low_x) - np.min(stair_low_x)])
        return xc, yc, w, h
