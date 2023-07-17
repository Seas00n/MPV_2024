import cv2
import numpy as np
import torch
from enum import Enum
from scipy.spatial.transform import Rotation
from Utils.IO import *
import scipy
from Utils.Algo import *


class Env_Type(Enum):
    Levelground = 0
    Slope = 1
    Upstair = 2
    Downstair = 3
    Obstacle = 4


class Environment:
    def __init__(self):
        self.classification_model = torch.load('/home/yuxuan/Project/CCH_Model/realworld_model_epoch49.pt',
                                               map_location=torch.device('cpu'))
        self.env_type_from_nn = Env_Type.Levelground
        self.env_type_buffer = np.zeros(5, dtype=np.uint64)
        self.pcd_2d = np.zeros([0, 2])
        self.img_binary = np.zeros((100, 100)).astype('uint8')

    def pcd_to_binary_image(self, pcd, imu):
        eular = imu[0:3]
        chosen_idx = np.logical_and(pcd[:, 0] < 0.05, pcd[:, 0] > 0.02, abs(pcd[:, 1]) < 2)
        pcd_new = pcd[chosen_idx, :]
        R = Rotation.from_euler('xyz', [eular[0] - 90.0, 0, 0], degrees=True).as_matrix()
        pcd_new = np.matmul(R, pcd_new.T)
        pcd_new = pcd_new.T
        chosen_y = pcd_new[:, 1]
        chosen_z = pcd_new[:, 2]
        self.img_binary = np.zeros((100, 100)).astype('uint8')
        if np.any(chosen_y):
            y_min = np.min(chosen_y)
            z_min = np.min(chosen_z)
            self.pcd_2d = np.zeros([len(chosen_y), 2])
            self.pcd_2d[:, 0] = chosen_z
            self.pcd_2d[:, 1] = chosen_y
            chosen_y = chosen_y - y_min
            chosen_z = chosen_z - z_min
            y_max = np.max(chosen_y)
            chosen_y = chosen_y + (1 - y_max)  # align with 1
            chosen_idx = np.logical_and(abs(chosen_y) < 1, abs(chosen_z) < 1)
            chosen_y = chosen_y[chosen_idx]
            chosen_z = chosen_z[chosen_idx]
            chosen_idx = np.logical_and(abs(chosen_y) > 0.01, abs(chosen_z) > 0.01)
            chosen_y = chosen_y[chosen_idx]
            chosen_z = chosen_z[chosen_idx]
            pixel_y = np.floor(100 * chosen_y).astype('int')
            pixel_z = np.floor(100 * chosen_z).astype('int')
            for i in range(np.size(pixel_y)):
                self.img_binary[pixel_y[i], pixel_z[i]] = 255

        else:
            self.pcd_2d = np.zeros([len(chosen_y), 2])

    def classification_from_img(self):
        img = np.asarray(self.img_binary, dtype=np.uint8).reshape(1, 1, 100, 100)
        img_torch = torch.tensor(img, dtype=torch.float)
        with torch.no_grad():
            output = self.classification_model(img_torch)

        pred = output.data.max(1)[1].cpu().numpy()
        self.env_type_buffer = fifo_data_vec(self.env_type_buffer, pred)
        pred = scipy.stats.mode(self.env_type_buffer)[0]
        self.env_type_from_nn = Env_Type(pred)

    def elegent_img(self):
        return cv2.resize(self.img_binary, (500, 500))

    def get_feature_ransac(self, flag):
        X = self.pcd_2d[:, 0]
        Y = self.pcd_2d[:, 1]
        ymin = min(Y)
        th = 0.05
        if flag == 0:
            X0 = X
            Y0 = Y
        else:
            X0 = X[np.where(Y - ymin > 0.06)[0]]
            Y0 = Y[np.where(Y - ymin > 0.06)[0]]
        inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
        y1 = Y0[inlier_mask1]
        mean_y1 = np.mean(y1)
        idx1 = np.where(abs(Y0 - mean_y1) < 0.008)[0]
        # 以第一次拟合的直线的y坐标均值为中心，选取在其上下一定范围内的点作为第一条直线，
        # 避免单纯依靠RANSAC拟合的直线太细，从而使第二次拟合的直线也在这层台阶上
        x1 = X0[idx1]
        y1 = Y0[idx1]
        X1 = np.delete(X0, idx1)
        Y1 = np.delete(Y0, idx1)

        inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X1, Y1, th)
        y2 = Y1[inlier_mask2]
        mean_y2 = np.mean(y2)
        idx2 = np.where(abs(Y1 - mean_y2) < 0.008)[0]
        # 以第一次拟合的直线的y坐标均值为中心，选取在其上下一定范围内的点作为第一条直线，
        # 避免单纯依靠RANSAC拟合的直线太细，从而使第二次拟合的直线也在这层台阶上
        x2 = X1[idx2]
        y2 = Y1[idx2]
        if mean_y1 > mean_y2:
            stair_high_x, stair_high_y = x1, y1
            stair_low_x, stair_low_y = x2, y2
        else:
            stair_high_x, stair_high_y = x2, y2
            stair_low_x, stair_low_y = x1, y1

        vertical_mean_x3 = (max(stair_high_x) + min(stair_low_x)) / 2
        vertical_idx3 = np.where(abs(X1 - vertical_mean_x3) < 0.006)[0]

        stair_high_fea_x = stair_high_x[stair_high_x - min(stair_high_x) < 0.03]
        stair_high_fea_y = stair_high_y[stair_high_x - min(stair_high_x) < 0.03]
        stair_high_fea = np.concatenate((stair_high_fea_x,stair_high_fea_y),1)

        if len(vertical_idx3)<1000:
            fea = None
        else:
            vertical_fea_x = X1[vertical_idx3]
            vertical_fea_y = Y1[vertical_idx3]
            vertical_fea = np.concatenate((vertical_fea_x,vertical_fea_y),1)

        return stair_high_fea, vertical_fea