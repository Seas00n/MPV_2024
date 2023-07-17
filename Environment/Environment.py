import cv2
import numpy as np
import torch
from enum import Enum
from scipy.spatial.transform import Rotation


class Env_Type(Enum):
    Levelground = 0
    Slope = 1
    Upstair = 2
    Downstair = 3
    Obstacle = 4


class Environment:
    def __init__(self):
        self.classification_model = torch.load('model/realworld_model_epoch_29.pt',
                                               map_location=torch.device('cpu'))
        self.env_type = Env_Type.Levelground

    def pcd_to_binary_image(self, pcd, imu):
        eular = imu[0:3]
        chosen_idx = np.logical_and(pcd[:, 0] < 0.05, pcd[:, 0] > 0.02, abs(pcd[:, 1]) < 2)
        pcd_new = pcd[chosen_idx, :]
        R = Rotation.from_euler('xyz', [eular[0]-90.0, 0, 0], degrees=True).as_matrix()
        pcd_new = np.matmul(R, pcd_new.T)
        pcd_new = pcd_new.T
        chosen_y = pcd_new[:, 1]
        chosen_z = pcd_new[:, 2]
        img_pcd = np.zeros((100, 100)).astype('uint8')
        if np.any(chosen_y):
            y_min = np.min(chosen_y)
            z_min = np.min(chosen_z)
            pcd_2d = np.zeros([len(chosen_y), 2])
            pcd_2d[:, 0] = chosen_z
            pcd_2d[:, 1] = chosen_y
            chosen_y = chosen_y - y_min
            chosen_z = chosen_z - z_min
            y_max = np.max(chosen_y)
            chosen_y = chosen_y + (1 - y_max)  # align with 1
            chosen_idx = np.logical_and(abs(chosen_y)< 1, abs(chosen_z) < 1)
            chosen_y = chosen_y[chosen_idx]
            chosen_z = chosen_z[chosen_idx]
            chosen_idx = np.logical_and(abs(chosen_y)> 0.01, abs(chosen_z) > 0.01)
            chosen_y = chosen_y[chosen_idx]
            chosen_z = chosen_z[chosen_idx]
            pixel_y = np.floor(100 * chosen_y).astype('int')
            pixel_z = np.floor(100 * chosen_z).astype('int')
            for i in range(np.size(pixel_y)):
                img_pcd[pixel_y[i], pixel_z[i]] = 255
        else:
            pcd_2d = np.zeros([len(chosen_y), 2])
        return img_pcd, pcd_2d
