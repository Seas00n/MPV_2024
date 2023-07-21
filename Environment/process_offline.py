import os

import cv2
import open3d as o3d
import numpy as np
from lcm_msg.pcd_lcm.pcd_xyz import *
from lcm_msg.lcm_init import *
import lcm
import time
from Environment import *

imu_buffer_path = "../Sensor/IM948/imu_buffer.npy"
data_save_path = "/media/yuxuan/Ubuntu 20.0/IMG_TEST/TEST1/"

env = Environment()

if __name__ == "__main__":
    file_list = os.listdir(data_save_path)
    img_list = file_list[1:]
    num_frame = int(len(img_list) / 2)

    xmove = 0
    ymove = 0

    try:
        for i in range(num_frame):
            env.img_binary = np.load(data_save_path + img_list[i])
            env.pcd_2d = np.load(data_save_path+img_list[i+num_frame])
            env.thin()
            cv2.imshow("feature_image", env.elegant_img())
            key = cv2.waitKey(1)
            time.sleep(0.01)
            print(i)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
