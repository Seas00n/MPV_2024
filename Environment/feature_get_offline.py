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
data_save_path = "/media/yuxuan/Ubuntu 20.0/IMG_TEST/TEST4/"

env = Environment()

if __name__ == "__main__":
    file_list = os.listdir(data_save_path)
    img_list = file_list[0:-1]
    num_img = len(img_list)
    try:
        for i in range(num_img):
            img = cv2.imread(data_save_path + img_list[i],cv2.IMREAD_GRAYSCALE)
            cv2.imshow("binaryimage", cv2.resize(img, (500, 500)))
            key = cv2.waitKey(1)
            time.sleep(0.1)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
