import sys

sys.path.append("/home/yuxuan/Project/MPV_2024/")
import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL
import os
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *
from Environment.Plot_ import *
from Estimator.fusion_algo import StateKalmanFilter
from Utils.IO import fifo_data_vec
from Utils.pcd_os_fast_plot import *

# 存储设置
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
kalman_x_buffer = []
kalman_y_buffer = []
time_buffer = []
pcd_os_buffer = [[], []]
env_paras_buffer = [[], []]
hbx_pcd2d_save = []

# pcd降采样设置
down_sample_rate = 3
if down_sample_rate % 2 == 1:
    num_points = int(38528 / down_sample_rate) + 1
    if num_points > 38528:
        num_points = 38528
else:
    num_points = int(38528 / down_sample_rate)

# 画图设置
use_fastplot = False
use_statekf = False

env = Environment()
str = input("按回车开始")

if __name__ == '__main__':
    imu_buffer = np.memmap("../Sensor/IM948/imu_thigh.npy", dtype='float32', mode='r', shape=(14,))
    # imu_buffer = np.memmap("/home/yuxuan/Project/fsm_ysc/log/imu_euler_acc.npy", dtype='float32', mode='r',shape=(9 * 1 + 1,))
    pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(num_points, 3))
    terrain_type = np.memmap('/home/yuxuan/Project/fsm_ysc/log/environment_type.npy', dtype='int8', mode='w+',
                             shape=(1,))
    num_frame = 90000

    if use_fastplot == True:
        fast_plot_ax = FastPlotCanvas()
    if use_statekf == True:
        state_kf = StateKalmanFilter()

    t0 = time.time()
    imu_initial_buffer = []
    count = 0
    while time.time() - t0 < 0.5:
        imu_data = imu_buffer[0:]
        imu_initial_buffer.append(imu_data)
        count += 1
        time.sleep(0.001)
    imu_initial = np.mean(np.array(imu_initial_buffer), axis=0)

    try:
        for i in range(num_frame):
            # print("----------------------------Frame[{}]------------------------".format(i))
            # print("load binary image and pcd to process")
            pcd_data_temp = pcd_buffer[:]
            imu_data = imu_buffer[0:]
            eular_angle = imu_data[7:10]
            # eular_angle = imu_data[0:3]
            env.pcd_to_binary_image(pcd_data_temp, eular_angle)
            env.classification_from_img()
            img = cv2.cvtColor(env.elegant_img(), cv2.COLORMAP_RAINBOW)
            paras = [0, 0, 0]
            if env.type_pred_from_nn == 1 or env.type_pred_from_nn == 2:
                paras = env.get_W_H()
            elif env.type_pred_from_nn == 3 or env.type_pred_from_nn == 4:
                paras = env.get_theta()
                if abs(paras[0]) < 6: # level-ground calibration
                    env.type_pred_from_nn = 0
            add_type(img, env_type=Env_Type(env.type_pred_from_nn), id=i)
            add_para(img, paras, env_type=Env_Type(env.type_pred_from_nn))
            # given env_type to fsm
            terrain_type[:] = env.type_pred_from_nn
            terrain_type.flush()
            cv2.imshow("binary", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            time.sleep(0.01)
            # print("=======////Totol Time:{}\\\====".format(dt))

    except KeyboardInterrupt:
        pass
