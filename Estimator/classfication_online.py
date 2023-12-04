import sys
import threading

sys.path.append("//")
import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import os
from Environment.Environment import *
from Environment.Plot_ import *
from scipy.spatial.transform import Rotation as R

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

# pcd降采样设置
down_sample_rate = 1
if down_sample_rate % 2 == 1:
    num_points = int(38527 / down_sample_rate) + 1
    if num_points > 38527:
        num_points = 38527
else:
    num_points = int(38527 / down_sample_rate)

# 画图设置
use_fastplot = False
use_statekf = False

env = Environment()
str = input("按回车开始")

imu_buffer = np.memmap("../Sensor/IM948/imu_thigh.npy", dtype='float32', mode='r', shape=(14,))
# imu_buffer = np.memmap("/home/yuxuan/Project/fsm_ysc/log/imu_euler_acc.npy", dtype='float32', mode='r',shape=(9 * 1 + 1,))
pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(num_points, 3))
terrain_type = np.memmap('/home/yuxuan/Project/fsm_ysc/log/environment_type.npy', dtype='int8', mode='w+', shape=(1,))
imu_data = np.zeros((14,))


# path_to_save_PCD = "/media/yuxuan/My Passport/testClassification2/PCD/"
# path_to_save_IMU = "/media/yuxuan/My Passport/testClassification2/IMU/"
# path_to_save_IMAGE = "/media/yuxuan/My Passport/testClassification2/IMAGE/3/"
# path_to_save_LABEL = "/media/yuxuan/My Passport/testClassification2/LABEL/3/"

path_to_save_IMAGE = "/media/yuxuan/My Passport/testClassification2/12/"
for path_ in [path_to_save_IMAGE]:
    f_list = os.listdir(path_)
    for file_ in f_list:
        os.remove(path_+file_)

# truth_label = 3

def pcd_to_binary_image(data, angle):
    angle = angle
    x = data[:, 0]
    y = -data[:, 1][abs(x) < 0.01]
    z = data[:, 2][abs(x) < 0.01]  # 降维

    img = np.zeros([100, 100])
    a1 = angle
    theta = (a1 / 180) * np.pi  # IMU测的欧拉角加个补偿，转成弧度

    y1 = (y * np.cos(theta) - z * np.sin(theta))  # 坐标系转换
    z1 = (z * np.cos(theta) + y * np.sin(theta))
    chosen_idx = np.logical_and(y1 < 1.2, y1 > 0.)
    pcd_y = y1[chosen_idx]
    pcd_z = z1[chosen_idx]

    if np.any(pcd_y):
        pcd_2d = np.zeros([len(pcd_y), 2])
        pcd_2d[:, 0] = pcd_y
        pcd_2d[:, 1] = -pcd_z

        p = pcd_y
        q = pcd_z
        p -= min(p)
        q -= min(q)
        q += 1 - max(q)

        for i in range(len(q)):
            if q[i] < 1 and q[i] > 0.01 and p[i] < 1 and p[i] > 0.01:
                p_int = int(100 * p[i])
                q_int = int(100 * q[i])
                img[q_int, p_int] = 1

        img = img * 255



    else:
        pcd_2d = np.zeros((len(y), 2))

    return img, pcd_2d


def imu_read_job():
    global imu_data
    while 1:
        imu_data = np.copy(imu_buffer[:])
        time.sleep(0.005)


def classfication_job():
    time.sleep(0.1)
    count = 0
    while 1:
        env.classification_from_img()
        img = cv2.cvtColor(env.elegant_img(), cv2.COLORMAP_RAINBOW)
        if env.type_pred_from_nn == 0:
            if env.theta > 0:
                env.type_pred_from_nn = 3
            elif env.theta < 0:
                env.type_pred_from_nn = 4
            # else:
            #     env.clear_slope_buffer()
        if env.type_pred_from_nn == 3 or env.type_pred_from_nn == 4:
            if env.theta < -1 and env.type_pred_from_nn == 3:
                env.type_pred_from_nn = 4
            elif env.theta > 1 and env.type_pred_from_nn == 4:
                env.type_pred_from_nn = 3
            if -5 < env.theta < 4:  # level-ground calibration
                env.type_pred_from_nn = 0
                print("矫正为平地:{}".format(env.theta))
                # env.clear_slope_buffer()
        terrain_type[:] = env.type_pred_from_nn
        terrain_type.flush()
        add_type(img, env_type=Env_Type(env.type_pred_from_nn), id=count)
        # add_para(img, paras, env_type=Env_Type(env.type_pred_from_nn))
        cv2.imshow("binary", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        time.sleep(0.05)
        count += 1
        # np.save(path_to_save_LABEL+"{}.npy".format(count), np.array([truth_label, env.type_pred_from_nn]))
        im = Image.fromarray(env.img_binary)
        im.save(path_to_save_IMAGE+"{}.png".format(count))

if __name__ == '__main__':
    num_frame = 90000

    # if use_fastplot == True:
    #     fast_plot_ax = FastPlotCanvas()
    # if use_statekf == True:
    #     state_kf = StateKalmanFilter()

    t0 = time.time()
    classfication_thread = threading.Thread(target=classfication_job)
    imu_thread = threading.Thread(target=imu_read_job)
    imu_thread.start()
    classfication_thread.start()
    try:
        for i in range(num_frame):
            # print("----------------------------Frame[{}]------------------------".format(i))
            # print("load binary image and pcd to process")
            # 先取pcd再取imu保障时间对齐
            pcd_data_temp = np.copy(pcd_buffer[:])
            eular_angle = np.copy(imu_data)
            eular_angle = eular_angle[10:]
            eular_angle = R.from_quat([eular_angle[1],eular_angle[2],
                                       eular_angle[3],eular_angle[0]]).as_euler('xyz',degrees=True)
            # eular_angle = imu_data[0:3]
            env.pcd_to_binary_image(pcd_data_temp, eular_angle)
            # CCH Code
            # env.img_binary, env.pcd_2d = pcd_to_binary_image(pcd_data_temp, eular_angle[0])
            t = time.time()
            # np.save(path_to_save_PCD+"{}.npy".format(t), pcd_data_temp)
            # np.save(path_to_save_IMU+"{}.npy".format(t), eular_angle)
            paras = [0, 0, 0]
            if env.type_pred_from_nn == 0:
                paras = env.get_theta()
                print("分类为平地的坡度:{}".format(env.theta))
            if env.type_pred_from_nn == 1 or env.type_pred_from_nn == 2:
                paras = env.get_W_H()
                env.clear_slope_buffer()
            elif env.type_pred_from_nn == 3 or env.type_pred_from_nn == 4:
                paras = env.get_theta()
                print("分类为斜坡的坡度:{}".format(env.theta))
            time.sleep(0.1)
            # print("=======////Totol Time:{}\\\====".format(dt))

    except KeyboardInterrupt:
        pass
