import cv2
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np
from lcm_msg.pcd_lcm.pcd_xyz import *
from lcm_msg.lcm_init import *
import lcm
import time
import PIL
import os
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *
from Utils.IO import fifo_data_vec

# 存储设置
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
pcd_os_buffer = [[], []]

# 画图设置
plot_3d = False

# 离线测试
use_data_set = False
data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4/"

# 在线测试
use_lcm = False
pcd_data = np.zeros((38528, 3))
pcd_data_temp = np.zeros((38528, 3))


def pcd_handler(channel, data):
    global pcd_data_temp
    msg = pcd_xyz.decode(data)
    pcd_data[:, 0] = np.array(msg.pcd_x)
    pcd_data[:, 1] = np.array(msg.pcd_y)
    pcd_data[:, 2] = np.array(msg.pcd_z)
    pcd_data_temp = (pcd_data - 10000) / 300.0  # int16_t to float


env = Environment()
str = input("按回车开始")

if __name__ == "__main__":
    if use_data_set:
        imu_data = np.load(data_save_path + "imu_data.npy")
        imu_data = imu_data[1:, :]
        idx_frame = np.arange(np.shape(imu_data)[0])
        num_frame = np.shape(idx_frame)[0]
    else:
        imu_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
        if use_lcm:
            pcd_msg, pcd_lc = pcd_lcm_initialize()
            subscriber = pcd_lc.subscribe("PCD_DATA", pcd_handler)
        else:
            pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(38528, 3))
        t0 = time.time()
        num_frame = 1000

    fig = plt.figure(figsize=(5, 5))

    if plot_3d:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    plt.ion()

    try:
        for i in range(num_frame):
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            plt.cla()
            if use_data_set:
                env.img_binary = np.load(data_save_path + "{}_img.npy".format(i))
                env.pcd_2d = np.load(data_save_path + "{}_pcd.npy".format(i))
            else:
                if use_lcm:
                    pcd_lc.handle()
                else:
                    pcd_data_temp = pcd_buffer[:]
                imu_data = imu_buffer[0:]
                eular_angle = imu_data[7:10]
                env.pcd_to_binary_image(pcd_data_temp, eular_angle)

            env.thin()
            if i == 0:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
                camera_x_buffer.append(0)
                camera_y_buffer.append(0)
                pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
                env.classification_from_img()
                pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_os)
            else:
                pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_opreator_system
                pcd_pre = pcd_pre_os.pcd_new
                pcd_new, pcd_new_os = env.pcd_thin, pcd_opreator_system(env.pcd_thin)
                xmove, ymove = 0, 0
                env.classification_from_img()
                pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
                fea_to_align_new, fea_to_align_pre, flag_method = align_fea(pcd_new=pcd_new_os,
                                                                            pcd_pre=pcd_pre_os,
                                                                            _print_=True)
                try:
                    if flag_method == 0:
                        xmove, ymove = icp_knn(pcd_s=fea_to_align_pre,
                                               pcd_t=fea_to_align_new)
                except Exception as e:
                    print("align method exception:{}".format(e))
                    xmove, ymove = 0, 0
                    flag_method = 1

                xmove_pre = camera_dx_buffer[-1]
                ymove_pre = camera_dy_buffer[-1]
                if flag_method == 1 or abs(xmove) > 0.1 or abs(ymove) > 0.1:
                    xmove = xmove_pre
                    ymove = ymove_pre
                    print("对齐失败，使用上一次的xmove_pre = {},ymove_pre = {}".format(xmove_pre, ymove_pre))

                print("当前最终xmove = {}, ymove = {}".format(xmove, ymove))
                camera_dx_buffer.append(xmove)
                camera_dy_buffer.append(ymove)
                camera_x_buffer.append(camera_x_buffer[-1] + xmove)
                camera_y_buffer.append(camera_y_buffer[-1] + ymove)

                if plot_3d:
                    continue
                else:
                    pcd_new_os.show_(ax, pcd_color='r', id=int(i))
                    # pcd_pre_os.show_(ax, pcd_color='b', id=int(i - 1), p_text=0.4, p_pcd=[-xmove, -ymove])
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    plt.draw()
                    plt.pause(0.001)
            # cv2.imshow("binaryimage", env.elegant_img())
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break
    except KeyboardInterrupt:
        pass
