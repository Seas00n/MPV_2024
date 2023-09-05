import datetime
import sys

sys.path.append("/home/yuxuan/Project/MPV_2024/")
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
from Environment.Plot_ import *
from Utils.IO import fifo_data_vec
from Utils.pcd_os_fast_plot import *
import fusion_algo


# 存储设置
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
fusion_x_buffer = []
fusion_y_buffer = []
time_buffer = [datetime.datetime.now()]
pcd_os_buffer = [[], []]

# 降采样设置
down_sample_rate = 5

# 画图设置
plot_3d = False
use_fastplot = True


# 离线测试
use_data_set = False
data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4/"

# 在线测试
use_lcm = False
pcd_data = np.zeros((int(38528 / down_sample_rate) + 1, 3))
pcd_data_temp = np.zeros((int(38528 / down_sample_rate) + 1, 3))

# 传感融合
use_fusion = False
imu_params = fusion_algo.ImuParameters()
imu_params.sigma_a_n = 0.001221
imu_params.sigma_a_b = 0.000048
init_nomial_state = np.zeros((11,))
init_nomial_state[6:8] = np.array([0, -9.81])
sigma_measurement_p = 0.000025
sigma_measurement = np.eye(2) * (sigma_measurement_p ** 2)
estimator = fusion_algo.ESEKF(init_nominal_state=init_nomial_state,
                              imu_parameters=imu_params)


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
        imu_dataset = np.load(data_save_path + "imu_data.npy")
        imu_dataset = imu_dataset[1:, :]
        idx_frame = np.arange(np.shape(imu_dataset)[0])
        num_frame = np.shape(idx_frame)[0]
    else:
        imu_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
        if use_lcm:
            pcd_msg, pcd_lc = pcd_lcm_initialize()
            subscriber = pcd_lc.subscribe("PCD_DATA", pcd_handler)
        else:
            num_points = int(38528 / down_sample_rate) + 1
            pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r',
                                   shape=(num_points, 3))
        t0 = time.time()
        num_frame = 1000


    if plot_3d:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        plt.ion()
    else:
        if not use_fastplot:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.ion()
        else:
            fast_plot_ax = FastPlotCanvas()

    t0 = time.time()
    imu_initial_buffer = []
    count = 0
    while time.time() - t0 < 0.5:
        if use_data_set:
            imu_initial_buffer.append(imu_dataset[count, :])
            count += 1
            time.sleep(0.001)
        else:
            imu_data = imu_buffer[0:]
            imu_initial_buffer.append(imu_data)
            count += 1
            time.sleep(0.001)
    imu_initial = np.mean(np.array(imu_initial_buffer), axis=0)
    init_nomial_state[8:11] = imu_initial[7:10]

    try:
        for i in range(num_frame):
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")


            if not use_fastplot:
                plt.cla()


            if use_data_set:
                env.img_binary = np.load(data_save_path + "{}_img.npy".format(i))
                env.pcd_2d = np.load(data_save_path + "{}_pcd.npy".format(i))
                imu_data = imu_data[i, :]
            else:
                if use_lcm:
                    pcd_lc.handle()
                else:
                    pcd_data_temp = pcd_buffer[:]
                imu_data = imu_buffer[0:]
                eular_angle = imu_data[7:10]
                env.pcd_to_binary_image(pcd_data_temp, eular_angle)


            t0 = datetime.datetime.now()
            env.thin()
            t1 = datetime.datetime.now()
            print("############Preprocess:{}#########".format((t1 - t0).total_seconds()*1000))


            if i == 0:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
                camera_x_buffer.append(0)
                camera_y_buffer.append(0)
                fusion_x_buffer.append(0)
                fusion_y_buffer.append(0)
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
                t0 = datetime.datetime.now()
                pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn, ax=None)
                t1 = datetime.datetime.now()
                print("###########FeatureExtra:{}###########".format((t1 - t0).total_seconds() * 1000))
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
                t0 = datetime.datetime.now()
                fea_to_align_new, fea_to_align_pre, flag_method = align_fea(pcd_new=pcd_new_os,
                                                                            pcd_pre=pcd_pre_os,
                                                                            _print_=True)
                t1 = datetime.datetime.now()
                print("##############=====FeatureRule:{}====############".format((t1 - t0).total_seconds() * 1000))

                try:
                    if flag_method == 0:
                        t0 = datetime.datetime.now()
                        xmove, ymove = icp_knn(pcd_s=fea_to_align_pre,
                                               pcd_t=fea_to_align_new)
                        t1 = datetime.datetime.now()
                        print("#################=====FeatureAlign:{}=====#########".format((t1 - t0).total_seconds() * 1000))
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

                if use_fusion:
                    imu_for_predict = np.array([imu_data[6:9], imu_data[0:3], imu_data[3:6]]).reshape((-1,))
                    if use_data_set:
                        time_stamp = round(imu_data[-1], 3)
                    else:
                        time_now = datetime.datetime.now()
                        time_stamp = (time_now - time_buffer[-1]).total_seconds()
                    estimator.predict(imu_for_predict, time_stamp)
                    estimator.update(np.array([camera_x_buffer[-1], camera_y_buffer[-1]]), sigma_measurement)
                    frame_pose = estimator.nominal_state[:2].copy()
                    fusion_x_buffer.append(frame_pose[0])
                    fusion_y_buffer.append(frame_pose[1])

                else:
                    fusion_x_buffer.append(camera_x_buffer[-1])
                    fusion_y_buffer.append(camera_y_buffer[-1])

                if plot_3d:
                    continue
                else:
                    t0 = datetime.datetime.now()
                    if not use_fastplot:
                        pcd_new_os.show_(ax, pcd_color='r', id=int(i), downsample=8)
                        # pcd_pre_os.show_(ax, pcd_color='b', id=int(i - 1), p_text=0.4, p_pcd=[-xmove, -ymove])
                        ax.set_xlim(-0.5, 1.5)
                        ax.set_ylim(-1.5, 0.5)
                        plt.draw()
                        plt.pause(0.0001)
                    else:
                        pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=-0.3, downsample=8)
                    t1 = datetime.datetime.now()
                    print("###############=====Plot:{}=====#########".format((t1 - t0).total_seconds() * 1000))


            # img = cv2.cvtColor(env.elegant_img(), cv2.COLORMAP_RAINBOW)
            # add_type(img, env_type=Env_Type(env.type_pred_from_nn), id=i)
            # cv2.imshow("binary", img)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break


            time_buffer.append(datetime.datetime.now())
            print("===//////==total:{}==\\\\\\===".format((time_buffer[-1]-time_buffer[-2]).total_seconds()*1000))



    except KeyboardInterrupt:
        pass

    print((time_buffer[-1] - time_buffer[0]).total_seconds())
