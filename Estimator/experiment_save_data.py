import sys
sys.path.append("//")
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
down_sample_rate = 1
if down_sample_rate % 2 == 1:
    num_points = int(38528 / down_sample_rate) + 1
    if num_points > 38528:
        num_points = 38527
else:
    num_points = int(38528 / down_sample_rate)

# 画图设置
use_fastplot = True
use_statekf = True

env = Environment()
str = input("按回车开始")
# 2 3次
#3 3ci
#4 3次
#5 3次
path_to_save = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/8/"
for path_ in [path_to_save]:
    f_list = os.listdir(path_)
    for file_ in f_list:
        os.remove(path_+file_)

if __name__ == '__main__':
    imu_buffer = np.memmap("../Sensor/IM948/imu_thigh.npy", dtype='float32', mode='r', shape=(14,))
    # imu_buffer = np.memmap("/home/yuxuan/Project/fsm_ysc/log/imu_euler_acc.npy", dtype='float32', mode='r',shape=(9 * 1 + 1,))
    pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(num_points, 3))
    #
    num_frame = 1000

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
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            pcd_data_temp = np.copy(pcd_buffer[:])
            imu_data = np.copy(imu_buffer[0:])
            eular_angle = imu_data[7:10]
            env.pcd_to_binary_image(pcd_data_temp, eular_angle)
            env.thin()
            if i == 0:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
                camera_x_buffer.append(0)
                camera_y_buffer.append(0)
                kalman_x_buffer.append(0)
                kalman_y_buffer.append(0)
                pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
                env.classification_from_img()
                pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_os)
                xc, yc, w, h = pcd_os.fea_to_env_paras()
                env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc, yc, w, h])
                time_buffer.append(datetime.datetime.now())
            else:
                pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_opreator_system
                pcd_pre = pcd_pre_os.pcd_new
                pcd_new, pcd_new_os = env.pcd_thin, pcd_opreator_system(env.pcd_thin)
                env.classification_from_img()
                pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn, ax=None)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
                fea_to_align_new, fea_to_align_pre, flag_method = align_fea(pcd_new=pcd_new_os,
                                                                            pcd_pre=pcd_pre_os,
                                                                            _print_=True)

                xmove, ymove = 0, 0
                try:
                    if flag_method == 0:
                        t0 = datetime.datetime.now()
                        xmove, ymove = icp_knn(pcd_s=fea_to_align_pre,
                                               pcd_t=fea_to_align_new)
                        t1 = datetime.datetime.now()
                        print("#################=====FeatureAlign:{}=====#########".format(
                            (t1 - t0).total_seconds() * 1000))
                except Exception as e:
                    print("align method exception:{}".format(e))
                    xmove, ymove = 0, 0
                    flag_method = 1
                xmove_pre = camera_dx_buffer[-1]
                ymove_pre = camera_dy_buffer[-1]
                if flag_method == 1 or abs(xmove) > 0.1 or abs(ymove) > 0.1:
                    xmove = 0
                    ymove = 0
                    print("对齐失败，使用上一次的xmove_pre = {},ymove_pre = {}".format(xmove_pre, ymove_pre))

                print("当前最终xmove = {}, ymove = {}".format(xmove, ymove))
                camera_dx_buffer.append(xmove)
                camera_dy_buffer.append(ymove)
                camera_x_buffer.append(camera_x_buffer[-1] + xmove)
                camera_y_buffer.append(camera_y_buffer[-1] + ymove)

                ##############
                np.save(path_to_save+"{}_pcd.npy".format(i), pcd_data_temp)
                np.save(path_to_save+"{}_imu.npy".format(i), imu_data)
                np.save(path_to_save+"{}_time.npy".format(i), datetime.datetime.now())

                time_buffer.append(datetime.datetime.now())
                dt = (time_buffer[-1] - time_buffer[-2]).total_seconds()
                if use_statekf:
                    state_kf.prediction(dt)
                    # state_vec: knee_px knee_py knee_vel knee_q ankle_q
                    state_vec = [camera_x_buffer[-1], camera_y_buffer[-1],
                                 (camera_x_buffer[-1]-camera_x_buffer[-2])/dt,
                                 (camera_y_buffer[-1]-camera_y_buffer[-2])/dt,
                                0, 0]
                    state_kf.update(state_vec)

                    kalman_x_buffer.append(state_kf.knee_pos[0])
                    kalman_y_buffer.append(state_kf.knee_pos[1])

                    model_prediction = state_kf.model_prediction(num=5, dt=0.02)
                    prediction_x = model_prediction[:, 0]
                    prediction_y = model_prediction[:, 1]

                xc_new, yc_new, w_new, h_new = pcd_new_os.fea_to_env_paras()
                if abs(xc_new + yc_new + w_new + h_new) > 0.04:
                    env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])
                else:
                    xc_pre, yc_pre, w_pre, h_pre = env_paras_buffer[-1][0], env_paras_buffer[-1][1], \
                        env_paras_buffer[-1][2], env_paras_buffer[-1][3]
                    if abs(xc_new + yc_new) <= 0.01:
                        xc_new, yc_new, w_new, h_new = xc_pre, yc_pre, w_pre, h_pre
                        env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])
                    else:
                        if w_new < 0.01:
                            w_new = w_pre
                        if h_new < 0.01:
                            h_new = h_pre
                        env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])

                if len(camera_x_buffer) > 20:
                    if use_statekf:
                        pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                                             p_pcd=[kalman_x_buffer[-1], kalman_y_buffer[-1]], downsample=8)
                        pcd_pre_os.show_fast(fast_plot_ax, "pre", id=int(i) - 1, p_text=[0.2, 0],
                                             p_pcd=[kalman_x_buffer[-2], kalman_y_buffer[-2]], downsample=8)
                        fast_plot_ax.set_camera_traj(np.array(kalman_x_buffer[-20:]), np.array(kalman_y_buffer[-20:]),
                                                     prediction_x=prediction_x, prediction_y=prediction_y)
                        if abs(xc_new + yc_new + w_new + h_new) > 0.04:
                            fast_plot_ax.set_env_paras(xc_new, yc_new, w_new, h_new,
                                                       p=[kalman_x_buffer[-1], kalman_y_buffer[-1]+0.05])
                        fast_plot_ax.update_canvas()
                    else:
                        # pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                        #                      p_pcd=[camera_x_buffer[-1], camera_y_buffer[-1]], downsample=8)
                        # pcd_pre_os.show_fast(fast_plot_ax, "pre", id=int(i) - 1, p_text=[0.2, 0],
                        #                      p_pcd=[camera_x_buffer[-2], camera_y_buffer[-2]], downsample=8)
                        # if abs(xc_new + yc_new + w_new + h_new) > 0.04:
                        #     fast_plot_ax.set_env_paras(xc_new, yc_new, w_new, h_new,
                        #                                p=[camera_x_buffer[-1], camera_y_buffer[-1] + 0.05])
                        # fast_plot_ax.update_canvas()
                        pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                                             p_pcd=[camera_x_buffer[-1], camera_y_buffer[-1]], downsample=8)
                        pcd_new_os.show_fast(fast_plot_ax, 'pre', id=int(i), p_text=[0.2, 0.3],
                                             p_pcd=[camera_x_buffer[-2], camera_y_buffer[-2]], downsample=8)
                        fast_plot_ax.set_camera_traj(np.array(camera_x_buffer), np.array(camera_y_buffer),
                                                     prediction_x=None, prediction_y=None)
                img = cv2.cvtColor(env.elegant_img(), cv2.COLORMAP_RAINBOW)
                add_type(img, env_type=Env_Type(env.type_pred_from_nn), id=i)
                cv2.imshow("binary", img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                print("=======////Totol Time:{}\\\====".format(dt))

    except KeyboardInterrupt:
        pass

