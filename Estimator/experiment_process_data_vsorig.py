import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("//")
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *
from Environment.Plot_ import *
from Estimator.fusion_algo import StateKalmanFilter
from Utils.IO import fifo_data_vec
from Utils.pcd_os_fast_plot import *
from Environment.backup.alignment import *


down_sample_rate = 1
if down_sample_rate % 2 == 1:
    num_points = int(38528 / down_sample_rate) + 1
    if num_points > 38528:
        num_points = 38527
else:
    num_points = int(38528 / down_sample_rate)

env = Environment()

experiment_idx = 0
file_path = "/media/yuxuan/My Passport/VIO_Experiment/vsOrig/{}/".format(experiment_idx)

num_frame_vio = int(len(os.listdir(file_path)) / 3)

camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
time_buffer = []
pcd_os_buffer = [[], []]

if __name__ == "__main__":
    fast_plot_ax = FastPlotCanvas()
    align_fail_time = 0
    start_idx = 0
    try:
        for i in range(start_idx, num_frame_vio):
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            pcd_data = np.load(file_path + "{}_pcd.npy".format(i + 1))
            imu_data = np.load(file_path + "{}_imu.npy".format(i + 1))
            eular_angle = imu_data[7:10]
            env.pcd_to_binary_image(pcd_data, eular_angle)
            env.thin()
            if i == start_idx:
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
                env.classification_from_img()
                pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn, ax=None)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
                fea_to_align_new, fea_to_align_pre, flag_fea_extrafail = align_fea(pcd_new=pcd_new_os,
                                                                                   pcd_pre=pcd_pre_os,
                                                                                   _print_=True)
                xmove, ymove = 0, 0
                flag_fea_alignfail = 1
                try:
                    if flag_fea_extrafail == 0:
                        t0 = datetime.datetime.now()
                        xmove, ymove = icp_knn(pcd_s=fea_to_align_pre,
                                               pcd_t=fea_to_align_new)
                        t1 = datetime.datetime.now()
                        print("#################=====FeatureAlign:{}=====#########".format(
                            (t1 - t0).total_seconds() * 1000))
                        flag_fea_alignfail = 0
                except Exception as e:
                    print("align method exception:{}".format(e))

                use_pre_move = True
                if flag_fea_alignfail == 1 or abs(xmove) > 0.15 or abs(ymove) > 0.15:  # 0.1 0.22
                    align_fail_time += 1
                    if use_pre_move:
                        if align_fail_time > 5:
                            xmove = 0
                            ymove = 0
                        else:
                            xmove_pre = camera_dx_buffer[-1]
                            ymove_pre = camera_dy_buffer[-1]
                            xmove = xmove_pre  # 0
                            ymove = ymove_pre  # 0
                            print("对齐失败，使用上一次的xmove_pre = {},ymove_pre = {}".format(xmove_pre, ymove_pre))
                    else:
                        xmove, ymove = 0, 0
                        print("对齐失败，xmove, ymove = 0")
                else:
                    align_fail_time = 0
                print("当前最终xmove = {}, ymove = {}".format(xmove, ymove))
                camera_dx_buffer.append(xmove)
                camera_dy_buffer.append(ymove)
                camera_x_buffer.append(camera_x_buffer[-1] + xmove)
                camera_y_buffer.append(camera_y_buffer[-1] + ymove)
                if len(camera_x_buffer)>5:
                    pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                                         p_pcd=[camera_x_buffer[-1], camera_y_buffer[-1]], downsample=8)
                    pcd_new_os.show_fast(fast_plot_ax, 'pre', id=int(i), p_text=[0.2, 0.3],
                                         p_pcd=[camera_x_buffer[-2], camera_y_buffer[-2]], downsample=8)
                    fast_plot_ax.set_camera_traj(np.array(camera_x_buffer), np.array(camera_y_buffer),
                                                 prediction_x=None, prediction_y=None)
                    fast_plot_ax.update_canvas()
                # time.sleep(0.2)
    except KeyboardInterrupt:
        pass
fig = plt.figure()
plt.plot(np.array(camera_x_buffer)-camera_x_buffer[0], np.array(camera_y_buffer)-camera_y_buffer[0])

plt.show()