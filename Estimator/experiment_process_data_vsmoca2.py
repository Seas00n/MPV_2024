import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.io as scio

sys.path.append("//")
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *
from Environment.Plot_ import *
from Estimator.fusion_algo import StateKalmanFilter
from Utils.IO import fifo_data_vec
from Utils.pcd_os_fast_plot import *
from Environment.backup.alignment import *
from Environment.alignment_open3d import *

use_statekf = False
env = Environment()

# 1 4 8 9 10 11 12 14
# 1 4 8 9
experiment_idx = 9
moca_align_file_path = "/home/yuxuan/Project/StairFeatureExtraction/MocaDataProcess/align_data/"
moca_data = np.load(moca_align_file_path + "Moca_align{}.npy".format(experiment_idx), allow_pickle=True)
idx_align = np.load(moca_align_file_path + "idx_align{}.npy".format(experiment_idx), allow_pickle=True)
idx_stair = np.arange(2, 2 + 3 * 8)
idx_cam = np.arange(26, 29)
idx_knee = np.arange(29, 32)
idx_ankle = np.arange(32, 35)
idx_heel = np.arange(35, 38)
idx_toe = np.arange(38, 41)

# idx_align = [0, 0]
file_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(experiment_idx)
save_path = "/media/yuxuan/My Passport/VIO_Experiment/Result2/"
cam_origin = moca_data[0, idx_cam]
stair = np.mean(moca_data[:, idx_stair], axis=0)
stairs_x = stair[0::3]
stairs_z = stair[2::3]
leg_pose = moca_data[:, idx_knee[0]:]
scio.savemat(save_path + "appendix_{}.mat".format(experiment_idx),
             {"cam_origin": cam_origin,
              "stair_x": stairs_x,
              "stair_z": stairs_z,
              "leg_pose": leg_pose}
             )

camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
camera_x_cch_buffer = []
camera_y_cch_buffer = []
camera_dx_cch_buffer = []
camera_dy_cch_buffer = []
camera_x_o3d_buffer = []
camera_y_o3d_buffer = []
camera_dx_o3d_buffer = []
camera_dy_o3d_buffer = []

kalman_x_buffer = []
kalman_y_buffer = []
time_buffer = []
pcd_os_buffer = [[], []]
alignment_flag_buffer = []
alignment_flag_cch_buffer = []
alignment_flag_o3d_buffer = []

time_cost_buffer = []
use_fast_plot = True

if __name__ == "__main__":
    if use_fast_plot:
        # fast_plot_ax = FastPlotCanvas()
        print()
    else:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        plt.ion()
        plt.show(block=False)
    align_fail_time = 0
    align_fail_time_cch = 0
    alignment_fail_time_o3d = 0
    start_idx = idx_align[1]
    end_idx = idx_align[3]
    if use_statekf:
        state_kf = StateKalmanFilter()

    try:
        for i in range(start_idx, end_idx):
            # if not use_fast_plot:
            #     ax1.cla()
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            pcd_data = np.load(file_path + "{}_pcd.npy".format(i), allow_pickle=True)
            pcd_data = pcd_data[0:-1, :]
            imu_data = np.load(file_path + "{}_imu.npy".format(i), allow_pickle=True)
            time_data = np.load(file_path + "{}_time.npy".format(i), allow_pickle=True)
            eular_angle = imu_data[7:10]
            env.pcd_to_binary_image(pcd_data, eular_angle)
            env.thin()
            if i == start_idx:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
                camera_x_buffer.append(0)
                camera_y_buffer.append(0)
                camera_dx_cch_buffer.append(0)
                camera_dy_cch_buffer.append(0)
                camera_x_cch_buffer.append(0)
                camera_y_cch_buffer.append(0)
                camera_dx_o3d_buffer.append(0)
                camera_dy_o3d_buffer.append(0)
                camera_x_o3d_buffer.append(0)
                camera_y_o3d_buffer.append(0)
                kalman_x_buffer.append(0)
                kalman_y_buffer.append(0)
                pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
                env.classification_from_img()
                pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_os)
                time_buffer.append(time_data)
                alignment_flag_buffer.append(0)
                alignment_flag_cch_buffer.append(0)
                alignment_flag_o3d_buffer.append(0)
            else:
                pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_opreator_system
                pcd_pre = pcd_pre_os.pcd_new
                pcd_new, pcd_new_os = env.pcd_thin, pcd_opreator_system(env.pcd_thin)
                env.classification_from_img()
                if not use_fast_plot:
                    pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn, ax=ax1)
                else:
                    pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn, ax=None)
                    # time_cost_buffer.append(pcd_new_os.cost_time)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)

                ######################################################################################
                time_buffer.append(time_data)
                dt = (time_buffer[-1] - time_buffer[-2]).total_seconds()
                use_pre_move = True
                #######################################################################################
                fea_to_align_new, fea_to_align_pre, flag_fea_extrafail = align_fea(pcd_new=pcd_new_os,
                                                                                   pcd_pre=pcd_pre_os,
                                                                                   _print_=True)
                if pcd_new_os.is_fea_B_gotten and pcd_new_os.is_fea_C_gotten:
                    print("Stair_Height:", pcd_new_os.Ccenter[1]-pcd_new_os.Bcenter[1])
                xmove, ymove = 0, 0
                flag_fea_alignfail = 1
                try:
                    if flag_fea_extrafail == 0:
                        xmove, ymove = icp_knn(pcd_s=fea_to_align_pre,
                                               pcd_t=fea_to_align_new,
                                               max_iterate=20)
                        flag_fea_alignfail = 0
                        time_cost_buffer.append(ct)
                except Exception as e:
                    print("align method exception:{}".format(e))

                if flag_fea_alignfail == 1 or abs(xmove) > 0.08 or abs(ymove) > 0.08 or dt > 0.2:  # 0.1 0.22
                    print("xmove = {}, ymove = {}".format(xmove, ymove))
                    align_fail_time += 1
                    flag_fea_alignfail = 1
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
                if xmove > 0:
                    xmove *= 1
                else:
                    xmove *= 1.2
                if ymove > 0:
                    ymove *= 1.02
                else:
                    ymove *= 1
                camera_x_buffer.append(camera_x_buffer[-1] + xmove)
                camera_y_buffer.append(camera_y_buffer[-1] + ymove)
                alignment_flag_buffer.append(flag_fea_alignfail)
                ##########################################################################################
                regis = icp_alignment(pcd_pre, pcd_new, flag=None)
                xmove_cch, ymove_cch = 0, 0
                flag_fea_alignfail_cch = 1
                try:
                    xmove_cch, ymove_cch, flag_fea_alignfail_cch = regis.alignment()
                except:
                    print("RANSAC valueerror! Use previous estimation!")

                if flag_fea_alignfail_cch == 1 or abs(xmove_cch) > 0.08 or abs(
                        ymove_cch) > 0.08 or dt > 0.2:  # 0.1 0.22
                    print("xmove_cch = {}, ymove_cch = {}".format(xmove_cch, ymove_cch))
                    align_fail_time_cch += 1
                    flag_fea_alignfail_cch = 1
                    if use_pre_move:
                        if align_fail_time_cch > 5:
                            xmove_cch = 0
                            ymove_cch = 0
                        else:
                            xmove_pre_cch = camera_dx_cch_buffer[-1]
                            ymove_pre_cch = camera_dy_cch_buffer[-1]
                            xmove_cch = xmove_pre_cch  # 0
                            ymove_cch = ymove_pre_cch  # 0
                            print("对齐失败，使用上一次的xmove_pre_cch = {},ymove_pre_cch = {}".format(xmove_pre_cch,
                                                                                                      ymove_pre_cch))
                    else:
                        xmove_cch, ymove_cch = 0, 0
                        print("对齐失败，xmove, ymove = 0")
                else:
                    align_fail_time_cch = 0
                print("当前最终xmovecch = {}, ymovecch = {}".format(xmove_cch, ymove_cch))
                camera_dx_cch_buffer.append(xmove_cch)
                camera_dy_cch_buffer.append(ymove_cch)
                camera_x_cch_buffer.append(camera_x_cch_buffer[-1] + xmove_cch)
                camera_y_cch_buffer.append(camera_y_cch_buffer[-1] + ymove_cch)
                alignment_flag_cch_buffer.append(flag_fea_alignfail_cch)
                ######################################################################################################
                regis = open3d_alignment(pcd_s=pcd_pre_os.pcd_new,
                                         pcd_t=pcd_new_os.pcd_new)
                trans = regis.alignment_new()
                xmove_o3d = -trans[1, 3]*1.4
                ymove_o3d = -trans[2, 3]*1.2
                flag_fea_alignfail_o3d = 0
                if abs(xmove_o3d) > 0.08 or abs(ymove_o3d) > 0.08 or dt > 0.2:  # 0.1 0.22
                    print("xmove_o3d = {}, ymove_o3d = {}".format(xmove_o3d, ymove_o3d))
                    alignment_fail_time_o3d += 1
                    flag_fea_alignfail_o3d = 1
                    if use_pre_move:
                        if alignment_fail_time_o3d > 5:
                            xmove_o3d = 0
                            ymove_o3d = 0
                        else:
                            xmove_pre_o3d = camera_dx_o3d_buffer[-1]
                            ymove_pre_o3d = camera_dy_o3d_buffer[-1]
                            xmove_o3d = xmove_pre_o3d  # 0
                            ymove_o3d = ymove_pre_o3d  # 0
                            print("对齐失败，使用上一次的xmove_pre_o3d = {},ymove_pre_o3d = {}".format(xmove_pre_o3d, ymove_pre_o3d))
                    else:
                        xmove_o3d, ymove_o3d = 0, 0
                        print("对齐失败，xmove, ymove = 0")
                else:
                    align_fail_time_o3d = 0
                print("当前最终xmoveo3d = {}, ymoveo3d = {}".format(xmove_o3d, ymove_o3d))
                camera_dx_o3d_buffer.append(xmove_o3d)
                camera_dy_o3d_buffer.append(ymove_o3d)
                camera_x_o3d_buffer.append(camera_x_o3d_buffer[-1] + xmove_o3d)
                camera_y_o3d_buffer.append(camera_y_o3d_buffer[-1] + ymove_o3d)
                alignment_flag_o3d_buffer.append(flag_fea_alignfail_o3d)
                ###################################################################################################
                if use_statekf:
                    state_kf.prediction(dt)
                    state_vec = [camera_x_buffer[-1], camera_y_buffer[-1],
                                 (camera_x_buffer[-1] - camera_x_buffer[-2]) / dt,
                                 (camera_y_buffer[-1] - camera_y_buffer[-2]) / dt,
                                 0, 0]
                    state_kf.update(state_vec)
                    kalman_x_buffer.append(state_kf.knee_pos[0])
                    kalman_y_buffer.append(state_kf.knee_pos[1])

                # if len(camera_x_buffer) > 2:
                #     if use_fast_plot:
                #         if use_statekf:
                #             pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                #                                  p_pcd=[kalman_x_buffer[-1], kalman_y_buffer[-1]], downsample=8)
                #             pcd_pre_os.show_fast(fast_plot_ax, 'pre', id=int(i), p_text=[0.2, 0.3],
                #                                  p_pcd=[kalman_x_buffer[-2], kalman_y_buffer[-2]], downsample=8)
                #             fast_plot_ax.set_camera_traj(np.array(kalman_x_buffer), np.array(kalman_y_buffer),
                #                                          prediction_x=None, prediction_y=None)
                #         else:
                #             pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                #                                  p_pcd=[camera_x_buffer[-1], camera_y_buffer[-1]], downsample=8)
                #             pcd_pre_os.show_fast(fast_plot_ax, 'pre', id=int(i), p_text=[0.2, 0.3],
                #                                  p_pcd=[camera_x_buffer[-2], camera_y_buffer[-2]], downsample=8)
                #             fast_plot_ax.set_camera_traj(np.array(camera_x_buffer), np.array(camera_y_buffer),
                #                                          prediction_x=None, prediction_y=None)
                #         fast_plot_ax.update_canvas()
                #         # cv2.imshow("PCD_2d",env.elegant_img())
                #         # cv2.waitKey(1)
                #     else:
                #         pcd_new_os.show_(ax1, pcd_color='r', id=i, p_text=0.1,
                #                          p_pcd=None, downsample=8)
                #         pcd_pre_os.show_(ax1, pcd_color='b', id=i, p_text=0.1,
                #                          p_pcd=None, downsample=8)
                #         ax1.set_xlim([-1, 1])
                #         ax1.set_ylim([-1, 1])
                #         plt.pause(0.01)
                # time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print(np.sum(np.array(alignment_flag_buffer)))
    print(np.sum(np.array(alignment_flag_cch_buffer)))
    print(np.sum(np.array(alignment_flag_o3d_buffer)))
    print(np.mean(np.array(time_cost_buffer)))
    fig, ax = plt.subplots()
    vio_x = np.array(camera_x_buffer) - camera_x_buffer[0]
    vio_x = gaussian_filter1d(vio_x, 1)
    vio_y = np.array(camera_y_buffer) - camera_y_buffer[0]
    vio_y = gaussian_filter1d(vio_y, 1)
    ax.plot(vio_x, vio_y)
    vio_x_cch = np.array(camera_x_cch_buffer) - camera_x_cch_buffer[0]
    vio_x_cch = gaussian_filter1d(vio_x_cch, 1)
    vio_y_cch = np.array(camera_y_cch_buffer) - camera_y_cch_buffer[0]
    vio_y_cch = gaussian_filter1d(vio_y_cch, 1)
    ax.plot(vio_x_cch, vio_y_cch)
    cam_moca = moca_data[:, idx_cam] * 0.001
    cam_moca = cam_moca - cam_moca[0, :]
    cam_x = cam_moca[:, 0]
    cam_x = gaussian_filter1d(cam_x, 1)
    cam_z = cam_moca[:, 2]
    cam_z = gaussian_filter1d(cam_z, 1)
    ax.plot(cam_x, cam_z)
    vio_x_o3d = np.array(camera_x_o3d_buffer) - camera_x_o3d_buffer[0]
    vio_x_o3d = gaussian_filter1d(vio_x_o3d, 1)
    vio_y_o3d = np.array(camera_y_o3d_buffer) - camera_y_o3d_buffer[0]
    vio_y_o3d = gaussian_filter1d(vio_y_o3d, 1)
    ax.plot(vio_x_o3d, vio_y_o3d)
    plt.show(block=True)


    # scio.savemat(save_path + "moca_final_{}.mat".format(experiment_idx), {"cam_x": cam_x, "cam_y": cam_z})
    # scio.savemat(save_path + "vio_final_{}.mat".format(experiment_idx), {"vio_x": vio_x, "vio_y": vio_y})
    # scio.savemat(save_path + "vio_cch_final_{}.mat".format(experiment_idx), {"vio_x_cch": vio_x_cch, "vio_y_cch": vio_y_cch})
    # scio.savemat(save_path + "vio_o3d_final_{}.mat".format(experiment_idx), {"vio_x_o3d": vio_x_o3d, "vio_y_o3d": vio_y_o3d})