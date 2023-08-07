import os

import cv2
import open3d as o3d
import numpy as np
from lcm_msg.pcd_lcm.pcd_xyz import *
from lcm_msg.lcm_init import *
import lcm
import time
from Environment import *
import matplotlib.pyplot as plt
from alignment import *
from my_feature import *

imu_buffer_path = "../Sensor/IM948/imu_buffer.npy"
data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4/"  # 3

env = Environment()
env_type_buffer = []
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
corner_buffer = []
env_type_buffer = []
alignment_flag_buffer = [[], []]
fea_A_buffer = [np.zeros((0, 2)), np.zeros((0, 2))]
fea_B_buffer = [np.zeros((0, 2)), np.zeros((0, 2))]
fea_C_buffer = [np.zeros((0, 2)), np.zeros((0, 2))]

xc_new, yc_new, w_new, h_new = 0, 0, 0, 0
xc_pre, yc_pre, w_pre, h_pre = 0, 0, 0, 0,

num_frame_to_get_paras = 1
pcd_multi_frame_buffer = []
for i in range(num_frame_to_get_paras):
    pcd_multi_frame_buffer.append(np.zeros((0, 2)))

use_method1 = True
if use_method1:
    traj_x = np.load("traj_x_method2.npy")
    traj_y = np.load("traj_y_method2.npy")
else:
    traj_x = np.load("traj_x_method1.npy")
    traj_y = np.load("traj_y_method1.npy")


# open3d_pcd_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4_OPEN3D/"
# img_list = os.listdir(open3d_pcd_save_path)
# for f in img_list:
#     os.remove(open3d_pcd_save_path + f)

def add_type(img, env_type):
    if env_type == Env_Type.Levelground:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    elif env_type == Env_Type.Upstair:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2)
    else:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)


def add_collision(xc, yc, w, h, ax, p=None):
    if p is None:
        p = [0, 0]
    ax.plot3D([-0.3, 0.3], [xc - w + p[0], xc - w + p[0]], [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1]], color='r',
              linewidth=4)
    ax.plot3D([-0.3, 0.3], [xc + p[0], xc + p[0]], [yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]], color='r', linewidth=4)
    ax.plot3D([-0.3, 0.3], [xc + p[0], xc + p[0]], [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1]], color='r', linewidth=4)
    ax.plot3D([-0.3, 0.3], [xc + w + p[0], xc + w + p[0]], [yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]], color='r',
              linewidth=4)
    ax.plot3D([0.3, 0.3, 0.3, 0.3], [xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]],
              [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1], yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]],
              color='r',
              linewidth=4)
    ax.plot3D([-0.3, -0.3, -0.3, -0.3], [xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]],
              [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1], yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]],
              color='r',
              linewidth=4)


def fifo(buffer, data):
    buffer[0:-1] = buffer[1:]
    buffer[-1] = data
    return buffer


if __name__ == "__main__":
    imu_data = np.load(data_save_path + "imu_data.npy")
    imu_data = imu_data[1:, :]
    idx_frame = np.arange(np.shape(imu_data)[0])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.ion()

    for i in idx_frame:
        print("------------Frame[{}]-----------------".format(i))
        print("load binary image and pcd to process")
        env.img_binary = np.load(data_save_path + "{}_img.npy".format(i))
        env.pcd_2d = np.load(data_save_path + "{}_pcd.npy".format(i))
        # 利用image binary进行地形分类
        env.classification_from_img()
        # todo:假设全部为上楼梯
        env.type_pred_from_nn = 1
        env_type_buffer.append(env.type_pred_from_nn)
        # 预处理2d点云
        env.thin()
        plt.cla()
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 3)
        ax.set_zlim(-1, 2)
        ax.view_init(elev=20, azim=-15)
        # np.save(open3d_pcd_save_path + "{}_pcd2d.npy".format(i), env.pcd_thin)
        # ax.plot3D(pcd_3d[0:-1:21, 0], pcd_3d[0:-1:21, 1], pcd_3d[0:-1:21, 2], '.:c')
        pcd_new = env.pcd_thin
        if i == 0:
            pcd_pre = pcd_new
            camera_dx_buffer.append(0)
            camera_dy_buffer.append(0)
            camera_x_buffer.append(0)
            camera_y_buffer.append(0)
            alignment_flag_buffer.append(0)
            if env.type_pred_from_nn == Env_Type.Upstair.value:
                # 特征提取
                fea_A_new, fea_B_new, fea_C_new, corner_new = get_fea_sa(pcd_new)
                fea_A_buffer = fifo(fea_A_buffer, fea_A_new)
                fea_B_buffer = fifo(fea_B_buffer, fea_B_new)
                fea_C_buffer = fifo(fea_C_buffer, fea_C_new)
                fea_A_get_pre = False
                if np.shape(fea_A_new)[0] > 10:
                    fea_A_get_pre = True
                fea_B_get_pre = False
                if np.shape(fea_B_new)[0] > 10:
                    fea_B_get_pre = True
                fea_C_get_pre = False
                if np.shape(fea_C_new)[0] > 10:
                    fea_C_get_pre = True
                corner_buffer.append(corner_new)

        else:
            env_type_new = env.type_pred_from_nn
            env_type_pre = env_type_buffer[-1]
            if env_type_new == env_type_pre == Env_Type.Upstair.value:
                # 取出上一轮流特征
                fea_A_pre = fea_A_buffer[-1]
                fea_B_pre = fea_B_buffer[-1]
                fea_C_pre = fea_C_buffer[-1]
                corner_pre = corner_buffer[-1]
                fea_A_get_pre = False
                if np.shape(fea_A_pre)[0] > 10:
                    fea_A_get_pre = True
                fea_B_get_pre = False
                if np.shape(fea_B_pre)[0] > 10:
                    fea_B_get_pre = True
                fea_C_get_pre = False
                if np.shape(fea_C_pre)[0] > 10:
                    fea_C_get_pre = True

                # 特征提取
                fea_A_new, fea_B_new, fea_C_new, corner_new = get_fea_sa(pcd_new)
                fea_A_get_new = False
                if np.shape(fea_A_new)[0] > 10:
                    fea_A_get_new = True
                fea_B_get_new = False
                if np.shape(fea_B_new)[0] > 10:
                    fea_B_get_new = True
                fea_C_get_new = False
                if np.shape(fea_C_new)[0] > 10:
                    fea_C_get_new = True

                # 用于对齐的特征
                pcd_to_align_new, pcd_to_align_pre, pcd_component_new, pcd_component_pre = align_rule(
                    fea_A=[fea_A_new, fea_A_pre],
                    fea_A_get=[fea_A_get_new, fea_A_get_pre],
                    fea_B=[fea_B_new, fea_B_pre],
                    fea_B_get=[fea_B_get_new, fea_B_get_pre],
                    fea_C=[fea_C_new, fea_C_pre],
                    fea_C_get=[fea_C_get_new, fea_C_get_pre],
                    corner_situation=[corner_new, corner_pre]
                )
                # 如果没有特征则随机选取
                flag_method1 = 0
                if np.shape(pcd_to_align_pre)[0] == np.shape(pcd_to_align_new)[0] == 0:
                    # idx = np.arange(0, min(np.shape(pcd_new)[0], np.shape(pcd_pre)[0]))
                    # np.random.shuffle(idx)
                    # idx_chosen = np.random.choice(idx, size=600)
                    # idx_chosen = np.sort(idx_chosen)
                    # pcd_to_align_new = pcd_new[idx_chosen, :]
                    # # idx_chosen = np.random.choice(idx, size=600)
                    # # idx_chosen = np.sort(idx_chosen)
                    # pcd_to_align_pre = pcd_pre[idx_chosen, :]
                    # pcd_component_new = [200, 200, 200]
                    # pcd_component_pre = [200, 200, 200]
                    flag_method1 = 1
                else:
                    flag_method1 = 0

                # 使用icp对齐
                # Method 1
                if env_type_new == Env_Type.Upstair.value and flag_method1 == 0:
                    xmove_method1, ymove_method1 = icp(pcd_s=pcd_to_align_pre, pcd_t=pcd_to_align_new,
                                                       pcd_s_component=pcd_component_pre,
                                                       pcd_t_component=pcd_component_new)
                    print("xmove_1 = {},ymove_1 = {}, flag = {}".format(xmove_method1, ymove_method1, flag_method1))
                else:
                    xmove_method1, ymove_method1 = 0, 0

                if abs(xmove_method1) > 0.05 or abs(ymove_method1) > 0.05:
                    # flag_method1 = 1
                    print("method1 移动距离过大")

                flag_method2 = 0
                # Method 2
                if env_type_new == Env_Type.Upstair.value:
                    regis = icp_alignment(pcd_pre, pcd_new, alignment_flag_buffer[-1])
                    try:
                        xmove_method2, ymove_method2, flag_method2 = regis.alignment()
                        print("xmove_2 = {},ymove_2 = {}, flag = {}".format(xmove, ymove, flag_method2))
                    except:
                        print("RANSAC valueerror! Use previous estimation!")
                else:
                    xmove_method2, ymove_method2 = 0, 0
                    flag_method2 = 1

                if abs(xmove_method2) > 0.05 or abs(ymove_method2) > 0.05:
                    # flag_method2 = 1
                    print("method2 移动距离过大")

                if use_method1:
                    xmove = xmove_method1
                    ymove = ymove_method1
                    flag = flag_method1
                else:
                    xmove = xmove_method2
                    ymove = ymove_method2
                    flag = flag_method2

                alignment_flag_buffer[0].append(flag_method1)
                alignment_flag_buffer[1].append(flag_method2)

                xmove_pre = camera_dx_buffer[-1]
                ymove_pre = camera_dy_buffer[-1]
                if flag == 1 or abs(xmove) > 0.05 or abs(ymove) > 0.05:
                    xmove = xmove_pre
                    ymove = ymove_pre
                    print("对齐失败，使用上一次的xmove_pre = {},ymove_pre = {}".format(xmove_pre, ymove_pre))

                print("当前最终xmove = {}, ymove = {}".format(xmove, ymove))
                print("参考最终xmove = {}, ymove = {}".format(traj_x[i] - traj_x[i - 1], traj_y[i] - traj_y[i - 1]))
                camera_dx_buffer.append(xmove)
                camera_dy_buffer.append(ymove)
                camera_x_buffer.append(camera_x_buffer[-1] + xmove)
                camera_y_buffer.append(camera_y_buffer[-1] + ymove)

                if i > num_frame_to_get_paras:
                    xc_new, yc_new, w_new, h_new = 0, 0, 0, 0
                    pcd_total = np.zeros((0, 2))
                    pcd_multi_frame_buffer = fifo(pcd_multi_frame_buffer, pcd_new)
                    idx_total = np.zeros((0,))
                    for j in range(num_frame_to_get_paras):
                        x_j = camera_x_buffer[-num_frame_to_get_paras + j]
                        y_j = camera_y_buffer[-num_frame_to_get_paras + j]
                        pcd_temp = np.copy(pcd_multi_frame_buffer[j])
                        pcd_temp[:, 0] += x_j
                        pcd_temp[:, 1] += y_j
                        pcd_total = np.vstack([pcd_total, pcd_temp])
                    # pcd_total = pcd_total[np.argsort(pcd_total[:, 1]), :]
                    pcd3d_multi = pcd2d_to_3d(pcd_total[0:-1:num_frame_to_get_paras, :], num_rows=5)
                    xx = pcd3d_multi[:, 0]
                    yy = pcd3d_multi[:, 1]
                    zz = pcd3d_multi[:, 2]
                    ax.scatter3D(xx[0:-1:51],
                                 yy[0:-1:51],
                                 zz[0:-1:51],
                                 linewidths=0.05,
                                 color='c',
                                 alpha=0.4)
                    # fea_point_A, fea_point_B, fea_point_C, corner_multi = get_paras_sa(pcd_total[0:-1:num_frame_to_get_paras])
                    # xc, yc, w, h = cal_paras_from_fea_sa(fea_point_A,fea_point_B,fea_point_C,corner_multi)
                    xc_new, yc_new, w_new, h_new = get_fea_sa_original(pcd_total[0:-1:num_frame_to_get_paras])
                    if xc_new*yc_new*w_new*h_new == 0:
                        xc_new = xc_pre
                        yc_new = yc_pre
                        w_new = w_pre
                        h_new = h_pre
                    if xc_new*yc_new*w_new*h_new != 0:
                        yy = np.repeat(xc_new, 5)
                        zz = np.repeat(yc_new, 5)
                        xx = np.linspace(-0.2, 0.2, 5)
                        add_collision(xc_new, yc_new, w_new, h_new, ax)
                    xc_pre = xc_new
                    yc_pre = yc_new
                    w_pre = w_new
                    h_pre = h_new

                # plot
                pcd3d_new = pcd2d_to_3d(pcd_new, num_rows=2)
                pcd3d_pre = pcd2d_to_3d(pcd_pre)

                # xx = pcd3d_pre[:, 0]
                # yy = pcd3d_pre[:, 1] + camera_x_buffer[-2]
                # zz = pcd3d_pre[:, 2] + camera_y_buffer[-2]
                # ax.plot3D(xx[0:-1:51],
                #           yy[0:-1:51],
                #           zz[0:-1:51],
                #           linewidth=5,
                #           color=plt.cm.jet(50),
                #           alpha=0.5)

                xx = pcd3d_new[:, 0]
                yy = pcd3d_new[:, 1] + camera_x_buffer[-1]
                zz = pcd3d_new[:, 2] + camera_y_buffer[-1]
                ax.plot3D(xx[0:-1:51],
                          yy[0:-1:51],
                          zz[0:-1:51],
                          linewidth=1,
                          color='blue')

                ax.plot3D(np.zeros((len(camera_x_buffer, ))),
                          np.array(camera_x_buffer),
                          np.array(camera_y_buffer),
                          color='b')

                # xx = pcd3d_pre[:, 0]
                # yy = pcd3d_pre[:, 1] + traj_x[i - 1]
                # zz = pcd3d_pre[:, 2] + traj_y[i - 1] - 0.2
                # ax.plot3D(xx[0:-1:51],
                #           yy[0:-1:51],
                #           zz[0:-1:51],
                #           linewidth=5,
                #           color=plt.cm.jet(220),
                #           alpha=0.5)
                #
                # xx = pcd3d_new[:, 0]
                # yy = pcd3d_new[:, 1] + traj_x[i]
                # zz = pcd3d_new[:, 2] + traj_y[i] - 0.2
                # ax.plot3D(xx[0:-1:51],
                #           yy[0:-1:51],
                #           zz[0:-1:51],
                #           linewidth=1,
                #           color='red')

                # ax.plot3D(np.zeros((len(camera_x_buffer, ) - 1)),
                #           traj_x[0:i],
                #           traj_y[0:i],
                #           color='red')

                fea_A_buffer = fifo(fea_A_buffer, fea_A_new)
                fea_B_buffer = fifo(fea_B_buffer, fea_B_new)
                fea_C_buffer = fifo(fea_C_buffer, fea_C_new)
                corner_buffer.append(corner_new)
                pcd_pre = pcd_new

                # ax.text(0.5, 1.8, 1.8, "corner_situation:{},id:{}".format(corner_new, i))
                # ax.text(0.5, 1.8, 1.2, "corner_pre:{},id:{}".format(corner_buffer[-2], i - 1))

                # if use_method1:
                #     ax.text(0.5, 0.2, 1.2, "method: new feature extraction", color='b')
                #     ax.text(0.5, 0.2, 1.0, "method: previous feature extraction", color='r')
                # else:
                #     ax.text(0.5, 0.2, 1.2, "method: previous feature extraction", color='b')
                #     ax.text(0.5, 0.2, 1.0, "method: new feature extraction", color='r')

                # if flag_method1 == 1:
                #     ax.text(0.5, 1.3, -0.5, "method new fails times:{}".format(sum(alignment_flag_buffer[0])), color='y')
                # else:
                #     ax.text(0.5, 1.3, -0.5, "method new fails times:{}".format(sum(alignment_flag_buffer[0])), color='g')
                #
                # if flag_method2 == 1:
                #     ax.text(0.5, 1.3, -0.8, "method pre fails times:{}".format(sum(alignment_flag_buffer[1])), color='y')
                # else:
                #     ax.text(0.5, 1.3, -0.8, "method pre fails times:{}".format(sum(alignment_flag_buffer[1])), color='g')

        plt.draw()
        plt.pause(0.01)
        if i == 1:
            plt.pause(5)

    if use_method1:
        np.save("traj_x_method1.npy", np.array(camera_x_buffer))
        np.save("traj_y_method1.npy", np.array(camera_y_buffer))
    else:
        np.save("traj_x_method2.npy", np.array(camera_x_buffer))
        np.save("traj_y_method2.npy", np.array(camera_y_buffer))