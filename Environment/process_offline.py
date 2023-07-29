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

imu_buffer_path = "../Sensor/IM948/imu_buffer.npy"
data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST3/"  # 3
open3d_pcd_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4_OPEN3D/"
env = Environment()
env_type_buffer = []
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
alignment_flag_buffer = []
img_list = os.listdir(open3d_pcd_save_path)
for f in img_list:
    os.remove(open3d_pcd_save_path + f)

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


if __name__ == "__main__":
    imu_data = np.load(data_save_path + "imu_data.npy")
    num_frame = np.size(imu_data, 0) - 1
    imu_data = imu_data[1:, :]
    idx_frame = np.arange(num_frame)
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    plt.ion()
    for i in idx_frame:
        print("Frame[{}]".format(i))
        env.img_binary = np.load(data_save_path + "{}_img.npy".format(i))
        env.pcd_2d = np.load(data_save_path + "{}_pcd.npy".format(i))
        env.classification_from_img()
        env_type_buffer.append(env.type_pred_from_nn)
        env.thin()
        plt.cla()
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 3)
        ax.set_zlim(-1, 2)
        ax.view_init(elev=10, azim=-15)
        np.save(open3d_pcd_save_path + "{}_pcd2d.npy".format(i), env.pcd_thin)
        # ax.plot3D(pcd_3d[0:-1:21, 0], pcd_3d[0:-1:21, 1], pcd_3d[0:-1:21, 2], '.:c')
        if i == 0:
            env.pcd_prev = env.pcd_2d
            camera_dx_buffer.append(0)
            camera_dy_buffer.append(0)
            camera_x_buffer.append(0)
            camera_y_buffer.append(0)
            alignment_flag_buffer.append(0)
        else:
            xc = yc = w = h = 0
            if env.type_pred_from_nn == Env_Type.Upstair.value:
                xc, yc, w, h = env.get_fea_sa()
                regis = icp_alignment(env.pcd_prev, env.pcd_thin, alignment_flag_buffer[-1])
                try:
                    xmove, ymove, flag = regis.alignment()
                except:
                    print("RANSAC valueerror! Use previous estimation!")

                if abs(xmove) > 0.05 or abs(ymove) > 0.05:
                    print("移动距离过大")
                    xmove_prev = camera_dx_buffer[-1]
                    ymove_prev = camera_dy_buffer[-1]
                    camera_dx_buffer.append(xmove_prev)
                    camera_dy_buffer.append(ymove_prev)
                else:
                    print("对齐成功")
                    camera_dx_buffer.append(xmove)
                    camera_dy_buffer.append(ymove)
            else:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
            camera_x_buffer.append(camera_x_buffer[-1] + camera_dx_buffer[-1])
            camera_y_buffer.append(camera_y_buffer[-1] + camera_dy_buffer[-1])
            env.pcd_prev = env.pcd_thin
            pcd_3d = env.pcd_from_body_to_world(imu_data[i, :])
            ax.plot3D(np.zeros((len(camera_x_buffer, ))),
                      np.array(camera_x_buffer),
                      np.array(camera_y_buffer),
                      color='m')
            ax.plot3D(pcd_3d[0:-1:21, 0],
                      pcd_3d[0:-1:21, 1] + camera_x_buffer[-1],
                      pcd_3d[0:-1:21, 2] + camera_y_buffer[-1],
                      '.:c')
            if xc * yc * w * h != 0:
                yy = np.repeat(xc, 5)
                zz = np.repeat(yc, 5)
                xx = np.linspace(-0.2, 0.2, 5)
                ax.plot3D(xx, yy + camera_x_buffer[-1], zz + camera_y_buffer[-1], color='purple', linewidth=2)
                add_collision(xc, yc, w, h, ax, p=[camera_x_buffer[-1], camera_y_buffer[-1]])
        plt.draw()
        plt.pause(0.05)
        img = env.elegant_img()
        add_type(img, Env_Type(env.type_pred_from_nn))
        cv2.imshow("binary", img)
        key = cv2.waitKey(1)
    print("here")
    np.save(open3d_pcd_save_path+"traj_x.npy",np.array(camera_x_buffer))
    np.save(open3d_pcd_save_path + "traj_y.npy", np.array(camera_y_buffer))
    np.save(open3d_pcd_save_path+"env_type_buffer.npy", np.array(env_type_buffer))
    plt.plot(idx_frame, np.array(env_type_buffer))

    plt.show()

