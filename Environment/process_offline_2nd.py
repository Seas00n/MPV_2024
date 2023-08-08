import os
import cv2
import matplotlib as mpl

from Environment import *
from alignment_knn import *
from feature_extra_new import *
from Utils.IO import fifo_data_vec
from alignment import icp_alignment
imu_buffer_path = "../Sensor/IM948/imu_buffer.npy"
data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST1/"  # 3

env = Environment()
env_type_buffer = []
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
pcd_os_buffer = [[], []]
alignment_flag_buffer = [[], []]
env_paras_buffer = [[], []]

# 特征提取方法设置
use_method1 = True
if use_method1:
    traj_x = np.load("traj_x_method2.npy")
    traj_y = np.load("traj_y_method2.npy")
else:
    traj_x = np.load("traj_x_method1.npy")
    traj_y = np.load("traj_y_method1.npy")

# 重构参数设置
use_multi_frame_rebuild = False
num_frame_rebuild = 3
pcd_multi_build_buffer = []
for i in range(num_frame_rebuild):
    pcd_multi_build_buffer.append(np.zeros((0, 2)))

# 画图设置
plot_3d = False



def add_type(img, env_type):
    if env_type == Env_Type.Levelground:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    elif env_type == Env_Type.Upstair:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2)
    else:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)


def add_collision(ax, xc, yc, w, h, p=None):
    if p is None:
        p = [0, 0]
    ax.plot3D([-0.3, 0.3], [xc - w + p[0], xc - w + p[0]], [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1]], color='c',
              linewidth=1)
    ax.plot3D([-0.3, 0.3], [xc + p[0], xc + p[0]], [yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]], color='c', linewidth=1)
    ax.plot3D([-0.3, 0.3], [xc + p[0], xc + p[0]], [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1]], color='c', linewidth=1)
    ax.plot3D([-0.3, 0.3], [xc + w + p[0], xc + w + p[0]], [yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]], color='c',
              linewidth=1)
    ax.plot3D([0.3, 0.3, 0.3, 0.3], [xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]],
              [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1], yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]],
              color='c',
              linewidth=1)
    ax.plot3D([-0.3, -0.3, -0.3, -0.3], [xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]],
              [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1], yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]],
              color='c',
              linewidth=1)


def add_pcd3d(ax, pcd2d, camera_pos, linewidth=1, color='r', alpha=1):
    pcd3d = pcd2d_to_3d(pcd2d)
    xx = pcd3d[:, 0]
    yy = pcd3d[:, 1] + camera_pos[0]
    zz = pcd3d[:, 2] + camera_pos[1]
    ax.plot3D(xx[0:-1:51],
              yy[0:-1:51],
              zz[0:-1:51],
              linewidth=linewidth,
              color=color,
              alpha=alpha)


def add_camera_trajectory(ax, camera_x, camera_y, linewidth=1, color='r'):
    ax.plot3D(np.zeros(np.shape(camera_x)[0]),
              camera_x,
              camera_y,
              linewidth=linewidth,
              color=color)


def pcd2d_to_3d(pcd_2d, num_rows=5):
    num_points = np.shape(pcd_2d)[0]
    pcd_3d = np.zeros((num_points * num_rows, 3))
    pcd_3d[:, 1:] = np.repeat(pcd_2d, num_rows, axis=0)
    x = np.linspace(-0.2, 0.2, num_rows).reshape((-1, 1))
    xx = np.repeat(x, num_points, axis=1)
    # weights_diag = np.diag(np.linspace(0.0001, -0.0001, num_rows))
    weights_diag = np.diag(np.linspace(0, 0, num_rows))
    idx = np.arange(num_points)
    idx_m = np.repeat(idx.reshape((-1, 1)).T, num_rows, axis=0)
    xx = xx + np.matmul(weights_diag, idx_m)
    pcd_3d[:, 0] = np.reshape(xx.T, (-1,))
    return pcd_3d


if __name__ == "__main__":
    imu_data = np.load(data_save_path + "imu_data.npy")
    imu_data = imu_data[1:, :]
    idx_frame = np.arange(np.shape(imu_data)[0])

    fig = plt.figure(figsize=(10, 10))
    if plot_3d:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 0)
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

        pcd_new = env.pcd_thin
        if i == 0:
            camera_dx_buffer.append(0)
            camera_dy_buffer.append(0)
            camera_x_buffer.append(0)
            camera_y_buffer.append(0)
            env_type_buffer.append(env.type_pred_from_nn)
            pcd_pre_os = pcd_opreator_system(pcd_new=pcd_new)
            # 特征提取
            pcd_pre_os.get_fea(_print_=True)
            pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_pre_os)
            # 地形参数
            xc, yc, w, h = pcd_pre_os.fea_to_env_paras()
            env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc, yc, w, h])
        else:
            pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_opreator_system
            pcd_pre = pcd_pre_os.pcd_new
            pcd_new_os = pcd_opreator_system(pcd_new=pcd_new)
            # 特征提取
            pcd_new_os.get_fea(_print_=True, ax=None, idx=i)
            pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
            # 特征组合
            fea_to_align_new, fea_to_align_pre, flag_method1 = align_fea(pcd_new=pcd_new_os,
                                                                         pcd_pre=pcd_pre_os,
                                                                         _print_=True)
            # icp配准
            xmove_method1, ymove_method1 = 0, 0
            try:
                xmove_method1, ymove_method1 = icp_knn(pcd_s=fea_to_align_pre,
                                                   pcd_t=fea_to_align_new)
                print("xmove1={},ymove1={}".format(xmove_method1, ymove_method1))
            except Exception as e:
                print("method1 exception:{}".format(e))
                xmove_method1, ymove_method1 = 0, 0
                flag_method1 = 1

            if abs(xmove_method1) > 0.05 or abs(ymove_method1) > 0.05:
                print("method1 移动距离过大")

            regis = icp_alignment(pcd_s=pcd_pre_os.pcd_new, pcd_t=pcd_new_os.pcd_new, flag=None)
            xmove_method2, ymove_method2 = 0, 0
            try:
                xmove_method2, ymove_method2, flag_method2 = regis.alignment()
                print("xmove2={},ymove2={}".format(xmove_method2, ymove_method2))
            except Exception as e:
                print(e)
                xmove_method2, ymove_method2 = 0, 0
                flag_method2 = 1

            if abs(xmove_method2) > 0.05 or abs(ymove_method2) > 0.05:
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

            if flag == 1 or abs(xmove) > 0.1 or abs(ymove) > 0.1:
                xmove = xmove_pre
                ymove = ymove_pre
                print("对齐失败，使用上一次的xmove_pre = {},ymove_pre = {}".format(xmove_pre, ymove_pre))

            print("当前最终xmove = {}, ymove = {}".format(xmove, ymove))
            # print("参考最终xmove = {}, ymove = {}".format(traj_x[i] - traj_x[i - 1], traj_y[i] - traj_y[i - 1]))
            camera_dx_buffer.append(xmove)
            camera_dy_buffer.append(ymove)
            camera_x_buffer.append(camera_x_buffer[-1] + xmove)
            camera_y_buffer.append(camera_y_buffer[-1] + ymove)

            # 地形重构
            if use_multi_frame_rebuild:
                if i > num_frame_rebuild:
                    pcd_total = np.zeros((0, 2))
                    pcd_multi_build_buffer = fifo_data_vec(pcd_multi_build_buffer, pcd_new)
                    for j in range(num_frame_rebuild):
                        x_j = camera_x_buffer[j - num_frame_rebuild]
                        y_j = camera_y_buffer[j - num_frame_rebuild]
                        pcd_temp = np.copy(pcd_multi_build_buffer[j])
                        pcd_temp[:, 0] += x_j
                        pcd_temp[:, 1] += y_j
                        pcd_total = np.vstack([pcd_total, pcd_temp])
                    if use_method1:
                        pcd_multi_build_os = pcd_opreator_system(pcd_new=pcd_new)
                        pcd_multi_build_os.get_fea(_print_=False)
                        pcd_multi_build_os.show_(ax, pcd_color='c')
                        xc_new, yc_new, w_new, h_new = pcd_multi_build_os.fea_to_env_paras()
                        print("multi_pcd_corner_situation:{}".format(pcd_multi_build_os.corner_situation))
                    else:
                        env_multi = Environment()
                        env_multi.pcd_thin = pcd_total
                        xc_new, yc_new, w_new, h_new = env_multi.get_fea_sa()

                    if abs(xc_new * yc_new * h_new * w_new) > 0.01:
                        env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])
                    else:
                        xc_pre, yc_pre, w_pre, h_pre = env_paras_buffer[-1][0], env_paras_buffer[-1][1], \
                            env_paras_buffer[-1][2], env_paras_buffer[-1][3]
                        if abs(xc_new * yc_new) <= 0.001:
                            xc_new, yc_new, w_new, h_new = xc_pre, yc_pre, w_pre, h_pre
                            env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])
                        else:
                            if w_new < 0.01:
                                w_new = w_pre
                            if h_new < 0.01:
                                h_new = h_pre
                            env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])

            else:
                if use_method1:
                    # todo: pcd_new_os.fea_to_env_paras()有时算出的h过小
                    xc_new, yc_new, w_new, h_new = pcd_new_os.fea_to_env_paras()
                    # xc_new, yc_new, w_new, h_new = env.get_fea_sa()
                else:
                    xc_new, yc_new, w_new, h_new = env.get_fea_sa()
                if abs(xc_new * yc_new * h_new * w_new) > 0.01:
                    env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])
                else:
                    xc_pre, yc_pre, w_pre, h_pre = env_paras_buffer[-1][0], env_paras_buffer[-1][1], \
                        env_paras_buffer[-1][2], env_paras_buffer[-1][3]
                    if abs(xc_new * yc_new) <= 0.001:
                        xc_new, yc_new, w_new, h_new = xc_pre, yc_pre, w_pre, h_pre
                        env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])
                    else:
                        if w_new < 0.01:
                            w_new = w_pre
                        if h_new < 0.01:
                            h_new = h_pre
                        env_paras_buffer = fifo_data_vec(env_paras_buffer, [xc_new, yc_new, w_new, h_new])


            # 画图
            if plot_3d:
                light_blue = mpl.cm.jet(50)
                light_red = mpl.cm.jet(220)
                add_pcd3d(ax, pcd_new, camera_pos=[camera_x_buffer[-1], camera_y_buffer[-1]], linewidth=1, color='blue')
                add_pcd3d(ax, pcd_pre, camera_pos=[camera_x_buffer[-2], camera_y_buffer[-2]], linewidth=5,
                          color=light_blue, alpha=0.2)
                add_camera_trajectory(ax, camera_x=np.array(camera_x_buffer), camera_y=np.array(camera_y_buffer),
                                      color='blue')

                add_pcd3d(ax, pcd_new, camera_pos=[traj_x[i], traj_y[i] - 0.2], linewidth=1, color='red')
                add_pcd3d(ax, pcd_pre, camera_pos=[traj_x[i - 1], traj_y[i - 1] - 0.2], linewidth=5,
                          color=light_red, alpha=0.2)
                add_camera_trajectory(ax, camera_x=traj_x[0:i], camera_y=traj_y[0:i], color='red')

                if use_multi_frame_rebuild:
                    if i > num_frame_rebuild:
                        add_collision(ax, xc_new + camera_x_buffer[-1], yc_new + camera_y_buffer[-1], w_new, h_new)
                else:
                    add_collision(ax, xc_new + camera_x_buffer[-1], yc_new + camera_y_buffer[-1], w_new, h_new)
                ax.text(0.5, 1.8, 1.8, "corner_situation:{},id:{}".format(pcd_new_os.corner_situation, i))
                ax.text(0.5, 1.8, 1.2, "corner_pre:{},id:{}".format(pcd_pre_os.corner_situation, i - 1))
                if use_method1:
                    ax.text(0.5, 0.2, 1.2, "method: new feature extraction", color='b')
                    ax.text(0.5, 0.2, 1.0, "method: previous feature extraction", color='r')
                else:
                    ax.text(0.5, 0.2, 1.2, "method: previous feature extraction", color='b')
                    ax.text(0.5, 0.2, 1.0, "method: new feature extraction", color='r')

                if flag_method1 == 1:
                    ax.text(0.5, 1.3, -0.5, "method new fails times:{}".format(sum(alignment_flag_buffer[0])),
                            color='y')
                else:
                    ax.text(0.5, 1.3, -0.5, "method new fails times:{}".format(sum(alignment_flag_buffer[0])),
                            color='g')

                if flag_method2 == 1:
                    ax.text(0.5, 1.3, -0.8, "method pre fails times:{}".format(sum(alignment_flag_buffer[1])),
                            color='y')
                else:
                    ax.text(0.5, 1.3, -0.8, "method pre fails times:{}".format(sum(alignment_flag_buffer[1])),
                            color='g')
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 3)
                ax.set_zlim(-1, 2)
                ax.view_init(elev=0, azim=0)
            else:
                pcd_new_os.show_(ax, pcd_color='r', id=int(i))
                # pcd_pre_os.show_(ax, pcd_color='b', id=int(i - 1), p_text=0.4, p_pcd=[-xmove, -ymove])
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
            plt.draw()
            plt.pause(0.1)

    if use_method1:
        np.save("traj_x_method1.npy", np.array(camera_x_buffer))
        np.save("traj_y_method1.npy", np.array(camera_y_buffer))
    else:
        np.save("traj_x_method2.npy", np.array(camera_x_buffer))
        np.save("traj_y_method2.npy", np.array(camera_y_buffer))
