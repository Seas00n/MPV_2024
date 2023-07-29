import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import stft
from Utils.Algo import *

stair_corner_situation = [1, 2, 3, 4, 5, 6, 7]
fea_Ax_buffer = [np.zeros((1,))]
fea_Bx_buffer = [np.zeros((1,))]
fea_Cx_buffer = [np.zeros((1,))]
fea_Ay_buffer = [np.zeros((1,))]
fea_By_buffer = [np.zeros((1,))]
fea_Cy_buffer = [np.zeros((1,))]
situation_buffer = []
####################
#   1.
#
#           |
#   --------|
#   |
#   |
#   |
#
#   2.
#           |
#           |
#   --------
#   3.
#
#
#           |---
#           |
#           |
#   --------
#   |
#   4.
#       -------
#       |
#   ---|
#   5.
#          |
#   |------|
#   |
#   6.
#             |
#             |
#       |-----
#       |
#   ----|
#   7.
#
#               |
#         |-----|
#   |-----|
#   |

if __name__ == '__main__':
    data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4_OPEN3D/"
    file_list = os.listdir(data_save_path)
    num_frames = len(file_list) - 3
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    for i in range(num_frames):
        corner_situation = 0
        fea_Ax_new = np.zeros((1,))
        fea_Bx_new = np.zeros((1,))
        fea_Cx_new = np.zeros((1,))
        fea_Ay_new = np.zeros((1,))
        fea_By_new = np.zeros((1,))
        fea_Cy_new = np.zeros((1,))
        ax1.cla()
        pcd_new = np.load(data_save_path + "{}_pcd2d.npy".format(i))
        ax1.scatter(pcd_new[:, 0], pcd_new[:, 1], color='red')
        X0 = pcd_new[:, 0].reshape((-1, 1))
        Y0 = pcd_new[:, 1].reshape((-1, 1))
        line1_success = False
        th = 0.05
        inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
        mean_line1 = np.nanmean(Y0[inlier_mask1])
        idx1 = np.where(abs(Y0 - mean_line1) < 0.01)[0]
        x1 = X0[idx1, :]
        y1 = Y0[idx1, :]
        line1_length = np.max(x1) - np.min(x1)
        diff_x1 = np.diff(x1, axis=0)
        if np.shape(idx1)[0] > 0 and line1_length > 0.15 and np.max(np.abs(diff_x1)) < 0.1:
            line1_success = True
            print("Line1 get")
        else:
            print("Try RANSAC Again")
            X_temp = np.delete(X0, idx1).reshape((-1, 1))
            Y_temp = np.delete(Y0, idx1).reshape((-1, 1))
            idx_all = np.delete(np.arange(0, np.shape(X0)[0]), idx1)
            try:
                inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X_temp, Y_temp, th)
                mean_line1 = mean_line_temp = np.nanmean(Y_temp[inlier_mask1])
                idx_temp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
                idx1 = idx_all[idx_temp]
                x1 = X0[idx1, :]
                y1 = Y0[idx1, :]
                line1_length = np.max(x1) - np.min(x1)
                diff_x1 = np.diff(x1, axis=0)
                if np.shape(idx_temp)[0] > 0 and line1_length > 0.15 and np.max(np.abs(diff_x1)) < 0.1:
                    line1_success = True
                    print("Line1 get")
                else:
                    print("Not get Line1")
                    print("line1_length:{}<0.15".format(line1_length))
                    print("diff_x1:{}>0.1".format(np.max(np.abs(diff_x1))))
                    line1_success = False
            except:
                print("RANSAC False")
                line1_success = False
        x2 = []
        y2 = []
        mean_line2 = 0
        if line1_success:
            # (X0,Y0)->(x1,y1),(X1,Y1)
            th = 0.12
            X1 = np.delete(X0, idx1).reshape((-1, 1))
            Y1 = np.delete(Y0, idx1).reshape((-1, 1))
            line2_success = False
            try:
                inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X1, Y1, th)
                mean_line2 = np.nanmean(Y1[inlier_mask2])
                idx2 = np.where(abs(Y1 - mean_line2) < 0.01)[0]
                x2 = X1[idx2, :]
                y2 = Y1[idx2, :]
                line2_length = np.max(x2) - np.min(x2)
                diff_x2 = np.diff(x2, axis=0)
                if np.shape(idx2)[0] > 20 and line2_length > 0.05 > np.max(abs(diff_x2)):
                    line2_success = True
                    print("Line2 get")
                elif np.shape(idx2)[0] > 20 and np.max(abs(diff_x2)) < 0.05 and line2_length < 0.05:
                    print("Try again")
                    X_temp = np.delete(X1, idx2).reshape((-1, 1))
                    Y_temp = np.delete(Y1, idx2).reshape((-1, 1))
                    idx_all = np.delete(np.arange(0, np.shape(X0)[0]), idx1)
                    idx_all = np.delete(idx_all, idx2)
                    inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X_temp, Y_temp, th)
                    mean_line2 = mean_line_temp = np.nanmean(Y_temp[inlier_mask2])
                    idx_temp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
                    idx2 = idx_all[idx_temp]
                    x2 = X1[idx2, :]
                    y2 = Y1[idx2, :]
                    line2_length = np.max(x2) - np.min(x2)
                    diff_x2 = np.diff(x2, axis=0)
                    if np.shape(idx_temp)[0] > 0 and line2_length > 0.05 > np.max(np.abs(diff_x2)):
                        line2_success = True
                        print("Line2 get")
                    else:
                        print("Not get Line2")
                        print("line1_length:{}<0.05".format(line2_length))
                        print("diff_x1:{}>0.05".format(np.max(np.abs(diff_x2))))
                        line2_success = False
            except Exception as e:
                print(e)
                line2_success = False
                print("Line2 RANSAC False")

            need_to_check_B = True
            need_to_check_A = False
            need_to_check_C = False
            if line1_success and line2_success:
                if mean_line1 > mean_line2:
                    # line1 is higher than line2
                    stair_high_x, stair_high_y = x1, y1
                    stair_low_x, stair_low_y = x2, y2
                else:
                    stair_high_x, stair_high_y = x2, y2
                    stair_low_x, stair_low_y = x1, y1
                ax1.plot(stair_low_x, stair_low_y, color='c', linewidth=2)
                ax1.plot(stair_high_x, stair_high_y, color='b', linewidth=2)
                ymin = np.min(pcd_new[:, 1])
                has_part_under_line1 = False
                if np.nanmean(stair_low_y) - ymin > 0.02:
                    has_part_under_line1 = True
                else:
                    has_part_under_line1 = False
                ymax = np.max(pcd_new[:, 1])
                has_part_up_line1 = False
                if ymax - np.nanmean(stair_high_y) > 0.02:
                    has_part_up_line1 = True
                else:
                    has_part_up_line1 = False
                xmax = np.max(pcd_new[:, 1])
                has_part_right_line1 = False
                if xmax - np.max(stair_high_x) > 0.05 and has_part_up_line1:
                    has_part_right_line1 = True
                else:
                    has_part_right_line1 = False
                xmin = np.min(pcd_new[:, 1])
                has_part_left_line1 = False
                if np.min(stair_low_x) - xmin > 0.05 and has_part_under_line1:
                    has_part_left_line1 = True
                else:
                    has_part_left_line1 = False
                if has_part_under_line1 and has_part_left_line1:
                    corner_situation = 7
                    need_to_check_A = True
                    need_to_check_B = True
                    need_to_check_C = True
                elif has_part_under_line1 and not has_part_left_line1:
                    corner_situation = 7
                    need_to_check_A = True
                    need_to_check_B = True
                    need_to_check_C = True
                elif not has_part_under_line1:
                    if has_part_up_line1:
                        corner_situation = 6
                        need_to_check_B = True
                        need_to_check_C = True
                    elif not has_part_up_line1:
                        corner_situation = 4
                        need_to_check_B = True
                        need_to_check_C = True
                print("env_type:{}".format(corner_situation))
                if need_to_check_A:
                    print('A feature finding')
                    stair_low_left_corner_x = np.min(stair_low_x)
                    idx_fea_A = np.where(abs(stair_low_x - stair_low_left_corner_x) < 0.01)[0]
                    if np.shape(idx_fea_A)[0] > 10:
                        fea_Ax_new = stair_low_x[idx_fea_A]
                        fea_Ay_new = stair_low_y[idx_fea_A]
                        ax1.plot(fea_Ax_new, fea_Ay_new, '.:g', linewidth=6)
                        print("find feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                    else:
                        mean_Ax = np.nanmean(stair_low_x[idx_fea_A])
                        mean_Ay = np.nanmean(stair_low_y[idx_fea_A])
                        rand = np.random.rand(20)
                        fea_Ax_new = mean_Ax + rand * 0.001
                        rand = np.random.rand(20)
                        fea_Ay_new = mean_Ay + rand * 0.001
                        print("complete feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))

                if need_to_check_B:
                    print('B feature finding')
                    stair_low_right_corner_x = min(np.max(stair_low_x), np.min(stair_high_x))
                    idx_fea_B = np.where(abs(stair_low_x - stair_low_right_corner_x) < 0.01)[0]
                    if np.shape(idx_fea_B)[0] > 10:
                        fea_Bx_new = stair_low_x[idx_fea_B]
                        fea_By_new = stair_low_y[idx_fea_B]
                        ax1.plot(fea_Bx_new, fea_By_new, '.:y', linewidth=6)
                        print("find feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                    else:
                        mean_Bx = np.nanmean(stair_low_x[idx_fea_B])
                        mean_By = np.nanmean(stair_low_y[idx_fea_B])
                        rand = np.random.rand(20)
                        fea_Bx_new = mean_Bx + rand * 0.001
                        rand = np.random.rand(20)
                        fea_By_new = mean_By + rand * 0.001
                        print("complete feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                if need_to_check_C:
                    print('C feature finding')
                    if not has_part_right_line1:
                        stair_high_left_corner_x = min(stair_high_x)
                        idx_fea_C = np.where(stair_high_x - stair_high_left_corner_x < 0.01)[0]
                        if np.shape(idx_fea_C)[0] > 10:
                            fea_Cx_new = stair_high_x[idx_fea_C]
                            fea_Cy_new = stair_high_y[idx_fea_C]
                            ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                            print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                        else:
                            mean_Cx = np.nanmean(stair_high_x[idx_fea_C])
                            mean_Cy = np.nanmean(stair_high_y[idx_fea_C])
                            rand = np.random.rand(20)
                            fea_Cx_new = mean_Cx + rand * 0.001
                            rand = np.random.rand(20)
                            fea_Cy_new = mean_Cy + rand * 0.001
                            print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        idx_right_part = np.where(X0 > np.max(stair_high_x))[0]
                        X_right_part = X0[idx_right_part, :]
                        Y_right_part = Y0[idx_right_part, :]
                        ax1.plot(fea_Bx_new, fea_By_new, '.:y')
                        stair_high_left_corner_x = np.min(X_right_part)
                        idx_fea_C = np.where(abs(X_right_part - stair_high_left_corner_x) < 0.01)[0]
                        if np.shape(idx_fea_C)[0] > 10:
                            fea_Cx_new = X_right_part[idx_fea_C]
                            fea_Cy_new = Y_right_part[idx_fea_C]
                            ax1.plot(fea_Cx_new, fea_Cy_new, '.:m')
                            print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                        else:
                            mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                            mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                            rand = np.random.rand(20)
                            fea_Cx_new = mean_Cx + rand * 0.001
                            rand = np.random.rand(20)
                            fea_Cy_new = mean_Cy + rand * 0.001
                            print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))

            elif line1_success and not line2_success:
                ax1.plot(x1, y1, color='c', linewidth=2)
                ymin = np.min(pcd_new[:, 1])
                has_part_under_line1 = False
                if np.nanmean(y1) - ymin > 0.02:
                    has_part_under_line1 = True
                else:
                    has_part_under_line1 = False
                ymax = np.max(pcd_new[:, 1])
                has_part_up_line1 = False
                if ymax - np.nanmean(y1) > 0.02:
                    has_part_up_line1 = True
                else:
                    has_part_up_line1 = False
                xmin = np.min(pcd_new[:, 1])
                has_part_left_line1 = False
                if np.min(x1) - xmin > 0.05 and has_part_under_line1:
                    has_part_left_line1 = True
                else:
                    has_part_left_line1 = False
                xmax = np.max(pcd_new[:, 1])
                has_part_right_line1 = False
                if xmax - np.max(x1) > 0.05 and has_part_up_line1:
                    has_part_right_line1 = True
                else:
                    has_part_right_line1 = False
                if has_part_under_line1 and has_part_left_line1:
                    if has_part_up_line1 and has_part_right_line1:
                        corner_situation = 3
                        need_to_check_A = True
                        need_to_check_C = True
                    elif has_part_up_line1 and not has_part_right_line1:
                        corner_situation = 1
                        need_to_check_A = True
                    elif not has_part_up_line1:
                        corner_situation = 5
                        need_to_check_C = True
                        need_to_check_B = False
                elif has_part_under_line1 and not has_part_left_line1:
                    if has_part_up_line1 and has_part_right_line1:
                        corner_situation = 3
                        need_to_check_A = True
                        need_to_check_C = True
                    elif has_part_up_line1 and not has_part_right_line1:
                        corner_situation = 1
                        need_to_check_A = True
                    elif not has_part_up_line1:
                        corner_situation = 5
                        need_to_check_C = True
                        need_to_check_B = False
                elif not has_part_under_line1:
                    if has_part_up_line1 and has_part_right_line1:
                        corner_situation = 4
                        need_to_check_C = True
                    elif has_part_up_line1 and not has_part_right_line1:
                        corner_situation = 2

                print("env_type:{}".format(corner_situation))
                if need_to_check_A:
                    print('A feature finding')
                    stair_low_left_corner_x = np.min(x1)
                    idx_fea_A = np.where(abs(x1 - stair_low_left_corner_x) < 0.01)[0]
                    if np.shape(idx_fea_A)[0] > 10:
                        fea_Ax_new = x1[idx_fea_A]
                        fea_Ay_new = y1[idx_fea_A]
                        ax1.plot(fea_Ax_new, fea_Ay_new, '.:g', linewidth=6)
                        print("find feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                    else:
                        mean_Ax = np.nanmean(x1[idx_fea_A])
                        mean_Ay = np.nanmean(y1[idx_fea_A])
                        rand = np.random.rand(20)
                        fea_Ax_new = mean_Ax + rand * 0.001
                        rand = np.random.rand(20)
                        fea_Ay_new = mean_Ay + rand * 0.001
                        print("complete feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                if need_to_check_B:
                    print('B feature finding')
                    stair_low_right_corner_x = np.max(x1)
                    idx_fea_B = np.where(abs(x1 - stair_low_right_corner_x) < 0.01)[0]
                    if np.shape(idx_fea_B)[0] > 10:
                        fea_Bx_new = x1[idx_fea_B]
                        fea_By_new = y1[idx_fea_B]
                        ax1.plot(fea_Bx_new, fea_By_new, '.:y', linewidth=6)
                        print("find feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                    else:
                        mean_Bx = np.nanmean(x1[idx_fea_B])
                        mean_By = np.nanmean(y1[idx_fea_B])
                        rand = np.random.rand(20)
                        fea_Bx_new = mean_Bx + rand * 0.001
                        rand = np.random.rand(20)
                        fea_By_new = mean_By + rand * 0.001
                        print("complete feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                if need_to_check_C:
                    print('C feature finding')
                    if has_part_right_line1:
                        idx_right_part = np.where(X0 > np.max(x1))[0]
                        X_right_part = X0[idx_right_part, :]
                        Y_right_part = Y0[idx_right_part, :]
                        stair_high_left_corner_x = np.min(X_right_part)
                        idx_fea_C = np.where(abs(X_right_part - stair_high_left_corner_x) < 0.01)[0]
                        if np.shape(idx_fea_C)[0] > 10:
                            fea_Cx_new = X_right_part[idx_fea_C]
                            fea_Cy_new = Y_right_part[idx_fea_C]
                            ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                            print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                        else:
                            mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                            mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                            rand = np.random.rand(20)
                            fea_Cx_new = mean_Cx + rand * 0.001
                            rand = np.random.rand(20)
                            fea_Cy_new = mean_Cy + rand * 0.001
                            print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        stair_low_left_corner_x = np.min(x1)
                        idx_fea_C = np.where(abs(x1 - stair_low_left_corner_x) < 0.01)[0]
                        if np.shape(idx_fea_C)[0] > 10:
                            fea_Cx_new = x1[idx_fea_C]
                            fea_Cy_new = y1[idx_fea_C]
                            ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                            print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                        else:
                            mean_Cx = np.nanmean(x1[idx_fea_C])
                            mean_Cy = np.nanmean(y1[idx_fea_C])
                            rand = np.random.rand(20)
                            fea_Cx_new = mean_Cx + rand * 0.001
                            rand = np.random.rand(20)
                            fea_Cy_new = mean_Cy + rand * 0.001
                            print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
        if i < 1:
            pcd_pre = pcd_new
            fea_Ax_buffer.append(fea_Ax_new)
            fea_Bx_buffer.append(fea_Bx_new)
            fea_Cx_buffer.append(fea_Cx_new)
            fea_Ay_buffer.append(fea_Ay_new)
            fea_By_buffer.append(fea_By_new)
            fea_Cy_buffer.append(fea_Cy_new)
        else:
            fea_Ax_buffer[0] = fea_Ax_buffer[1]
            fea_Bx_buffer[0] = fea_Bx_buffer[1]
            fea_Cx_buffer[0] = fea_Cx_buffer[1]
            fea_Ay_buffer[0] = fea_Ay_buffer[1]
            fea_By_buffer[0] = fea_By_buffer[1]
            fea_Cy_buffer[0] = fea_Cy_buffer[1]

            fea_Ax_buffer.append(fea_Ax_new)
            fea_Bx_buffer.append(fea_Bx_new)
            fea_Cx_buffer.append(fea_Cx_new)
            fea_Ay_buffer.append(fea_Ay_new)
            fea_By_buffer.append(fea_By_new)
            fea_Cy_buffer.append(fea_Cy_new)
            pcd_pre = pcd_new

        situation_buffer.append(corner_situation)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-1, 0)
        plt.text(0.1, -0.1, 'id: {}'.format(i))
        plt.text(0.1, -0.2, 'corner: {}'.format(corner_situation))
        print('plt_draw')
        # plt.draw()
        print("____________________________{}____________________".format(i))
        plt.pause(0.01)