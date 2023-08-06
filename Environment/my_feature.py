import numpy as np
from Utils.Algo import *
import cv2
import sys


def align_rule(fea_A, fea_A_get, fea_B, fea_B_get, fea_C, fea_C_get,
               corner_situation):
    pcd_to_align_new = np.zeros((0, 2))
    pcd_to_align_pre = np.zeros((0, 2))
    fea_A_new = fea_A[0]
    fea_A_pre = fea_A[1]
    fea_B_new = fea_B[0]
    fea_B_pre = fea_B[1]
    fea_C_new = fea_C[0]
    fea_C_pre = fea_C[1]
    fea_A_get_new = fea_A_get[0]
    fea_A_get_pre = fea_A_get[1]
    fea_B_get_new = fea_B_get[0]
    fea_B_get_pre = fea_B_get[1]
    fea_C_get_new = fea_C_get[0]
    fea_C_get_pre = fea_C_get[1]
    corner_new = corner_situation[0]
    corner_pre = corner_situation[1]
    fea_component_new = [0, 0, 0]
    fea_component_pre = [0, 0, 0]
    if corner_new == 5 and corner_pre == 1:
        if fea_C_get_new and fea_A_get_pre:
            fea_len = min(np.shape(fea_C_new)[0], np.shape(fea_A_pre)[0])
            pcd_to_align_new = np.vstack([pcd_to_align_new, fea_C_new[0:fea_len, :]])
            pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_A_pre[0:fea_len, :]])
            print("fea_C and fea_A")
            fea_component_new = [0, 0, fea_len]
            fea_component_pre = [fea_len, 0, 0]
            return pcd_to_align_new, pcd_to_align_pre, fea_component_new, fea_component_pre
    if corner_new == 1 and corner_pre == 5:
        if fea_A_get_new and fea_C_get_pre:
            fea_len = min(np.shape(fea_A_new)[0], np.shape(fea_C_pre)[0])
            pcd_to_align_new = np.vstack([pcd_to_align_new, fea_A_new[0:fea_len, :]])
            pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_C_pre[0:fea_len, :]])
            print("fea_A and fea_C")
            fea_component_new = [fea_len, 0, 0]
            fea_component_pre = [0, 0, fea_len]
            return pcd_to_align_new, pcd_to_align_pre, fea_component_new, fea_component_pre
    if corner_new == 1 and corner_pre == 6:
        if fea_A_get_new and fea_C_get_pre:
            fea_len = min(np.shape(fea_A_new)[0], np.shape(fea_C_pre)[0])
            pcd_to_align_new = np.vstack([pcd_to_align_new, fea_A_new[0:fea_len, :]])
            pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_C_pre[0:fea_len, :]])
            print("fea_A and fea_C")
            fea_component_new = [fea_len, 0, 0]
            fea_component_pre = [0, 0, fea_len]
            return pcd_to_align_new, pcd_to_align_pre, fea_component_new, fea_component_pre
    if corner_new == 6 and corner_pre == 1:
        if fea_C_get_new and fea_A_get_pre:
            fea_len = min(np.shape(fea_C_new)[0], np.shape(fea_A_pre)[0])
            pcd_to_align_new = np.vstack([pcd_to_align_new, fea_C_new[0:fea_len, :]])
            pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_A_pre[0:fea_len, :]])
            print("fea_C and fea_A")
            fea_component_new = [0, 0, fea_len]
            fea_component_pre = [fea_len, 0, 0]
            return pcd_to_align_new, pcd_to_align_pre, fea_component_new, fea_component_pre
    if corner_new == 1 and corner_pre == 4:
        if fea_A_get_new and fea_C_get_pre:
            fea_len = min(np.shape(fea_A_new)[0], np.shape(fea_C_pre)[0])
            pcd_to_align_new = np.vstack([pcd_to_align_new, fea_A_new[0:fea_len, :]])
            pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_C_pre[0:fea_len, :]])
            print("fea_A and fea_C")
            fea_component_new = [fea_len, 0, 0]
            fea_component_pre = [0, 0, fea_len]
            return pcd_to_align_new, pcd_to_align_pre, fea_component_new, fea_component_pre
    if fea_A_get_new and fea_A_get_pre:
        fea_len = min(np.shape(fea_A_new)[0], np.shape(fea_A_pre)[0])
        pcd_to_align_new = np.vstack([pcd_to_align_new, fea_A_new[0:fea_len, :]])
        pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_A_pre[0:fea_len, :]])
        print("fea_A and fea_A")
        fea_component_new[0] = fea_len
        fea_component_pre[0] = fea_len
    if fea_B_get_new and fea_B_get_pre:
        fea_len = min(np.shape(fea_B_new)[0], np.shape(fea_B_pre)[0])
        pcd_to_align_new = np.vstack([pcd_to_align_new, fea_B_new[0:fea_len, :]])
        pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_B_pre[0:fea_len, :]])
        print("fea_B and fea_B")
        fea_component_new[1] = fea_len
        fea_component_pre[1] = fea_len
    if fea_C_get_new and fea_C_get_pre:
        fea_len = min(np.shape(fea_C_new)[0], np.shape(fea_C_pre)[0])
        pcd_to_align_new = np.vstack([pcd_to_align_new, fea_C_new[0:fea_len, :]])
        pcd_to_align_pre = np.vstack([pcd_to_align_pre, fea_C_pre[0:fea_len, :]])
        print("fea_C and fea_C")
        fea_component_new[2] = fea_len
        fea_component_pre[2] = fea_len
    return pcd_to_align_new, pcd_to_align_pre, fea_component_new, fea_component_pre


def get_fea_sa(pcd_new):
    fea_Ax_new = np.zeros((1,))
    fea_Bx_new = np.zeros((1,))
    fea_Cx_new = np.zeros((1,))
    fea_Ay_new = np.zeros((1,))
    fea_By_new = np.zeros((1,))
    fea_Cy_new = np.zeros((1,))
    corner_situation = 0
    X0 = pcd_new[:, 0].reshape((-1, 1))
    Y0 = pcd_new[:, 1].reshape((-1, 1))
    line1_success = False
    th = 0.1
    x1 = []
    y1 = []
    mean_line1 = 0
    try:
        inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
        mean_line1 = np.nanmean(Y0[inlier_mask1])
        idx1 = np.where(abs(Y0 - mean_line1) < 0.01)[0]
        x1 = X0[idx1, :]
        y1 = Y0[idx1, :]
        line1_length = np.max(x1) - np.min(x1)
        diff_x1 = np.diff(x1, axis=0)
        if np.shape(idx1)[0] > 0 and line1_length > 0.1 and np.max(np.abs(diff_x1)) < 0.1:
            line1_success = True
            print("Line1 get")
        else:
            print("Line1 Try RANSAC Again")
            X_temp = np.delete(X0, idx1).reshape((-1, 1))
            Y_temp = np.delete(Y0, idx1).reshape((-1, 1))
            idx_all = np.delete(np.arange(0, np.shape(X0)[0]), idx1)
            inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X_temp, Y_temp, th)
            mean_line1 = mean_line_temp = np.nanmean(Y_temp[inlier_mask1])
            idx_temp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
            idx1 = idx_all[idx_temp]
            x1 = X0[idx1, :]
            y1 = Y0[idx1, :]
            line1_length = np.max(x1) - np.min(x1)
            diff_x1 = np.diff(x1, axis=0)
            if np.shape(idx_temp)[0] > 0 and line1_length > 0.1 and np.max(np.abs(diff_x1)) < 0.1:
                line1_success = True
                print("Line1 get")
            else:
                print("Not get Line1")
                print("line1_length:{}<0.15".format(line1_length) + "diff_x1:{}>0.1".format(np.max(np.abs(diff_x1))))
                line1_success = False
    except Exception as e:
        print(e)
        print("Line1 RANSAC False, No Line in the picture")
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
                print("Line2 RANSAC Try again")
                X_temp = np.delete(X1, idx2).reshape((-1, 1))
                Y_temp = np.delete(Y1, idx2).reshape((-1, 1))
                idx_all = np.delete(np.arange(0, np.shape(X0)[0]), idx1)
                idx_all = np.delete(idx_all, idx2)
                inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X_temp, Y_temp, th)
                mean_line2 = mean_line_temp = np.nanmean(Y_temp[inlier_mask2])
                idx_temp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
                idx2 = idx_all[idx_temp]
                x2 = X0[idx2, :]
                y2 = Y0[idx2, :]
                line2_length = np.max(x2) - np.min(x2)
                diff_x2 = np.diff(x2, axis=0)
                if np.shape(idx_temp)[0] > 0 and line2_length > 0.05 > np.max(np.abs(diff_x2)):
                    line2_success = True
                    print("Line2 get")
                else:
                    print("Not get Line2")
                    print(
                        "line2_length:{}<0.05".format(line2_length) + "diff_x2:{}>0.05".format(np.max(np.abs(diff_x2))))
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
            # ax1.plot(stair_low_x, stair_low_y, color='c', linewidth=2)
            # ax1.plot(stair_high_x, stair_high_y, color='b', linewidth=2)
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
            if xmax - np.max(stair_high_x) > 0.02 and has_part_up_line1:
                has_part_right_line1 = True
            else:
                has_part_right_line1 = False
            xmin = np.min(pcd_new[:, 1])
            has_part_left_line1 = False
            if np.min(stair_low_x) - xmin > 0.02 and has_part_under_line1:
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
                stair_low_left_corner_y = stair_low_y[np.argmin(stair_low_x)]
                # idx_fea_A = np.where(abs(stair_low_x - stair_low_left_corner_x) < 0.05)[0]
                idx_fea_A = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_left_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_left_corner_y) < 0.05)
                if np.shape(idx_fea_A)[0] > 10:
                    # fea_Ax_new = stair_low_x[idx_fea_A]
                    # fea_Ay_new = stair_low_y[idx_fea_A]
                    fea_Ax_new = pcd_new[idx_fea_A, 0]
                    fea_Ay_new = pcd_new[idx_fea_A, 1]
                    # ax1.plot(fea_Ax_new, fea_Ay_new, '.:g', linewidth=6)
                    print("find feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                else:
                    mean_Ax = np.nanmean(stair_low_x[idx_fea_A])
                    mean_Ay = np.nanmean(stair_low_y[idx_fea_A])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ax_new = mean_Ax + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ay_new = mean_Ay + rand * 0.001
                    print("complete feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))

            if need_to_check_B:
                print('B feature finding')
                if np.max(stair_low_x) < np.min(stair_high_x):
                    stair_low_right_corner_x = np.max(stair_low_x)
                    stair_low_right_corner_y = stair_low_y[np.argmax(stair_low_x)]
                else:
                    stair_low_right_corner_x = np.min(stair_high_x)
                    stair_low_right_corner_y = np.nanmean(stair_low_y)
                # idx_fea_B = np.where(abs(stair_low_x - stair_low_right_corner_x) < 0.05)[0]
                idx_fea_B = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_right_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_right_corner_y) < 0.05)
                if np.shape(idx_fea_B)[0] > 10:
                    # fea_Bx_new = stair_low_x[idx_fea_B]
                    # fea_By_new = stair_low_y[idx_fea_B]
                    fea_Bx_new = pcd_new[idx_fea_B, 0].reshape((-1, 1))
                    fea_By_new = pcd_new[idx_fea_B, 1].reshape((-1, 1))
                    # ax1.plot(fea_Bx_new, fea_By_new, '.:y', linewidth=6)
                    print("find feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                else:
                    mean_Bx = np.nanmean(stair_low_x[idx_fea_B])
                    mean_By = np.nanmean(stair_low_y[idx_fea_B])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Bx_new = mean_Bx + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_By_new = mean_By + rand * 0.001
                    print("complete feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
            if need_to_check_C:
                print('C feature finding')
                if not has_part_right_line1:
                    stair_high_left_corner_x = np.min(stair_high_x)
                    stair_high_left_corner_y = stair_high_y[np.argmin(stair_high_x)]
                    # idx_fea_C = np.where(stair_high_x - stair_high_left_corner_x < 0.05)[0]
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = stair_high_x[idx_fea_C]
                        # fea_Cy_new = stair_high_y[idx_fea_C]
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(stair_high_x[idx_fea_C])
                        mean_Cy = np.nanmean(stair_high_y[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                else:
                    idx_right_part = np.where(X0 > np.max(stair_high_x))[0]
                    X_right_part = X0[idx_right_part, :]
                    Y_right_part = Y0[idx_right_part, :]
                    # ax1.plot(fea_Bx_new, fea_By_new, '.:y')
                    stair_high_left_corner_x = np.min(X_right_part)
                    stair_high_left_corner_y = Y_right_part[np.argmin(X_right_part)]
                    # idx_fea_C = np.where(abs(X_right_part - stair_high_left_corner_x) < 0.05)[0]
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = X_right_part[idx_fea_C]
                        # fea_Cy_new = Y_right_part[idx_fea_C]
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m')
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                        mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))

        elif line1_success and not line2_success:
            # ax1.plot(x1, y1, color='c', linewidth=2)
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
            if np.min(x1) - xmin > 0.02 and has_part_under_line1:
                has_part_left_line1 = True
            else:
                has_part_left_line1 = False
            xmax = np.max(pcd_new[:, 1])
            has_part_right_line1 = False
            if xmax - np.max(x1) > 0.02 and has_part_up_line1:
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
                stair_low_left_corner_y = y1[np.argmin(x1)]
                # idx_fea_A = np.where(abs(x1 - stair_low_left_corner_x) < 0.05)[0]
                idx_fea_A = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_left_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_left_corner_y) < 0.05)
                if np.shape(idx_fea_A)[0] > 10:
                    # fea_Ax_new = x1[idx_fea_A]
                    # fea_Ay_new = y1[idx_fea_A]
                    fea_Ax_new = pcd_new[idx_fea_A, 0].reshape((-1, 1))
                    fea_Ay_new = pcd_new[idx_fea_A, 1].reshape((-1, 1))
                    # ax1.plot(fea_Ax_new, fea_Ay_new, '.:g', linewidth=6)
                    print("find feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                else:
                    mean_Ax = np.nanmean(x1[idx_fea_A])
                    mean_Ay = np.nanmean(y1[idx_fea_A])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ax_new = mean_Ax + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ay_new = mean_Ay + rand * 0.001
                    print("complete feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
            if need_to_check_B:
                print('B feature finding')
                stair_low_right_corner_x = np.max(x1)
                stair_low_right_corner_y = y1[np.argmax(x1)]
                # idx_fea_B = np.where(abs(x1 - stair_low_right_corner_x) < 0.05)[0]
                idx_fea_B = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_right_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_right_corner_y) < 0.05)
                if np.shape(idx_fea_B)[0] > 10:
                    # fea_Bx_new = x1[idx_fea_B]
                    # fea_By_new = y1[idx_fea_B]
                    fea_Bx_new = pcd_new[idx_fea_B, 0].reshape((-1, 1))
                    fea_By_new = pcd_new[idx_fea_B, 1].reshape((-1, 1))
                    # ax1.plot(fea_Bx_new, fea_By_new, '.:y', linewidth=6)
                    print("find feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                else:
                    mean_Bx = np.nanmean(x1[idx_fea_B])
                    mean_By = np.nanmean(y1[idx_fea_B])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Bx_new = mean_Bx + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_By_new = mean_By + rand * 0.001
                    print("complete feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
            if need_to_check_C:
                print('C feature finding')
                if has_part_right_line1:
                    idx_right_part = np.where(X0 > np.max(x1))[0]
                    X_right_part = X0[idx_right_part, :]
                    Y_right_part = Y0[idx_right_part, :]
                    stair_high_left_corner_x = np.min(X_right_part)
                    stair_high_left_corner_y = Y_right_part[np.argmin(X_right_part)]
                    # idx_fea_C = np.where(abs(X_right_part - stair_high_left_corner_x) < 0.05)[0]
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = X_right_part[idx_fea_C]
                        # fea_Cy_new = Y_right_part[idx_fea_C]
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                        mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                else:
                    stair_high_left_corner_x = np.min(x1)
                    stair_high_left_corner_y = y1[np.argmin(x1)]
                    # idx_fea_C = np.where(abs(x1 - stair_high_left_corner_x) < 0.05)[0]
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = x1[idx_fea_C]
                        # fea_Cy_new = y1[idx_fea_C]
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(x1[idx_fea_C])
                        mean_Cy = np.nanmean(y1[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
    fea_A = np.zeros((0, 2))
    if np.shape(fea_Ax_new)[0] > 10:
        fea_A = np.hstack([fea_Ax_new, fea_Ay_new])
    fea_B = np.zeros((0, 2))
    if np.shape(fea_Bx_new)[0] > 10:
        fea_B = np.hstack([fea_Bx_new, fea_By_new])
    fea_C = np.zeros((0, 2))
    if np.shape(fea_Cx_new)[0] > 10:
        fea_C = np.hstack([fea_Cx_new, fea_Cy_new])

    return fea_A, fea_B, fea_C, corner_situation


def get_paras_sa(pcd_new):
    fea_Ax_new = np.zeros((1,))
    fea_Bx_new = np.zeros((1,))
    fea_Cx_new = np.zeros((1,))
    fea_Ay_new = np.zeros((1,))
    fea_By_new = np.zeros((1,))
    fea_Cy_new = np.zeros((1,))
    fea_point_A = np.array([0,0])
    fea_point_B = np.array([0,0])
    fea_point_C = np.array([0,0])
    corner_situation = 0
    X0 = pcd_new[:, 0].reshape((-1, 1))
    Y0 = pcd_new[:, 1].reshape((-1, 1))
    line1_success = False
    th = 0.1
    x1 = []
    y1 = []
    mean_line1 = 0
    try:
        inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
        mean_line1 = np.nanmean(Y0[inlier_mask1])
        idx1 = np.where(abs(Y0 - mean_line1) < 0.01)[0]
        x1 = X0[idx1, :]
        y1 = Y0[idx1, :]
        line1_length = np.max(x1) - np.min(x1)
        diff_x1 = np.diff(x1, axis=0)
        if np.shape(idx1)[0] > 0 and line1_length > 0.1 and np.max(np.abs(diff_x1)) < 0.1:
            line1_success = True
            print("Line1 get")
        else:
            print("Line1 Try RANSAC Again")
            X_temp = np.delete(X0, idx1).reshape((-1, 1))
            Y_temp = np.delete(Y0, idx1).reshape((-1, 1))
            idx_all = np.delete(np.arange(0, np.shape(X0)[0]), idx1)
            inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X_temp, Y_temp, th)
            mean_line1 = mean_line_temp = np.nanmean(Y_temp[inlier_mask1])
            idx_temp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
            idx1 = idx_all[idx_temp]
            x1 = X0[idx1, :]
            y1 = Y0[idx1, :]
            line1_length = np.max(x1) - np.min(x1)
            diff_x1 = np.diff(x1, axis=0)
            if np.shape(idx_temp)[0] > 0 and line1_length > 0.1 and np.max(np.abs(diff_x1)) < 0.1:
                line1_success = True
                print("Line1 get")
            else:
                print("Not get Line1")
                print("line1_length:{}<0.15".format(line1_length) + "diff_x1:{}>0.1".format(np.max(np.abs(diff_x1))))
                line1_success = False
    except Exception as e:
        print(e)
        print("Line1 RANSAC False, No Line in the picture")
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
                print("Line2 RANSAC Try again")
                X_temp = np.delete(X1, idx2).reshape((-1, 1))
                Y_temp = np.delete(Y1, idx2).reshape((-1, 1))
                idx_all = np.delete(np.arange(0, np.shape(X0)[0]), idx1)
                idx_all = np.delete(idx_all, idx2)
                inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X_temp, Y_temp, th)
                mean_line2 = mean_line_temp = np.nanmean(Y_temp[inlier_mask2])
                idx_temp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
                idx2 = idx_all[idx_temp]
                x2 = X0[idx2, :]
                y2 = Y0[idx2, :]
                line2_length = np.max(x2) - np.min(x2)
                diff_x2 = np.diff(x2, axis=0)
                if np.shape(idx_temp)[0] > 0 and line2_length > 0.05 > np.max(np.abs(diff_x2)):
                    line2_success = True
                    print("Line2 get")
                else:
                    print("Not get Line2")
                    print(
                        "line2_length:{}<0.05".format(line2_length) + "diff_x2:{}>0.05".format(np.max(np.abs(diff_x2))))
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
            # ax1.plot(stair_low_x, stair_low_y, color='c', linewidth=2)
            # ax1.plot(stair_high_x, stair_high_y, color='b', linewidth=2)
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
            if xmax - np.max(stair_high_x) > 0.02 and has_part_up_line1:
                has_part_right_line1 = True
            else:
                has_part_right_line1 = False
            xmin = np.min(pcd_new[:, 1])
            has_part_left_line1 = False
            if np.min(stair_low_x) - xmin > 0.02 and has_part_under_line1:
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
                stair_low_left_corner_y = stair_low_y[np.argmin(stair_low_x)][0]
                fea_point_A = np.array([stair_low_left_corner_x, stair_low_left_corner_y])
                # idx_fea_A = np.where(abs(stair_low_x - stair_low_left_corner_x) < 0.05)[0]
                idx_fea_A = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_left_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_left_corner_y) < 0.05)
                if np.shape(idx_fea_A)[0] > 10:
                    # fea_Ax_new = stair_low_x[idx_fea_A]
                    # fea_Ay_new = stair_low_y[idx_fea_A]
                    fea_Ax_new = pcd_new[idx_fea_A, 0]
                    fea_Ay_new = pcd_new[idx_fea_A, 1]
                    # ax1.plot(fea_Ax_new, fea_Ay_new, '.:g', linewidth=6)
                    print("find feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                else:
                    mean_Ax = np.nanmean(stair_low_x[idx_fea_A])
                    mean_Ay = np.nanmean(stair_low_y[idx_fea_A])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ax_new = mean_Ax + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ay_new = mean_Ay + rand * 0.001
                    print("complete feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
            if need_to_check_B:
                print('B feature finding')
                if np.max(stair_low_x) < np.min(stair_high_x):
                    stair_low_right_corner_x = np.max(stair_low_x)
                    stair_low_right_corner_y = stair_low_y[np.argmax(stair_low_x)][0]
                else:
                    stair_low_right_corner_x = np.min(stair_high_x)
                    stair_low_right_corner_y = np.nanmean(stair_low_y)
                # idx_fea_B = np.where(abs(stair_low_x - stair_low_right_corner_x) < 0.05)[0]
                fea_point_B = np.array([stair_low_right_corner_x, stair_low_right_corner_y])
                idx_fea_B = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_right_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_right_corner_y) < 0.05)
                if np.shape(idx_fea_B)[0] > 10:
                    # fea_Bx_new = stair_low_x[idx_fea_B]
                    # fea_By_new = stair_low_y[idx_fea_B]
                    fea_Bx_new = pcd_new[idx_fea_B, 0].reshape((-1, 1))
                    fea_By_new = pcd_new[idx_fea_B, 1].reshape((-1, 1))
                    # ax1.plot(fea_Bx_new, fea_By_new, '.:y', linewidth=6)
                    print("find feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                else:
                    mean_Bx = np.nanmean(stair_low_x[idx_fea_B])
                    mean_By = np.nanmean(stair_low_y[idx_fea_B])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Bx_new = mean_Bx + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_By_new = mean_By + rand * 0.001
                    print("complete feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))

            if need_to_check_C:
                print('C feature finding')
                if not has_part_right_line1:
                    stair_high_left_corner_x = np.min(stair_high_x)
                    stair_high_left_corner_y = stair_high_y[np.argmin(stair_high_x)][0]
                    # idx_fea_C = np.where(stair_high_x - stair_high_left_corner_x < 0.05)[0]
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    fea_point_C = np.array([stair_high_left_corner_x, stair_high_left_corner_y])
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = stair_high_x[idx_fea_C]
                        # fea_Cy_new = stair_high_y[idx_fea_C]
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(stair_high_x[idx_fea_C])
                        mean_Cy = np.nanmean(stair_high_y[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                else:
                    idx_right_part = np.where(X0 > np.max(stair_high_x))[0]
                    X_right_part = X0[idx_right_part, :]
                    Y_right_part = Y0[idx_right_part, :]
                    # ax1.plot(fea_Bx_new, fea_By_new, '.:y')
                    stair_high_left_corner_x = np.min(X_right_part)
                    stair_high_left_corner_y = Y_right_part[np.argmin(X_right_part)][0]
                    # idx_fea_C = np.where(abs(X_right_part - stair_high_left_corner_x) < 0.05)[0]
                    fea_point_C = np.array([stair_high_left_corner_x, stair_high_left_corner_y])
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = X_right_part[idx_fea_C]
                        # fea_Cy_new = Y_right_part[idx_fea_C]
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m')
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                        mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))

        elif line1_success and not line2_success:
            # ax1.plot(x1, y1, color='c', linewidth=2)
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
            if np.min(x1) - xmin > 0.02 and has_part_under_line1:
                has_part_left_line1 = True
            else:
                has_part_left_line1 = False
            xmax = np.max(pcd_new[:, 1])
            has_part_right_line1 = False
            if xmax - np.max(x1) > 0.02 and has_part_up_line1:
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
                stair_low_left_corner_y = y1[np.argmin(x1)][0]
                # idx_fea_A = np.where(abs(x1 - stair_low_left_corner_x) < 0.05)[0]
                fea_point_A = np.array([stair_low_left_corner_x, stair_low_left_corner_y])
                idx_fea_A = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_left_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_left_corner_y) < 0.05)
                if np.shape(idx_fea_A)[0] > 10:
                    # fea_Ax_new = x1[idx_fea_A]
                    # fea_Ay_new = y1[idx_fea_A]
                    fea_Ax_new = pcd_new[idx_fea_A, 0].reshape((-1, 1))
                    fea_Ay_new = pcd_new[idx_fea_A, 1].reshape((-1, 1))
                    # ax1.plot(fea_Ax_new, fea_Ay_new, '.:g', linewidth=6)
                    print("find feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
                else:
                    mean_Ax = np.nanmean(x1[idx_fea_A])
                    mean_Ay = np.nanmean(y1[idx_fea_A])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ax_new = mean_Ax + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ay_new = mean_Ay + rand * 0.001
                    print("complete feature A:{},{}".format(np.nanmean(fea_Ax_new), np.nanmean(fea_Ay_new)))
            if need_to_check_B:
                print('B feature finding')
                stair_low_right_corner_x = np.max(x1)
                stair_low_right_corner_y = y1[np.argmax(x1)][0]
                # idx_fea_B = np.where(abs(x1 - stair_low_right_corner_x) < 0.05)[0]
                fea_point_B = np.array([stair_low_right_corner_x, stair_low_right_corner_y])
                idx_fea_B = np.logical_and(np.abs(pcd_new[:, 0] - stair_low_right_corner_x) < 0.05,
                                           np.abs(pcd_new[:, 1] - stair_low_right_corner_y) < 0.05)
                if np.shape(idx_fea_B)[0] > 10:
                    # fea_Bx_new = x1[idx_fea_B]
                    # fea_By_new = y1[idx_fea_B]
                    fea_Bx_new = pcd_new[idx_fea_B, 0].reshape((-1, 1))
                    fea_By_new = pcd_new[idx_fea_B, 1].reshape((-1, 1))
                    # ax1.plot(fea_Bx_new, fea_By_new, '.:y', linewidth=6)
                    print("find feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
                else:
                    mean_Bx = np.nanmean(x1[idx_fea_B])
                    mean_By = np.nanmean(y1[idx_fea_B])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Bx_new = mean_Bx + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_By_new = mean_By + rand * 0.001
                    print("complete feature B:{},{}".format(np.nanmean(fea_Bx_new), np.nanmean(fea_By_new)))
            if need_to_check_C:
                print('C feature finding')
                if has_part_right_line1:
                    idx_right_part = np.where(X0 > np.max(x1))[0]
                    X_right_part = X0[idx_right_part, :]
                    Y_right_part = Y0[idx_right_part, :]
                    stair_high_left_corner_x = np.min(X_right_part)
                    stair_high_left_corner_y = Y_right_part[np.argmin(X_right_part)][0]
                    # idx_fea_C = np.where(abs(X_right_part - stair_high_left_corner_x) < 0.05)[0]
                    fea_point_C = np.array([stair_high_left_corner_x, stair_high_left_corner_y])
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = X_right_part[idx_fea_C]
                        # fea_Cy_new = Y_right_part[idx_fea_C]
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                        mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                else:
                    stair_high_left_corner_x = np.min(x1)
                    stair_high_left_corner_y = y1[np.argmin(x1)][0]
                    # idx_fea_C = np.where(abs(x1 - stair_high_left_corner_x) < 0.05)[0]
                    fea_point_C = np.array([stair_high_left_corner_x,stair_high_left_corner_y])
                    idx_fea_C = np.logical_and(np.abs(pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                               np.abs(pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                    if np.shape(idx_fea_C)[0] > 10:
                        # fea_Cx_new = x1[idx_fea_C]
                        # fea_Cy_new = y1[idx_fea_C]
                        fea_Cx_new = pcd_new[idx_fea_C, 0].reshape((-1, 1))
                        fea_Cy_new = pcd_new[idx_fea_C, 1].reshape((-1, 1))
                        # ax1.plot(fea_Cx_new, fea_Cy_new, '.:m', linewidth=6)
                        print("find feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))
                    else:
                        mean_Cx = np.nanmean(x1[idx_fea_C])
                        mean_Cy = np.nanmean(y1[idx_fea_C])
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cx_new = mean_Cx + rand * 0.001
                        rand = np.random.rand(20).reshape((-1, 1))
                        fea_Cy_new = mean_Cy + rand * 0.001
                        print("complete feature C:{},{}".format(np.nanmean(fea_Cx_new), np.nanmean(fea_Cy_new)))

    return fea_point_A, fea_point_B, fea_point_C, corner_situation


def cal_paras_from_fea_sa(fea_point_A, fea_point_B, fea_point_C, corner_situation):
    get_fea_point_A = True
    get_fea_point_B = True
    get_fea_point_C = True
    xc, yc, w, h = 0, 0, 0, 0
    print(corner_situation)
    if fea_point_A[0]*fea_point_A[1] == 0:
        get_fea_point_A = False
    if fea_point_B[0]*fea_point_B[1] == 0:
        get_fea_point_B = False
    if fea_point_C[0] * fea_point_C[1] == 0:
        get_fea_point_C = False
    if get_fea_point_A and get_fea_point_C:
        xc, yc = fea_point_A[0], fea_point_A[1]
        if get_fea_point_B:
            w = fea_point_B[0]-fea_point_C[0]
            h = fea_point_C[1]-fea_point_B[1]
        else:
            w = fea_point_C[0]-fea_point_A[0]
            h = fea_point_C[1]-fea_point_A[1]
    if get_fea_point_A and not get_fea_point_C:
        xc, yc = fea_point_A[0], fea_point_A[1]
        if get_fea_point_B:
            w = fea_point_B[0]-fea_point_A[0]

    if not get_fea_point_A and get_fea_point_C:
        xc, yc = fea_point_C[0], fea_point_C[1]
        if get_fea_point_B:
            h = fea_point_C[1]-fea_point_B[1]

    return xc, yc, w, h


def is_converge(x, y, scale):
    scale = scale * 0.001
    a = abs(x) < scale
    b = abs(y) < scale
    return a & b


def del_miss(indeces, dist, max_dist, th_rate=0.7):
    th_dist = max_dist * th_rate
    return np.array(indeces[:][np.where(dist[:] < th_dist)[0]])


def icp(pcd_s, pcd_t, pcd_s_component, pcd_t_component, max_iterate=50):
    min_len = min(np.shape(pcd_s)[0], np.shape(pcd_t)[0])
    pcd_s_temp = pcd_s[0:min_len, :].astype(np.float32)
    pcd_t_temp = pcd_t[0:min_len, :].astype(np.float32)
    print("pcd_s:{}".format(pcd_s_component))
    print("pcd_t:{}".format(pcd_t_component))
    knn = cv2.ml.KNearest_create()
    responses = np.array(range(len(pcd_t_temp[:, 0]))).astype(np.float32)
    knn.train(pcd_s_temp, cv2.ml.ROW_SAMPLE, responses)
    xmove, ymove = 0, 0
    scale_x = np.max(pcd_s_temp[:, 0]) - np.min(pcd_s_temp[:, 0])
    scale_y = np.max(pcd_s_temp[:, 1]) - np.min(pcd_s_temp[:, 1])

    scale = max(scale_x, scale_y)
    for i in range(max_iterate):

        ret, results, neighbours, dist = knn.findNearest(pcd_t_temp, 1)

        indeces = results.astype(np.int32)

        max_dist = sys.maxsize
        indeces = del_miss(indeces, dist, max_dist)

        x_i = pcd_s_temp[indeces, 0]
        y_i = pcd_s_temp[indeces, 1]
        x_j = pcd_t_temp[indeces, 0]
        y_j = pcd_t_temp[indeces, 1]

        dist_x = np.nanmean(x_i - x_j)
        dist_y = np.nanmean(y_i - y_j)

        pcd_t_temp[:, 0] += dist_x
        pcd_t_temp[:, 1] += dist_y
        xmove += dist_x
        ymove += dist_y

        if (is_converge(dist_x, dist_y, scale)):
            break

    return xmove, ymove

def get_fea_sa_original(pcd_new):
        xc = yc = w = h = 0
        X = pcd_new[:, 0]
        Y = pcd_new[:, 1]
        if np.max(Y) - np.min(Y) < 0.1:
            print('点云高度差过小，当前地形应该是平地')
            return xc, yc, w, h
        th = 0.05
        X0 = X[Y - np.min(Y) < 0.25].reshape((-1, 1))
        Y0 = Y[Y - np.min(Y) < 0.25].reshape((-1, 1))
        try:
            inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
            mean_y1 = np.mean(Y0[inlier_mask1])
            idx1 = np.where(abs(Y0 - mean_y1) < 0.08)[0]  # 0.08
            x1 = X0[idx1, :]
            y1 = Y0[idx1, :]
            X1 = np.delete(X0, idx1).reshape((-1, 1))
            Y1 = np.delete(Y0, idx1).reshape((-1, 1))
        except:
            print("第一次拟合失败")
            return xc, yc, w, h
        try:
            inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X1, Y1, th)
            mean_y2 = np.mean(Y1[inlier_mask2])
            idx2 = np.where(abs(Y1 - mean_y2) < 0.08)[0]  # 0.08
            x2 = X1[idx2, :]
            y2 = Y1[idx2, :]
        except:
            print("第二次拟合错误，相机视角有问题或被阻挡")
            left_down_conner_y = np.min(Y1)
            right_up_conner_y = np.max(Y1)
            height_from_left_conner = mean_y1 - left_down_conner_y
            height_to_right_conner = right_up_conner_y - mean_y1
            # -----------------------#
            #      %>>>>>>          #
            #      ?                #
            #      ?                #
            #      ?                #
            # -----------------------#
            if height_from_left_conner > 0.05 and np.min(x1) > -0.05:
                xc = np.min(x1)
                h = height_from_left_conner
                yc = np.mean(y1)
                w = 0
            # -----------------------#
            #           %            #
            #           ?            #
            #   ?>>>>>>>?            #
            #   ?                    #
            # -----------------------#
            elif height_to_right_conner > 0.05:
                xc = np.max(x1)
                h = height_to_right_conner
                yc = np.max(Y1)
                w = 0
            else:
                xc = 0
                yc = 0
                h = 0
            return xc, yc, w, h
        if len(x1) * len(x2) == 0:
            return xc, yc, w, h
        if mean_y1 > mean_y2:
            stair_high_x, stair_high_y = x1, y1
            stair_low_x, stair_low_y = x2, y2
        else:
            stair_high_x, stair_high_y = x2, y2
            stair_low_x, stair_low_y = x1, y1
        w = np.max([np.max(stair_high_x) - np.max(stair_low_x),
                    np.min(stair_high_x) - np.min(stair_low_x)])
        h = np.mean(stair_high_y) - np.mean(stair_low_y)
        if np.mean(stair_low_y) - np.min(Y1) > 0.05 and np.min(stair_low_x) > -0.05:
            xc = np.min(stair_low_x)
            yc = np.mean(stair_low_y)
        else:
            xc = np.min(stair_high_x)
            yc = np.mean(stair_high_y)
        if w > 0.35 and np.max(stair_low_x) - np.min(stair_low_x) > 0.35:
            print("第一节台阶")
            w = np.max([np.max(stair_high_x) - np.max(stair_low_x),
                        np.max(stair_high_x) - np.min(stair_high_x)])
        elif w > 0.35 and np.max(stair_high_x) - np.min(stair_high_x) > 0.35:
            print("最后一个台阶")
            w = np.max([np.min(stair_high_x) - np.min(stair_low_x),
                        np.max(stair_low_x) - np.min(stair_low_x)])
        return xc, yc, w, h
