import matplotlib.pyplot as plt
import numpy as np
from Utils.Algo import *
import cv2
import sys


class pcd_opreator_system(object):
    def __init__(self, pcd_new):
        self.Acenter = np.zeros((0, 2))
        self.Bcenter = np.zeros((0, 2))
        self.Ccenter = np.zeros((0, 2))
        self.fea_A = np.zeros((0, 2))
        self.fea_B = np.zeros((0, 2))
        self.fea_C = np.zeros((0, 2))
        self.is_fea_A_gotten = False  # 是否提取到A
        self.is_fea_B_gotten = False  # 是否提取到B
        self.is_fea_C_gotten = False  # 是否提取到C
        self.corner_situation = 0
        self.pcd_new = pcd_new
        self.num_line = 0
        self.fea_extra_over = False

    def get_fea(self, env_type, _print_=False, ax=None, idx=0):
        if env_type == 1:
            self.get_fea_sa(_print_=_print_, ax=ax, idx = idx)
        self.fea_extra_over = True

    def get_fea_sa(self, _print_=False, ax=None, idx = 0):
        line1_success = False
        x1, y1, idx1 = [], [], []
        mean_line1 = 0
        x1, y1, mean_line1, idx1, line1_success = self.ransac_process_1(th_ransac_k=0.1, th_length=0.1, th_interval=0.1,
                                                                        _print_=_print_)
        if line1_success:
            self.num_line = 1
        else:
            self.num_line = 0
            return

        if idx == 45:
            stop = 1
        line2_success = False
        x2, y2, idx2 = [], [], []
        mean_line2 = 0
        if line1_success:
            x2, y2, mean_line2, idx2, line2_success = self.ransac_process_2(idx1, th_ransac_k=0.12, th_length=0.05,
                                                                            th_interval=0.05, _print_=_print_)
            if line2_success:
                self.num_line = 2
            else:
                self.num_line = 1

        self.need_to_check_B = True
        self.need_to_check_A = False
        self.need_to_check_C = False

        if line1_success and line2_success:
            if mean_line1 > mean_line2:
                # line1 is higher than line2
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x2, y2, idx2
            else:
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x2, y2, idx2
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.rebundacy_check_under_left(self.stair_low_x, self.stair_low_y, ax=ax)
            self.rebundacy_check_up_right(self.stair_high_x, self.stair_high_y, ax=ax)
            self.classify_corner_situation(num_stair=2, _print_=_print_)
            self.get_fea_A(_print_=_print_)
            self.get_fea_B(_print_=_print_)
            self.get_fea_C(_print_=_print_)

        if line1_success and not line2_success:
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
            self.rebundacy_check_under_left(self.stair_low_x, self.stair_low_y, ax=ax)
            self.rebundacy_check_up_right(self.stair_low_x, self.stair_low_y, ax=ax)
            self.classify_corner_situation(num_stair=1, _print_=_print_)
            self.get_fea_A(_print_=_print_)
            self.get_fea_B(_print_=_print_)
            self.get_fea_C(_print_=_print_)

    def get_fea_A(self, _print_=False):
        if self.need_to_check_A:
            if _print_:
                print('A feature finding')
            Acenter_x = np.min(self.stair_low_x)
            Acenter_y = self.stair_low_y[np.argmin(self.stair_low_x)][0]
            idx_fea_A = np.logical_and(np.abs(self.pcd_new[:, 0] - Acenter_x) < 0.05,
                                       np.abs(self.pcd_new[:, 1] - Acenter_y) < 0.05)
            if np.shape(idx_fea_A)[0] > 10:
                fea_Ax_new = self.pcd_new[idx_fea_A, 0].reshape((-1, 1))
                fea_Ay_new = self.pcd_new[idx_fea_A, 1].reshape((-1, 1))
                if _print_:
                    print("find feature A:{},{}".format(Acenter_x, Acenter_y))
            else:
                mean_Ax = np.nanmean(self.stair_low_x[idx_fea_A])
                mean_Ay = np.nanmean(self.stair_low_y[idx_fea_A])
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Ax_new = mean_Ax + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Ay_new = mean_Ay + rand * 0.001
                if _print_:
                    print("complete feature A:{},{}".format(Acenter_x, Acenter_y))
            self.Acenter = np.array([Acenter_x, Acenter_y])
            self.fea_A = np.hstack([fea_Ax_new, fea_Ay_new])
            self.is_fea_A_gotten = True
        else:
            self.is_fea_A_gotten = False

    def get_fea_B(self, _print_=False):
        if self.need_to_check_B:
            if _print_:
                print('B feature finding')
            Bcenter_x = stair_low_right_corner_x = np.max(self.stair_low_x)
            Bcenter_y = stair_low_right_corner_y = self.stair_low_y[np.argmax(self.stair_low_x)][0]
            idx_fea_B = np.logical_and(np.abs(self.pcd_new[:, 0] - stair_low_right_corner_x) < 0.05,
                                       np.abs(self.pcd_new[:, 1] - stair_low_right_corner_y) < 0.05)
            if np.shape(idx_fea_B)[0] > 10:
                fea_Bx_new = self.pcd_new[idx_fea_B, 0].reshape((-1, 1))
                fea_By_new = self.pcd_new[idx_fea_B, 1].reshape((-1, 1))
                if _print_:
                    print("find feature B:{},{}".format(Bcenter_x, Bcenter_y))
            else:
                mean_Bx = np.nanmean(self.stair_low_x[idx_fea_B])
                mean_By = np.nanmean(self.stair_low_y[idx_fea_B])
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Bx_new = mean_Bx + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_By_new = mean_By + rand * 0.001
                if _print_:
                    print("complete feature B:{},{}".format(Bcenter_x, Bcenter_y))
            self.Bcenter = np.array([Bcenter_x, Bcenter_y])
            self.fea_B = np.hstack([fea_Bx_new, fea_By_new])
            self.is_fea_B_gotten = True
        else:
            self.is_fea_B_gotten = False

    def get_fea_C(self, _print_=False):
        if _print_:
            print('C feature finding')
        if self.need_to_check_C:
            if self.has_part_right_line:
                print("Feature C finding in Condition1")
                X_right_part = self.pcd_right_up_part[:, 0]
                Y_right_part = self.pcd_right_up_part[:, 1]
                Ccenter_x = np.min(X_right_part)
                Ccenter_y = Y_right_part[np.argmin(X_right_part)]
                idx_fea_C = np.logical_and(np.abs(self.pcd_new[:, 0] - Ccenter_x) < 0.05,
                                           np.abs(self.pcd_new[:, 1] - Ccenter_y) < 0.05)
                if np.shape(idx_fea_C)[0] > 10:
                    fea_Cx_new = self.pcd_new[idx_fea_C, 0].reshape((-1, 1))
                    fea_Cy_new = self.pcd_new[idx_fea_C, 1].reshape((-1, 1))
                    if _print_:
                        print("find feature C:{},{}".format(Ccenter_x, Ccenter_y))
                else:
                    mean_Cx = np.nanmean(X_right_part[idx_fea_C])
                    mean_Cy = np.nanmean(Y_right_part[idx_fea_C])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Cx_new = mean_Cx + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Cy_new = mean_Cy + rand * 0.001
                    if _print_:
                        print("complete feature C:{},{}".format(Ccenter_x, Ccenter_y))
            else:
                print("Feature C finding in Condition2")
                Ccenter_x = stair_high_left_corner_x = np.min(self.stair_high_x)
                Ccenter_y = stair_high_left_corner_y = self.stair_high_y[np.argmin(self.stair_high_x)][0]
                idx_fea_C = np.logical_and(np.abs(self.pcd_new[:, 0] - stair_high_left_corner_x) < 0.05,
                                           np.abs(self.pcd_new[:, 1] - stair_high_left_corner_y) < 0.05)
                if np.shape(idx_fea_C)[0] > 10:
                    fea_Cx_new = self.pcd_new[idx_fea_C, 0].reshape((-1, 1))
                    fea_Cy_new = self.pcd_new[idx_fea_C, 1].reshape((-1, 1))
                    if _print_:
                        print("find feature C:{},{}".format(Ccenter_x, Ccenter_y))
                else:
                    mean_Cx = np.nanmean(self.stair_high_x[idx_fea_C])
                    mean_Cy = np.nanmean(self.stair_high_y[idx_fea_C])
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Cx_new = mean_Cx + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Cy_new = mean_Cy + rand * 0.001
                    if _print_:
                        print("complete feature C:{},{}".format(Ccenter_x, Ccenter_y))
            self.Ccenter = np.array([Ccenter_x, Ccenter_y])
            self.fea_C = np.hstack([fea_Cx_new, fea_Cy_new])
            self.is_fea_C_gotten = True
        else:
            self.is_fea_C_gotten = False

    def classify_corner_situation(self, num_stair, _print_=False):
        if num_stair == 2:
            if self.has_part_under_line:
                if self.has_part_up_line:
                    self.has_part_right_line = False  # 不考虑最右侧第三个台阶的C角点
                    self.corner_situation = 7
                    self.need_to_check_A = True
                    self.need_to_check_B = True
                    self.need_to_check_C = True
                elif not self.has_part_up_line:
                    self.corner_situation = 3
                    self.need_to_check_A = True
                    self.need_to_check_B = True
                    self.need_to_check_C = True
            elif not self.has_part_under_line:
                if self.has_part_up_line:
                    self.has_part_right_line = False  # 不考虑最右侧第三个台阶的C角点
                    self.corner_situation = 6
                    self.need_to_check_B = True
                    self.need_to_check_C = True
                elif not self.has_part_up_line:
                    self.corner_situation = 4
                    self.need_to_check_B = True
                    self.need_to_check_C = True

        else:
            if self.has_part_under_line and self.has_part_left_line:
                if self.has_part_up_line and self.has_part_right_line:
                    self.corner_situation = 3
                    self.need_to_check_A = True
                    self.need_to_check_C = True
                    self.need_to_check_B = True
                elif self.has_part_up_line and not self.has_part_right_line:
                    self.corner_situation = 1
                    self.need_to_check_A = True
                    self.need_to_check_B = True
                elif not self.has_part_up_line:
                    self.corner_situation = 5
                    self.need_to_check_C = True
                    self.need_to_check_B = False
            elif self.has_part_under_line and not self.has_part_left_line:
                if self.has_part_up_line and self.has_part_right_line:
                    self.corner_situation = 3
                    self.need_to_check_A = True
                    self.need_to_check_B = True
                    self.need_to_check_C = True
                elif self.has_part_up_line and not self.has_part_right_line:
                    self.corner_situation = 1
                    self.need_to_check_A = True
                    self.need_to_check_B = True
                elif not self.has_part_up_line:
                    self.corner_situation = 5
                    self.need_to_check_C = True
                    self.need_to_check_B = False
            elif not self.has_part_under_line:
                if self.has_part_up_line and self.has_part_right_line:
                    self.corner_situation = 4
                    self.need_to_check_C = True
                    self.need_to_check_B = True
                elif self.has_part_up_line and not self.has_part_right_line:
                    self.corner_situation = 2
                    self.need_to_check_B = True
        if _print_:
            print("corner_situation:{}".format(self.corner_situation))

    def rebundacy_check_under_left(self, stair_low_x, stair_low_y, ax=None):
        ymin = np.min(self.pcd_new[:, 1])
        xmin = np.min(self.pcd_new[:, 0])
        self.has_part_under_line = False
        print("Under_line:{}".format(np.nanmean(stair_low_y) - ymin))
        if np.nanmean(stair_low_y) - ymin > 0.02:
            self.has_part_under_line = True
        else:
            self.has_part_under_line = False

        self.has_part_left_line = False
        print("Left_line:{}".format(np.min(stair_low_x) - xmin))
        if self.has_part_under_line and np.min(stair_low_x) - xmin > 0.03:
            left_min_y = self.pcd_new[np.argmin(self.pcd_new[:, 0]), 1]
            if np.nanmean(stair_low_y) - left_min_y > 0.015:
                idx_under_left = np.where(np.abs(self.pcd_new[:, 1] - left_min_y) < 0.015)[0]
                under_left_part = self.pcd_new[idx_under_left, :]
                if ax is not None:
                    ax.scatter(under_left_part[:, 0], under_left_part[:, 1], color='c', linewidths=6)
                diff_under_left = np.diff(under_left_part, axis=0)
                idx_continuous = np.where(np.abs(diff_under_left) < 0.005)[0]
                left_part_continuous = under_left_part[idx_continuous]
                if np.shape(left_part_continuous)[0] > 10 and np.min(stair_low_x) - np.min(
                        left_part_continuous) > 0.03:
                    self.has_part_left_line = True
        else:
            self.has_part_left_line = False

    def rebundacy_check_up_right(self, stair_high_x, stair_high_y, ax=None):
        ymax = np.max(self.pcd_new[:, 1])
        xmax = np.max(self.pcd_new[:, 0])
        self.has_part_up_line = False
        print("Up_line:{}".format(ymax - np.nanmean(stair_high_y)))
        if ymax - np.nanmean(stair_high_y) > 0.02:
            self.has_part_up_line = True
            print("Up_line_Part find")
        else:
            self.has_part_up_line = False

        self.has_part_right_line = False
        print("Right_line:{}".format(xmax - np.max(stair_high_x)))
        if self.has_part_up_line and xmax - np.max(stair_high_x) > 0.04:
            right_max_y = self.pcd_new[np.argmax(self.pcd_new[:, 0]), 1]
            if right_max_y - np.nanmean(stair_high_y) > 0.02:
                idx_up_right = np.where(np.abs(self.pcd_new[:, 1] - right_max_y) < 0.015)[0]
                up_right_part = self.pcd_new[idx_up_right, :]
                if ax is not None:
                    ax.scatter(up_right_part[:, 0], up_right_part[:, 1], color='c', linewidths=6)
                diff_up_right = np.diff(up_right_part, axis=0)
                idx_continuous = np.where(np.abs(diff_up_right) < 0.005)[0]
                right_part_continuous = up_right_part[idx_continuous]
                if np.shape(right_part_continuous)[0] > 10 and np.max(right_part_continuous) - np.max(
                        stair_high_x) > 0.03:
                    self.has_part_right_line = True
                    # 由于步骤比较繁琐这里直接存储right_part_continuous
                    self.pcd_right_up_part = right_part_continuous
        else:
            self.has_part_right_line = False

        # todo: 把ransac_process_1和ransac_process_2整合为一个ransac_process

    def ransac_process_1(self, th_ransac_k=0.1, th_length=0.1, th_interval=0.1, _print_=False):
        x1 = []
        y1 = []
        mean_line1 = 0
        idx1 = []
        line1_success = False
        X0 = self.pcd_new[:, 0].reshape((-1, 1))
        Y0 = self.pcd_new[:, 1].reshape((-1, 1))
        idx_x1_in_X0 = np.zeros((0,))
        idx_X0_in_X0 = np.arange(np.shape(X0)[0])
        try:
            inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th_ransac_k)
            mean_line1 = np.nanmean(Y0[inlier_mask1])
            idx_x1_in_X0 = np.where(abs(Y0 - mean_line1) < 0.01)[0]
            x1 = X0[idx_x1_in_X0, :]
            y1 = Y0[idx_x1_in_X0, :]
            line1_length = np.max(x1) - np.min(x1)
            diff_x1 = np.diff(x1, axis=0)
            # 条件：
            # 1.直线点数>20
            # 2.直线长度>th_length
            # 3.点和点之间水平距离<th_interval
            if np.shape(idx_x1_in_X0)[0] > 20 and line1_length >= th_length and np.max(np.abs(diff_x1)) < th_interval:
                line1_success = True
                if _print_:
                    print("Line1 get")
            # 如果直线过短重新提取一次
            elif np.shape(idx_x1_in_X0)[0] > 20 and line1_length < th_length and np.max(np.abs(diff_x1)) < th_interval:
                if _print_:
                    print("Line1 Try RANSAC Again")
                X_temp = np.delete(X0, idx_x1_in_X0).reshape((-1, 1))
                Y_temp = np.delete(Y0, idx_x1_in_X0).reshape((-1, 1))
                idx_Xtemp_in_X0 = np.delete(idx_X0_in_X0, idx_x1_in_X0)
                inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X_temp, Y_temp, th_ransac_k)
                mean_line1 = mean_line_temp = np.nanmean(Y_temp[inlier_mask1])
                idx_x1_in_Xtemp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
                idx_x1_in_X0 = idx_Xtemp_in_X0[idx_x1_in_Xtemp]
                x1 = X0[idx_x1_in_X0, :]
                y1 = Y0[idx_x1_in_X0, :]
                line1_length = np.max(x1) - np.min(x1)
                diff_x1 = np.diff(x1, axis=0)
                if np.shape(idx_x1_in_X0)[0] > 20 and line1_length > th_length and np.max(
                        np.abs(diff_x1)) < th_interval:
                    line1_success = True
                    if _print_:
                        print("Line1 get")
                else:
                    line1_success = False
                    if _print_:
                        print("Not get Line1")
                        print(
                            "line1_length:{}<{}".format(line1_length, th_length) + "diff_x1:{}>{}".format(
                                np.max(np.abs(diff_x1)), th_interval))

        except Exception as e:
            line1_success = False
            if _print_:
                print(e)
                print("Line1 RANSAC False, No Line in the picture")
        return x1, y1, mean_line1, idx_x1_in_X0, line1_success

    def ransac_process_2(self, idx1, th_ransac_k=0.1, th_length=0.1, th_interval=0.1, _print_=False):
        x2 = []
        y2 = []
        mean_line2 = 0
        idx2 = []
        line2_success = False
        X0 = self.pcd_new[:, 0].reshape((-1, 1))
        Y0 = self.pcd_new[:, 1].reshape((-1, 1))
        idx_all = np.arange(np.shape(X0)[0])
        X1 = np.delete(X0, idx1).reshape((-1, 1))
        Y1 = np.delete(Y0, idx1).reshape((-1, 1))
        idx_x1_in_X0 = idx1
        idx_X1_in_X0 = np.delete(idx_all, idx_x1_in_X0)
        idx_x2_in_X0 = np.zeros((0,))
        try:
            inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X1, Y1, th_ransac_k)
            mean_line2 = np.nanmean(Y1[inlier_mask2])
            idx_x2_in_X1 = np.where(abs(Y1 - mean_line2) < 0.01)[0]
            idx_x2_in_X0 = idx_X1_in_X0[idx_x2_in_X1]
            x2 = X0[idx_x2_in_X0, :]
            y2 = Y0[idx_x2_in_X0, :]
            line2_length = np.max(x2) - np.min(x2)
            diff_x2 = np.diff(x2, axis=0)
            if np.shape(idx_x2_in_X0)[0] > 20 and line2_length >= th_length and np.max(abs(diff_x2)) < th_interval:
                line2_success = True
                if _print_:
                    print("Line2 get")
            elif np.shape(idx_x2_in_X0)[0] > 20 and np.max(abs(diff_x2)) < th_interval and line2_length < th_length:
                if _print_:
                    print("Line2 RANSAC Try again")
                X_temp = np.delete(X1, idx_x2_in_X1).reshape((-1, 1))
                Y_temp = np.delete(Y1, idx_x2_in_X1).reshape((-1, 1))
                idx_Xtemp_in_X0 = np.delete(idx_X1_in_X0, idx_x2_in_X0)
                inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X_temp, Y_temp, th_ransac_k)
                mean_line2 = mean_line_temp = np.nanmean(Y_temp[inlier_mask2])
                idx_x2_in_Xtemp = np.where(abs(Y_temp - mean_line_temp) < 0.01)[0]
                idx_x2_in_X0 = idx_Xtemp_in_X0[idx_x2_in_Xtemp]
                x2 = X0[idx_x2_in_X0, :]
                y2 = Y0[idx_x2_in_X0, :]
                line2_length = np.max(x2) - np.min(x2)
                diff_x2 = np.diff(x2, axis=0)
                if np.shape(idx_x2_in_X0)[0] > 0 and line2_length > 0.05 > np.max(np.abs(diff_x2)):
                    line2_success = True
                    if _print_:
                        print("Line2 get")
                else:
                    line2_success = False
                    if _print_:
                        print("Not get Line2")
                        print(
                            "line2_length:{}<{}".format(line2_length, th_length) + "diff_x2:{}>{}".format(
                                np.max(np.abs(diff_x2)), th_interval))
        except Exception as e:
            line2_success = False
            if _print_:
                print(e)
                print("Line2 RANSAC False")
        return x2, y2, mean_line2, idx_x2_in_X0, line2_success

    def show_(self, ax, pcd_color='r', id=0, p_text=0.1, p_pcd=None):
        if p_pcd is None:
            p_pcd = [0, 0]
        if self.fea_extra_over:
            ax.scatter(self.pcd_new[0:-1:10, 0]+p_pcd[0], self.pcd_new[0:-1:10, 1]+p_pcd[1], color=pcd_color, alpha=0.1)
            if self.num_line == 2:
                ax.plot(self.stair_low_x+p_pcd[0], self.stair_low_y+p_pcd[1], color='black', linewidth=2)
                ax.plot(self.stair_high_x+p_pcd[0], self.stair_high_y+p_pcd[1], color='black', linewidth=2)
            elif self.num_line == 1:
                ax.plot(self.stair_low_x+p_pcd[0], self.stair_low_y+p_pcd[1], color='black', linewidth=2)
            if self.is_fea_A_gotten:
                ax.scatter(self.fea_A[0:-1:5, 0]+p_pcd[0], self.fea_A[0:-1:5, 1]+p_pcd[1], color='m', linewidths=1)
            if self.is_fea_B_gotten:
                ax.scatter(self.fea_B[0:-1:5, 0]+p_pcd[0], self.fea_B[0:-1:5, 1]+p_pcd[1], color='g', linewidths=1)
            if self.is_fea_C_gotten:
                ax.scatter(self.fea_C[0:-1:5, 0]+p_pcd[0], self.fea_C[0:-1:5, 1]+p_pcd[1], color='y', linewidths=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 0)
            plt.text(p_text, -0.1, 'id: {}'.format(id))
            plt.text(p_text, -0.2, 'corner: {}'.format(self.corner_situation))

    def fea_to_env_paras(self):
        xc, yc, w, h = 0, 0, 0, 0
        if not self.fea_extra_over:
            print("Please Extract Feature First")
        if np.max(self.pcd_new[:, 1]) - np.min(self.pcd_new[:, 1]) < 0.1:
            print("点云高度差过小，近似为平地")
            xc, yc, w, h = 0, 0, 0, 0
            return xc, yc, w, h
        if self.num_line == 0:
            print("Line1 拟合失败，无法检测到直线")
            xc, yc, w, h = 0, 0, 0, 0
            return xc, yc, w, h
        if self.num_line == 1:
            print("Line2 拟合失败，无法检测第二条直线")
            X0 = self.pcd_new[:, 0]
            Y0 = self.pcd_new[:, 1]
            left_down_conner_y = np.min(self.stair_low_y)
            right_up_conner_y = np.max(self.stair_high_y)
            mean_y = np.mean(self.stair_low_y)
            height_B0_to_A1 = mean_y - left_down_conner_y
            height_B1_to_A2 = right_up_conner_y - mean_y
            if self.corner_situation == 3:
                xc = self.Bcenter[0]
                yc = self.Bcenter[1]
                h = height_B1_to_A2
                w = self.Bcenter[0] - self.Acenter[0]
            elif self.corner_situation == 6 or self.corner_situation == 5 or self.corner_situation == 4:
                xc = self.Ccenter[0]
                yc = self.Ccenter[1]
                h = height_B0_to_A1
                w = 0
            elif self.corner_situation == 1:
                xc = self.Acenter[0]
                yc = self.Acenter[1]
                h = 0
                w = self.Bcenter[0] - self.Acenter[0]
            elif self.corner_situation == 2:
                xc, yc, w, h = 0, 0, 0, 0

            return xc, yc, w, h
        if self.num_line == 2:
            print("同时检测到两个台阶")
            w = np.max([np.max(self.stair_high_x) - np.max(self.stair_low_x),
                        np.min(self.stair_high_x) - np.min(self.stair_low_x)])
            h = np.mean(self.stair_high_y) - np.mean(self.stair_low_y)
            if self.corner_situation == 7 or self.corner_situation == 3:
                xc = self.Acenter[0]
                yc = self.Acenter[1]
            elif self.corner_situation == 6 or self.corner_situation == 4:
                xc = self.Ccenter[0]
                yc = self.Ccenter[1]
            return xc, yc, w, h
