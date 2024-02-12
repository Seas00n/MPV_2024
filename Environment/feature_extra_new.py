import math
import time

import matplotlib.pyplot as plt
import numpy as np
from Utils.Algo import *
import cv2
import sys
import datetime
from Utils.pcd_os_fast_plot import *


class pcd_opreator_system(object):
    def __init__(self, pcd_new):
        self.Acenter = np.zeros((0, 2))
        self.Bcenter = np.zeros((0, 2))
        self.Ccenter = np.zeros((0, 2))

        self.fea_A = np.zeros((0, 2))
        self.fea_B = np.zeros((0, 2))
        self.fea_C = np.zeros((0, 2))
        self.fea_D = np.zeros((0, 2))
        self.fea_E = np.zeros((0, 2))
        self.fea_F = np.zeros((0, 2))
        self.is_fea_A_gotten = False  # 是否提取到A
        self.is_fea_B_gotten = False  # 是否提取到B
        self.is_fea_C_gotten = False  # 是否提取到C
        self.is_fea_D_gotten = False
        self.is_fea_E_gotten = False
        self.is_fea_F_gotten = False
        self.corner_situation = 0
        self.pcd_new = pcd_new
        self.num_line = 0
        self.fea_extra_over = False
        self.env_type = 0
        self.env_rotate = 0
        self.cost_time = 0

    def get_fea(self, _print_=False, ax=None, idx=0, nn_class=0):
        # self.env_type = self.get_env_type_up_or_down(ax=ax, nn_class_prior=nn_class)
        t0 = datetime.datetime.now()
        self.env_type = 1
        if _print_:
            print("Env_Type:{}".format(self.env_type))
            plt.text(0, 0.1, "Env_Type:{}".format(self.env_type))
        if self.env_type == 1:
            self.get_fea_sa(_print_=_print_, ax=ax, idx=idx)
            self.fea_extra_over = True
        if self.env_type == 2:
            self.get_fea_sd(_print_=_print_, ax=ax, idx=idx)
            self.fea_extra_over = True
        if self.env_type == 3:
            self.get_fea_ob(_print_=_print_, ax=ax, idx=idx)
            self.fea_extra_over = True
        diff = datetime.datetime.now() - t0
        self.cost_time = diff.total_seconds()

    def get_env_type_up_or_down(self, ax=None, nn_class_prior=0):
        env_type = 0
        # TODO: 可以汇总非ransac的直线提取函数
        # TODO: 进行地面水平矫正的条件
        xmin = np.min(self.pcd_new[:, 0])
        xmax = np.max(self.pcd_new[:, 0])
        y_xmin = self.pcd_new[np.argmin(self.pcd_new[:, 0]), 1]
        y_xmax = self.pcd_new[np.argmax(self.pcd_new[:, 0]), 1]

        if nn_class_prior == 0:
            # 进行地面水平矫正
            line_theta = math.atan((y_xmax - y_xmin) / (xmax - xmin))
            if abs(line_theta) < 0.2:
                self.env_rotate += line_theta
                x = self.pcd_new[:, 0]
                y = self.pcd_new[:, 1]
                self.pcd_new[:, 0] = np.cos(line_theta) * x + np.sin(line_theta) * y
                self.pcd_new[:, 1] = -np.sin(line_theta) * x + np.cos(line_theta) * y

        ymax = np.max(self.pcd_new[:, 1])
        line_highest = self.pcd_new[np.where(np.abs(self.pcd_new[:, 1] - ymax) < 0.015)[0], :]
        y_line_hest = np.nanmean(line_highest[:, 1])
        xmin_line_hest = line_highest[np.argmin(line_highest[:, 0]), 0]
        xmax_line_hest = line_highest[np.argmax(line_highest[:, 0]), 0]
        if ax is not None:
            ax.scatter(xmin_line_hest, y_line_hest, color='c', linewidths=4)
            ax.scatter(xmax_line_hest, y_line_hest, color='b', linewidths=4)
        idx_right_part = np.where(np.logical_and(self.pcd_new[:, 0] > xmax_line_hest + 0.02,
                                                 self.pcd_new[:, 1] < y_line_hest - 0.03))[0]
        idx_left_part = np.where(np.logical_and(self.pcd_new[:, 0] < xmin_line_hest - 0.02,
                                                self.pcd_new[:, 1] < y_line_hest - 0.03))[0]
        check_right = True
        check_left = True
        if np.shape(idx_right_part)[0] == 0:
            check_right = False
        if np.shape(idx_left_part)[0] == 0:
            check_left = False
        if not check_right and not check_left:
            env_type = 0
            return env_type

        elif not check_left and check_right:
            ymin_right = np.min(self.pcd_new[idx_right_part, 1])
            line_lowest = self.pcd_new[np.where(np.abs(self.pcd_new[:, 1] - ymin_right) < 0.015)[0], :]
            y_line_lest = np.nanmean(line_lowest[:, 1])
            x_line_lest = line_lowest[np.argmax(line_lowest[:, 0]), 0]
            if ax is not None:
                print("not check Left and check Right")
                ax.scatter(x_line_lest, y_line_lest, color='g', linewidths=4)
            if abs(y_line_hest - y_line_lest) < 0.02:
                env_type = 0
                return env_type
            elif y_line_hest - y_line_lest > 0.015 and x_line_lest - xmax_line_hest > 0.02:
                env_type = 2
                return env_type
            else:
                return 0
        elif not check_right and check_left:
            ymin_left = np.min(self.pcd_new[idx_left_part, 1])
            line_lowest = self.pcd_new[np.where(np.abs(self.pcd_new[:, 1] - ymin_left) < 0.015)[0], :]
            y_line_lest = np.nanmean(line_lowest[:, 1])
            x_line_lest = line_lowest[np.argmin(line_lowest[:, 0]), 0]
            if abs(y_line_hest - y_line_lest) < 0.02:
                env_type = 0
                return env_type
            elif y_line_hest - y_line_lest > 0.02 and xmin_line_hest - x_line_lest > 0:
                env_type = 1
                if ax is not None:
                    print("not check Right and check Left")
                    ax.scatter(x_line_lest, y_line_lest, color='y', linewidths=4)
                return env_type
            else:
                return 0
        elif check_right and check_left:
            line_lowest_right = self.pcd_new[idx_right_part, :]
            ymin_right = np.min(line_lowest_right[:, 1])
            line_lowest_right = line_lowest_right[np.where(np.abs(line_lowest_right[:, 1] - ymin_right) < 0.015)[0], :]
            y_right_lest = np.nanmean(line_lowest_right[:, 1])
            x_right_lest = line_lowest_right[np.argmax(line_lowest_right[:, 0]), 0]

            line_lowest_left = self.pcd_new[idx_left_part, :]
            ymin_left = np.min(line_lowest_left[:, 1])
            line_lowest_left = line_lowest_left[np.where(np.abs(line_lowest_left[:, 1] - ymin_left) < 0.015)[0], :]
            y_left_lest = np.nanmean(line_lowest_left[:, 1])
            x_left_lest = line_lowest_left[np.argmin(line_lowest_left[:, 0]), 0]
            if ax is not None:
                print("check Right and check Left")
                ax.plot(line_lowest_right[:, 0], line_lowest_right[:, 1], color='black', linewidth=2)
                ax.plot(line_lowest_left[:, 0], line_lowest_left[:, 1], color='black', linewidth=2)
                ax.scatter(x_right_lest, y_right_lest, color='g', linewidths=4)
                ax.scatter(x_left_lest, y_left_lest, color='y', linewidths=4)
                print(y_line_hest - y_left_lest)
                print(y_line_hest - y_right_lest)
            if y_line_hest - y_left_lest > 0.02 and y_line_hest - y_right_lest > 0.02:
                if xmin_line_hest - x_left_lest > 0.02 and x_right_lest - xmax_line_hest > 0.02:
                    env_type = 3
                    # 在此处提取
                    self.obs_high_level_x = line_highest[:, 0]
                    self.obs_high_mean_y = y_line_hest
                    self.obs_low_x = xmin_line_hest
                    self.obs_low_y = y_left_lest
                    return env_type
            elif abs(y_line_hest - y_right_lest) < 0.03 and y_line_hest - ymin_left > 0.08:
                if xmin_line_hest - x_left_lest > 0.1:
                    env_type = 1
                    return env_type
            elif abs(y_line_hest - y_left_lest) < 0.03 and y_line_hest - ymin_right > 0.08:
                if x_right_lest - xmax_line_hest > 0.1:
                    env_type = 2
                    return env_type
            elif abs(y_line_hest - y_left_lest) < 0.02 and abs(y_line_hest - y_right_lest) < 0.02:
                env_type = 0
                return env_type
        return 0

    def get_fea_sd(self, _print_=False, ax=None, idx=0):
        line1_success = False
        x1, y1, idx1 = [], [], []
        mean_line1 = 0
        # 需要调节th_length使得尽量有一条以上的直线
        x1, y1, mean_line1, idx1, line1_success = self.ransac_process_1(th_ransac_k=0.15, th_length=0.1,
                                                                        th_interval=0.1,
                                                                        _print_=_print_)

        if line1_success:
            self.num_line = 1
            x1, y1, mean_line1, idx1 = self.line_ground_calibrate(idx1)
        else:
            self.num_line = 0
            return
        # 对下楼梯希望从相对较长的主直线下方寻找第二条直线，所以扩大idx1的范围
        idx1_new = np.where(self.pcd_new[:, 1] > mean_line1 - 0.02)[0]

        line2_success = False
        x2, y2, idx2 = [], [], []
        mean_line2 = 0
        if line1_success:
            x2, y2, mean_line2, idx2, line2_success = self.ransac_process_2(idx1_new, th_ransac_k=0.12, th_length=0.1,
                                                                            th_interval=0.05, _print_=_print_)
            if line2_success:
                self.num_line = 2
            else:
                self.num_line = 1

        self.need_to_check_D = False
        self.need_to_check_E = False
        self.need_to_check_F = False

        if line1_success and line2_success:
            if mean_line1 > mean_line2:
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x2, y2, idx2
            else:
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x2, y2, idx2
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.rebundacy_check_up_left_sd(self.stair_high_x, self.stair_high_y, ax=ax)
            self.rebundacy_check_under_right_sd(self.stair_low_x, self.stair_low_y, ax=ax)
            self.classify_corner_situation_sd(num_stair=2, _print_=_print_)
            self.get_fea_D(_print_=_print_)
            self.get_fea_E(_print_=_print_)
            self.get_fea_F(_print_=_print_)

        elif line1_success and not line2_success:
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.rebundacy_check_up_left_sd(self.stair_high_x, self.stair_high_y, ax=ax)
            self.rebundacy_check_under_right_sd(self.stair_low_x, self.stair_low_y, ax=ax)
            self.classify_corner_situation_sd(num_stair=1, _print_=_print_)
            self.get_fea_D(_print_=_print_)
            self.get_fea_E(_print_=_print_)
            self.get_fea_F(_print_=_print_)

    def get_fea_ob(self, _print_=False, ax=None, idx=0):
        self.corner_situation = 8
        self.need_to_check_B = True
        self.need_to_check_C = True
        self.need_to_check_D = True

        if _print_:
            print("B feature finding")
        Bcenter_x = self.obs_low_x
        Bcenter_y = self.obs_low_y
        idx_fea_B = np.logical_and(np.abs(self.pcd_new[:, 0] - Bcenter_x) < 0.02,
                                   np.abs(self.pcd_new[:, 1] - Bcenter_y) < 0.02)
        if np.shape(self.pcd_new[idx_fea_B, 0])[0] > 10:
            fea_Bx_new = self.pcd_new[idx_fea_B, 0].reshape((-1, 1))
            fea_By_new = self.pcd_new[idx_fea_B, 1].reshape((-1, 1))
            if _print_:
                print("find feature B:{},{}".format(Bcenter_x, Bcenter_y))
        else:
            mean_Bx = Bcenter_x
            mean_By = Bcenter_y
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Bx_new = mean_Bx + rand * 0.001
            rand = np.random.rand(20).reshape((-1, 1))
            fea_By_new = mean_By + rand * 0.001
            if _print_:
                print("complete feature B:{},{}".format(Bcenter_x, Bcenter_y))
        self.Bcenter = np.array([Bcenter_x, Bcenter_y])
        self.fea_B = np.hstack([fea_Bx_new, fea_By_new])
        self.is_fea_B_gotten = True

        if _print_:
            print("C feature finding")
        Ccenter_x = np.min(self.obs_high_level_x)
        Ccenter_y = self.obs_high_mean_y
        idx_fea_C = np.logical_and(np.abs(self.pcd_new[:, 0] - Ccenter_x) < 0.02,
                                   np.abs(self.pcd_new[:, 1] - Ccenter_y) < 0.02)

        if np.shape(self.pcd_new[idx_fea_C, 0])[0] > 10:
            fea_Cx_new = self.pcd_new[idx_fea_C, 0].reshape((-1, 1))
            fea_Cy_new = self.pcd_new[idx_fea_C, 1].reshape((-1, 1))
            if _print_:
                print("find feature C:{},{}".format(Ccenter_x, Ccenter_y))
        else:
            mean_Cx = Ccenter_x
            mean_Cy = Ccenter_y
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Cx_new = mean_Cx + rand * 0.001
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Cy_new = mean_Cy + rand * 0.001
            if _print_:
                print("complete feature C:{},{}".format(Ccenter_x, Ccenter_y))
        self.Ccenter = np.array([Ccenter_x, Ccenter_y])
        self.fea_C = np.hstack([fea_Cx_new, fea_Cy_new])
        self.is_fea_C_gotten = True

        if _print_:
            print("D feature finding")
        Dcenter_x = np.max(self.obs_high_level_x)
        Dcenter_y = self.obs_high_mean_y
        idx_fea_D = np.logical_and(np.abs(self.pcd_new[:, 0] - Dcenter_x) < 0.02,
                                   np.abs(self.pcd_new[:, 1] - Dcenter_y) < 0.02)
        if np.shape(self.pcd_new[idx_fea_D, 0])[0] > 10:
            fea_Dx_new = self.pcd_new[idx_fea_D, 0].reshape((-1, 1))
            fea_Dy_new = self.pcd_new[idx_fea_D, 1].reshape((-1, 1))
            if _print_:
                print("find feature D:{},{}".format(Dcenter_x, Dcenter_y))
        else:
            mean_Dx = Dcenter_x
            mean_Dy = Dcenter_y
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Dx_new = mean_Dx + rand * 0.001
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Dy_new = mean_Dy + rand * 0.001
            if _print_:
                print("complete feature C:{},{}".format(Dcenter_x, Dcenter_y))
        self.Dcenter = np.array([Dcenter_x, Dcenter_y])
        self.fea_D = np.hstack([fea_Dx_new, fea_Dy_new])
        self.is_fea_D_gotten = True

    def get_fea_sa(self, _print_=False, ax=None, idx=0):
        line1_success = False
        x1, y1, idx1 = [], [], []
        mean_line1 = 0
        x1, y1, mean_line1, idx1, line1_success = self.ransac_process_1(th_ransac_k=0.15, th_length=0.05,
                                                                        th_interval=0.15, _print_=_print_)
        if line1_success:
            self.num_line = 1
            # x1, y1, mean_line1, idx1 = self.line_ground_calibrate(idx1)
        else:
            self.num_line = 0
            return

        line2_success = False
        x2, y2, idx2 = [], [], []
        mean_line2 = 0
        if line1_success:
            x2, y2, mean_line2, idx2, line2_success = self.ransac_process_2(idx1, th_ransac_k=0.12, th_length=0.02,
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
            self.rebundacy_check_under_left_sa(self.stair_low_x, self.stair_low_y, ax=ax)
            self.rebundacy_check_up_right_sa(self.stair_high_x, self.stair_high_y, ax=ax)
            self.classify_corner_situation_sa(num_stair=2, _print_=_print_)
            self.get_fea_A(_print_=_print_)
            self.get_fea_B(_print_=_print_)
            self.get_fea_C(_print_=_print_)

        if line1_success and not line2_success:
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
            self.rebundacy_check_under_left_sa(self.stair_low_x, self.stair_low_y, ax=ax)
            self.rebundacy_check_up_right_sa(self.stair_low_x, self.stair_low_y, ax=ax)
            self.classify_corner_situation_sa(num_stair=1, _print_=_print_)
            self.get_fea_A(_print_=_print_)
            self.get_fea_B(_print_=_print_)
            self.get_fea_C(_print_=_print_)

    def classify_corner_situation_sa(self, num_stair, _print_=False):
        if num_stair == 2:
            if self.has_part_under_line:
                if self.has_part_up_line:
                    self.has_part_right_line = False  # 不考虑最右侧第三个台阶的C角点
                    self.corner_situation = 3
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
                    self.corner_situation = 4
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

    def rebundacy_check_under_left_sa(self, stair_low_x, stair_low_y, ax=None):
        ymin = np.min(self.pcd_new[:, 1])
        xmin = np.min(self.pcd_new[:, 0])
        self.has_part_under_line = False
        print("Under_line:{}".format(np.nanmean(stair_low_y) - ymin))
        if np.nanmean(stair_low_y) - ymin > 0.02:
            idx_under = np.logical_and(self.pcd_new[:, 1] < np.nanmean(stair_low_y),
                                       np.abs(self.pcd_new[:, 0] - np.min(stair_low_x)) < 0.015)
            under_normal_part = self.pcd_new[idx_under, :]
            diff_under = np.diff(under_normal_part, axis=0)
            idx_continuous = np.where(np.abs(diff_under[:, 1]) < 0.005)[0]
            under_continuous_part = under_normal_part[idx_continuous, :]
            if np.shape(idx_continuous)[0] > 10 and np.nanmean(stair_low_y) - np.min(
                    under_continuous_part[:, 1]) > 0.005:
                self.has_part_under_line = True
        else:
            self.has_part_under_line = False

        self.has_part_left_line = False
        print("Left_line:{}".format(np.min(stair_low_x) - xmin))
        if self.has_part_under_line and np.min(stair_low_x) - xmin > 0.02:
            left_min_y = self.pcd_new[np.argmin(self.pcd_new[:, 0]), 1]
            if np.nanmean(stair_low_y) - left_min_y > 0.01:
                idx_under_left = np.where(np.abs(self.pcd_new[:, 1] - left_min_y) < 0.015)[0]
                under_left_part = self.pcd_new[idx_under_left, :]
                if ax is not None:
                    ax.scatter(under_left_part[:, 0], under_left_part[:, 1], color='c', linewidths=6)
                diff_under_left = np.diff(under_left_part, axis=0)
                idx_continuous = np.where(np.abs(diff_under_left[:, 0]) < 0.005)[0]
                left_part_continuous = under_left_part[idx_continuous]
                if np.shape(left_part_continuous)[0] > 10 and np.min(stair_low_x) - np.min(
                        left_part_continuous) > 0.02:
                    self.has_part_left_line = True
        else:
            self.has_part_left_line = False

    def rebundacy_check_up_right_sa(self, stair_high_x, stair_high_y, ax=None):
        ymax = np.max(self.pcd_new[:, 1])
        xmax = np.max(self.pcd_new[:, 0])
        self.has_part_up_line = False
        print("Up_line:{}".format(ymax - np.nanmean(stair_high_y)))
        if ymax - np.nanmean(stair_high_y) > 0.01:
            idx_up = np.logical_and(self.pcd_new[:, 1] > np.nanmean(stair_high_y),
                                    np.abs(self.pcd_new[:, 0] - np.max(stair_high_x)) < 0.015)
            up_normal_part = self.pcd_new[idx_up, :]
            diff_up = np.diff(up_normal_part, axis=0)
            idx_continuous = np.where(np.abs(diff_up[:, 1]) < 0.005)[0]
            up_continuous_part = up_normal_part[idx_continuous, :]
            if np.shape(idx_continuous)[0] > 10 and np.max(up_continuous_part[:, 1]) - np.nanmean(stair_high_y) > 0.005:
                self.has_part_up_line = True
        else:
            self.has_part_up_line = False

        self.has_part_right_line = False
        print("Right_line:{}".format(xmax - np.max(stair_high_x)))
        if self.has_part_up_line and xmax - np.max(stair_high_x) > 0.02:
            right_max_y = self.pcd_new[np.argmax(self.pcd_new[:, 0]), 1]
            if right_max_y - np.nanmean(stair_high_y) > 0.01:
                idx_up_right = np.where(np.abs(self.pcd_new[:, 1] - right_max_y) < 0.015)[0]
                up_right_part = self.pcd_new[idx_up_right, :]
                if ax is not None:
                    ax.scatter(up_right_part[:, 0], up_right_part[:, 1], color='c', linewidths=6)
                diff_up_right = np.diff(up_right_part, axis=0)
                idx_continuous = np.where(np.abs(diff_up_right[:, 0]) < 0.005)[0]
                right_part_continuous = up_right_part[idx_continuous]
                if np.shape(right_part_continuous)[0] > 10 and np.max(right_part_continuous) - np.max(stair_high_x) > 0.02:
                    self.has_part_right_line = True
                    # 由于步骤比较繁琐这里直接存储right_part_continuous
                    self.pcd_right_up_part_sa = right_part_continuous
        else:
            self.has_part_right_line = False

        # TODO: 把ransac_process_1和ransac_process_2整合为一个ransac_process

    def rebundacy_check_up_left_sd(self, stair_high_x, stair_high_y, ax=None):
        ymax = np.max(self.pcd_new[:, 1])
        self.has_part_up_line = False
        self.has_part_left_line = False
        if ymax - np.nanmean(stair_high_y) > 0.02:
            idx_up = np.where(np.abs(self.pcd_new[:, 1] - ymax) < 0.04)[0]
            up_part = self.pcd_new[idx_up, :]
            if ax is not None:
                ax.scatter(up_part[:, 0], up_part[:, 1], color='c', linewidths=6)
            diff_up = np.diff(up_part, axis=0)
            idx_continuous = np.where(np.abs(diff_up[:, 0]) < 0.005)[0]
            up_continuous_part = up_part[idx_continuous, :]
            if np.shape(up_continuous_part)[0] > 10 and np.max(up_continuous_part[:, 0]) - np.min(
                    up_continuous_part[:, 0]) > 0.01:
                self.has_part_up_line = True
                self.has_part_left_line = True
                self.pcd_left_up_part_sd = up_continuous_part

    def rebundacy_check_under_right_sd(self, stair_low_x, stair_low_y, ax=None):
        ymin = np.min(self.pcd_new[:, 1])
        self.has_part_under_line = False
        self.has_part_right_line = False
        if np.nanmean(stair_low_y) - ymin > 0.02:
            idx_under = np.where(self.pcd_new[:, 1] < np.nanmean(stair_low_y) - 0.02)[0]
            under_part = self.pcd_new[idx_under, :]
            if ax is not None:
                ax.scatter(under_part[:, 0], under_part[:, 1], color='c', linewidths=6)
            diff_under = np.diff(under_part, axis=0)
            idx_under_x_continuous = np.where(abs(diff_under[:, 0]) < 0.01)[0]
            under_x_continuous_part = diff_under[idx_under_x_continuous, :]
            if np.shape(under_x_continuous_part)[0] > 10 and np.max(under_x_continuous_part[:, 0]) - np.min(
                    under_x_continuous_part[:, 0]) > 0.01:
                self.has_part_under_line = True
                self.has_part_right_line = True

    def classify_corner_situation_sd(self, num_stair, _print_=False):
        if num_stair == 2:
            if self.has_part_up_line and self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    self.has_part_under_line, self.has_part_right_line = False, False  # 放弃从最下方提取特征
                    self.corner_situation = 10
                    self.need_to_check_D = True
                    self.need_to_check_E = True
                    self.need_to_check_F = True
                elif not self.has_part_under_line and not self.has_part_right_line:
                    self.corner_situation = 10
                    self.need_to_check_D = True
                    self.need_to_check_E = True
                    self.need_to_check_F = True
            elif not self.has_part_up_line and not self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    self.corner_situation = 10
                    self.need_to_check_D = True
                    self.need_to_check_E = True
                    self.need_to_check_F = True
                elif not self.has_part_under_line and not self.has_part_right_line:
                    self.corner_situation = 9
                    self.need_to_check_D = True
                    self.need_to_check_E = True
        elif num_stair == 1:
            if self.has_part_up_line and self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    self.corner_situation = 10
                    self.need_to_check_D = True
                    self.need_to_check_E = True
                    self.need_to_check_F = True
                elif not self.has_part_under_line and not self.has_part_right_line:
                    self.corner_situation = 9
                    self.need_to_check_D = True
                    self.need_to_check_E = True
            elif not self.has_part_up_line and not self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    self.corner_situation = 9
                    self.need_to_check_D = True
                    self.need_to_check_E = True
                elif not self.has_part_under_line and not self.has_part_right_line:
                    self.corner_situation = 0

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
        something_print = 0
        try:
            inlier_mask1, outlier_mask1, line_y_ransac1, line_X1, theta_line1 = RANSAC(X0, Y0, th_ransac_k)
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
                    something_print = 1
            # 如果直线过短重新提取一次
            elif np.shape(idx_x1_in_X0)[0] > 20 and line1_length < th_length and np.max(np.abs(diff_x1)) < th_interval:
                if _print_:
                    print("Line1 Try RANSAC Again")
                    something_print = 1
                X_temp = np.delete(X0, idx_x1_in_X0).reshape((-1, 1))
                Y_temp = np.delete(Y0, idx_x1_in_X0).reshape((-1, 1))
                idx_Xtemp_in_X0 = np.delete(idx_X0_in_X0, idx_x1_in_X0)
                inlier_mask1, outlier_mask1, line_y_ransac1, line_X1, theta_line1 = RANSAC(X_temp, Y_temp, th_ransac_k)
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
                        something_print = 1
                else:
                    line1_success = False
                    if _print_:
                        print("Not get Line1")
                        print(
                            "line1_length:{}<{}".format(line1_length, th_length) + "diff_x1:{}>{}".format(np.max(np.abs(diff_x1)), th_interval))
                        something_print = 1
        except Exception as e:
            line1_success = False
            if _print_:
                print(e)
                print("Line1 RANSAC False, No Line in the picture")
                something_print = 1
        if something_print == 0:
            stop = 1
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
            inlier_mask2, outlier_mask2, line_y_ransac2, line_X2, theta_line2 = RANSAC(X1, Y1, th_ransac_k)
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
                inlier_mask2, outlier_mask2, line_y_ransac2, line_X2, theta_line2 = RANSAC(X_temp, Y_temp, th_ransac_k)
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

    def get_fea_A(self, _print_=False):
        if self.need_to_check_A:
            if _print_:
                print('A feature finding')
            Acenter_x = np.min(self.stair_low_x)
            Acenter_y = self.stair_low_y[np.argmin(self.stair_low_x)][0]
            idx_fea_A = np.logical_and(np.abs(self.pcd_new[:, 0] - Acenter_x) < 0.015,
                                       np.abs(self.pcd_new[:, 1] - Acenter_y) < 0.015)
            if np.shape(self.pcd_new[idx_fea_A, 0])[0] > 10:
                fea_Ax_new = self.pcd_new[idx_fea_A, 0].reshape((-1, 1))
                fea_Ay_new = self.pcd_new[idx_fea_A, 1].reshape((-1, 1))
                if _print_:
                    print("find feature A:{},{}".format(Acenter_x, Acenter_y))
            else:
                mean_Ax = Acenter_x
                mean_Ay = Acenter_y
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Ax_new = mean_Ax + rand * 0.0001
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Ay_new = mean_Ay + rand * 0.0001
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
            Bcenter_x = stair_low_right_corner_x = max(np.min(self.stair_high_x), np.max(self.stair_low_x))
            # Bcenter_x = stair_low_right_corner_x = np.min(self.stair_high_x)
            Bcenter_y = stair_low_right_corner_y = np.nanmean(self.stair_low_y)
            # Bcenter_y = stair_low_right_corner_y = self.stair_low_y[np.argmax(self.stair_low_x)][0]
            idx_fea_B = np.logical_and(np.abs(self.pcd_new[:, 0] - stair_low_right_corner_x) < 0.015,
                                       np.abs(self.pcd_new[:, 1] - stair_low_right_corner_y) < 0.015)
            if np.shape(idx_fea_B)[0] > 10:
                fea_Bx_new = self.pcd_new[idx_fea_B, 0].reshape((-1, 1))
                fea_By_new = self.pcd_new[idx_fea_B, 1].reshape((-1, 1))
                if _print_:
                    print("find feature B:{},{}".format(Bcenter_x, Bcenter_y))
            else:
                mean_Bx = Bcenter_x
                mean_By = Bcenter_y
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Bx_new = mean_Bx + rand * 0.0001
                rand = np.random.rand(15).reshape((-1, 1))
                fea_By_new = mean_By + rand * 0.0001
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
                X_right_part = self.pcd_right_up_part_sa[:, 0]
                Y_right_part = self.pcd_right_up_part_sa[:, 1]
                Ccenter_x = np.min(X_right_part)
                Ccenter_y = Y_right_part[np.argmin(X_right_part)]
            else:
                print("Feature C finding in Condition2")
                Ccenter_x = np.min(self.stair_high_x)
                # Ccenter_y = self.stair_high_y[np.argmin(self.stair_high_x)][0]
                Ccenter_y = np.nanmean(self.stair_high_y)

            idx_fea_C = np.logical_and(np.abs(self.pcd_new[:, 0] - Ccenter_x) < 0.015,
                                       np.abs(self.pcd_new[:, 1] - Ccenter_y) < 0.015)
            if np.shape(idx_fea_C)[0] > 10:
                fea_Cx_new = self.pcd_new[idx_fea_C, 0].reshape((-1, 1))
                fea_Cy_new = self.pcd_new[idx_fea_C, 1].reshape((-1, 1))
                if _print_:
                    print("find feature C:{},{}".format(Ccenter_x, Ccenter_y))
            else:
                mean_Cx = Ccenter_x
                mean_Cy = Ccenter_y
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Cx_new = mean_Cx + rand * 0.0001
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Cy_new = mean_Cy + rand * 0.0001
                if _print_:
                    print("complete feature C:{},{}".format(Ccenter_x, Ccenter_y))
            self.Ccenter = np.array([Ccenter_x, Ccenter_y])
            self.fea_C = np.hstack([fea_Cx_new, fea_Cy_new])
            self.is_fea_C_gotten = True
        else:
            self.is_fea_C_gotten = False

    def get_fea_D(self, _print_=False):
        if self.need_to_check_D:
            if _print_:
                print("D feature finding")
            if self.has_part_up_line:
                Dcenter_x = np.max(self.pcd_left_up_part_sd[:, 0])
                Dcenter_y = self.pcd_left_up_part_sd[np.argmax(self.pcd_left_up_part_sd[:, 0]), 1]
            elif not self.has_part_up_line:
                Dcenter_x = np.max(self.stair_high_x[:, 0])
                Dcenter_y = self.stair_high_y[np.argmax(self.stair_high_x[:, 0])][0]

            idx_fea_D = np.logical_and(np.abs(self.pcd_new[:, 0] - Dcenter_x) < 0.03,
                                       np.abs(self.pcd_new[:, 1] - Dcenter_y) < 0.03)
            if np.shape(idx_fea_D)[0] > 10:
                fea_Dx_new = self.pcd_new[idx_fea_D, 0].reshape((-1, 1))
                fea_Dy_new = self.pcd_new[idx_fea_D, 1].reshape((-1, 1))
                if _print_:
                    print("find feature C:{},{}".format(Dcenter_x, Dcenter_y))
            else:
                mean_Dx = Dcenter_x
                mean_Dy = Dcenter_y
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Dx_new = mean_Dx + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Dy_new = mean_Dy + rand * 0.001
                if _print_:
                    print("complete feature D:{},{}".format(Dcenter_x, Dcenter_y))
            self.Dcenter = np.array([Dcenter_x, Dcenter_y])
            self.fea_D = np.hstack([fea_Dx_new, fea_Dy_new])
            self.is_fea_D_gotten = True
        else:
            self.is_fea_D_gotten = False

    def get_fea_E(self, _print_=False):
        if self.need_to_check_E:
            if _print_:
                print('E feature finding')
            if self.has_part_up_line:
                Ecenter_x = min(np.min(self.stair_high_x), np.max(self.pcd_left_up_part_sd[:, 0]))
                Ecenter_y = np.nanmean(self.stair_high_y)
            else:
                Ecenter_x = min(np.max(self.stair_high_x), np.min(self.stair_low_x))
                Ecenter_y = np.nanmean(self.stair_low_y)

            idx_fea_E = np.logical_and(np.abs(self.pcd_new[:, 0] - Ecenter_x) < 0.02,
                                       np.abs(self.pcd_new[:, 1] - Ecenter_y) < 0.02)
            if np.shape(self.pcd_new[idx_fea_E, 0])[0] > 10:
                fea_Ex_new = self.pcd_new[idx_fea_E, 0].reshape((-1, 1))
                fea_Ey_new = self.pcd_new[idx_fea_E, 1].reshape((-1, 1))
                if _print_:
                    print("find feature E:{},{}".format(Ecenter_x, Ecenter_y))
            else:
                mean_Ex = Ecenter_x
                mean_Ey = Ecenter_y
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Ex_new = mean_Ex + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Ey_new = mean_Ey + rand * 0.001
                if _print_:
                    print("complete feature E:{},{}".format(Ecenter_x, Ecenter_y))
            self.Ecenter = np.array([Ecenter_x, Ecenter_y])
            self.fea_E = np.hstack([fea_Ex_new, fea_Ey_new])
            self.is_fea_E_gotten = True
        else:
            self.is_fea_E_gotten = False

    def get_fea_F(self, _print_=False):
        if self.need_to_check_F:
            if _print_:
                print('F feature finding')
            if self.has_part_up_line:
                Fcenter_x = np.max(self.stair_high_x)
                Fcenter_y = np.nanmean(self.stair_high_y)
            else:
                Fcenter_x = np.max(self.stair_low_x)
                Fcenter_y = np.nanmean(self.stair_low_y)
            idx_fea_F = np.logical_and(np.abs(self.pcd_new[:, 0] - Fcenter_x) < 0.02,
                                       np.abs(self.pcd_new[:, 1] - Fcenter_y) < 0.02)
            if np.shape(self.pcd_new[idx_fea_F, 0])[0] > 10:
                fea_Fx_new = self.pcd_new[idx_fea_F, 0].reshape((-1, 1))
                fea_Fy_new = self.pcd_new[idx_fea_F, 1].reshape((-1, 1))
                if _print_:
                    print("find feature B:{},{}".format(Fcenter_x, Fcenter_y))
            else:
                mean_Fx = Fcenter_x
                mean_Fy = Fcenter_y
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Fx_new = mean_Fx + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Fy_new = mean_Fy + rand * 0.001
                if _print_:
                    print("complete feature F:{},{}".format(Fcenter_x, Fcenter_y))
            self.Fcenter = np.array([Fcenter_x, Fcenter_y])
            self.fea_F = np.hstack([fea_Fx_new, fea_Fy_new])
            self.is_fea_F_gotten = True
        else:
            self.is_fea_F_gotten = False

    def line_ground_calibrate(self, idx_x1_in_X0):
        if self.num_line > 0:
            X0 = self.pcd_new[:, 0].reshape((-1, 1))
            Y0 = self.pcd_new[:, 1].reshape((-1, 1))
            x1 = X0[idx_x1_in_X0, :]
            y1 = Y0[idx_x1_in_X0, :]
            k, b = np.polyfit(x1.reshape((-1,)), y1.reshape((-1,)), 1)
            line_theta = math.atan(k)
            x = self.pcd_new[:, 0]
            y = self.pcd_new[:, 1]
            self.pcd_new[:, 0] = np.cos(line_theta) * x + np.sin(line_theta) * y
            self.pcd_new[:, 1] = -np.sin(line_theta) * x + np.cos(line_theta) * y
            self.env_rotate += line_theta
            X0 = self.pcd_new[:, 0].reshape((-1, 1))
            Y0 = self.pcd_new[:, 1].reshape((-1, 1))
            x1 = X0[idx_x1_in_X0, :]
            y1 = Y0[idx_x1_in_X0, :]
            mean_line1 = np.nanmean(y1)
            idx_x1_in_X0 = np.where(abs(Y0 - mean_line1) < 0.01)[0]
        return x1, y1, mean_line1, idx_x1_in_X0

    def show_(self, ax, pcd_color='r', id=0, p_text=0.1, p_pcd=None, downsample=2):
        if p_pcd is None:
            p_pcd = [0, 0]
        ax.scatter(self.pcd_new[0:-1:downsample, 0] + p_pcd[0],
                   self.pcd_new[0:-1:downsample, 1] + p_pcd[1], color=pcd_color,
                   alpha=0.1)
        plt.text(p_text, -0.1, 'id: {}'.format(id))
        if self.fea_extra_over:
            plt.text(p_text, 0.5, 'theta: {}'.format(round(self.env_rotate, 2)))
            plt.text(p_text, -0.3, "time cost:{}ms".format(round(1000 * self.cost_time, 2)))
            if self.num_line == 2:
                ax.plot(self.stair_low_x + p_pcd[0], self.stair_low_y + p_pcd[1], color='black', linewidth=2)
                ax.plot(self.stair_high_x + p_pcd[0], self.stair_high_y + p_pcd[1], color='tab:gray', linewidth=2)
            elif self.num_line == 1:
                ax.plot(self.stair_low_x + p_pcd[0], self.stair_low_y + p_pcd[1], color='black', linewidth=2)
            if self.env_type == 1:
                if self.is_fea_A_gotten:
                    ax.scatter(self.fea_A[0:-1:5, 0] + p_pcd[0], self.fea_A[0:-1:5, 1] + p_pcd[1], color='g',
                               linewidths=2)
                if self.is_fea_B_gotten:
                    ax.scatter(self.fea_B[0:-1:5, 0] + p_pcd[0], self.fea_B[0:-1:5, 1] + p_pcd[1], color='y',
                               linewidths=2)
                if self.is_fea_C_gotten:
                    ax.scatter(self.fea_C[0:-1:5, 0] + p_pcd[0], self.fea_C[0:-1:5, 1] + p_pcd[1], color='m',
                               linewidths=2)
            # elif self.env_type == 2:
            #     if self.is_fea_D_gotten:
            #         ax.scatter(self.fea_D[0:-1:2, 0] + p_pcd[0], self.fea_D[0:-1:2, 1] + p_pcd[1], color='tab:purple',
            #                    linewidths=2)
            #     if self.is_fea_E_gotten:
            #         ax.scatter(self.fea_E[0:-1:2, 0] + p_pcd[0], self.fea_E[0:-1:2, 1] + p_pcd[1], color='tab:green',
            #                    linewidths=2)
            #     if self.is_fea_F_gotten:
            #         ax.scatter(self.fea_F[0:-1:2, 0] + p_pcd[0], self.fea_F[0:-1:2, 1] + p_pcd[1], color='tab:orange',
            #                    linewidths=2)
            #
            # elif self.env_type == 3:
            #     if self.is_fea_B_gotten:
            #         ax.scatter(self.fea_B[0:-1:5, 0] + p_pcd[0], self.fea_B[0:-1:5, 1] + p_pcd[1], color='y',
            #                    linewidths=1)
            #     if self.is_fea_C_gotten:
            #         ax.scatter(self.fea_C[0:-1:5, 0] + p_pcd[0], self.fea_C[0:-1:5, 1] + p_pcd[1], color='m',
            #                    linewidths=1)
            #     if self.is_fea_D_gotten:
            #         ax.scatter(self.fea_D[0:-1:5, 0] + p_pcd[0], self.fea_D[0:-1:5, 1] + p_pcd[1], color='tab:pink',
            #                    linewidths=1)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.text(p_text, -0.2, 'corner: {}'.format(self.corner_situation))

    def show_fast(self, fast_plot_ax: FastPlotCanvas, type, id=0, p_text=[-0.5, 0], p_pcd=None, downsample=2):
        if p_pcd is None:
            p_pcd = [0, 0]
        fast_plot_ax.set_pcd(self.pcd_new[0:-1:downsample] + p_pcd, type)
        if self.fea_extra_over:
            if self.env_type == 1:
                if self.is_fea_A_gotten:
                    fast_plot_ax.set_fea_A(self.fea_A[0:-1:5] + p_pcd)
                if self.is_fea_B_gotten:
                    fast_plot_ax.set_fea_B(self.fea_B[0:-1:5] + p_pcd)
                if self.is_fea_C_gotten:
                    fast_plot_ax.set_fea_C(self.fea_C[0:-1:5] + p_pcd)
            elif self.env_type == 2:
                if self.is_fea_D_gotten:
                    fast_plot_ax.set_fea_D(self.fea_D[0:-1:5] + p_pcd)
                if self.is_fea_E_gotten:
                    fast_plot_ax.set_fea_E(self.fea_E[0:-1:5] + p_pcd)
                if self.is_fea_F_gotten:
                    fast_plot_ax.set_fea_F(self.fea_F[0:-1:5] + p_pcd)

            elif self.env_type == 3:
                if self.is_fea_B_gotten:
                    fast_plot_ax.set_fea_B(self.fea_B[0:-1:5] + p_pcd)
                if self.is_fea_C_gotten:
                    fast_plot_ax.set_fea_C(self.fea_C[0:-1:5] + p_pcd)
                if self.is_fea_D_gotten:
                    fast_plot_ax.set_fea_D(self.fea_D[0:-1:5] + p_pcd)
            fast_plot_ax.set_info(p_text[0], p_text[1], type, id, self.corner_situation, self.env_rotate)

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
