import datetime
# !/usr/bin/env python3




import time
import cv2
import numpy as np
import glob
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import sys

import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import *
from sklearn import linear_model, datasets
from scipy import io
import os


def is_converge(x ,y, scale):

    scale = scale * 0.001
    a = abs(x) < scale
    b = abs(y) < scale
    return a & b


def del_miss(indeces, dist, max_dist, th_rate=0.7):
    th_dist = max_dist * th_rate
    return np.array(indeces[:][np.where(dist[:] < th_dist)[0]])




def thin(pcd_2d):

    cloud = pcd_2d

    nb1 = NearestNeighbors(n_neighbors=20 ,algorithm='auto')
    nb1.fit(pcd_2d)
    dis, idx = nb1.kneighbors(cloud)

    x = cloud[idx ,:][: ,: ,0]
    y = cloud[idx ,:][: ,: ,1]
    thin_edge = np.array([np.mean(x ,1) ,np.mean(y ,1)]).T
    #
    X = thin_edge[:, 0][thin_edge[: ,1] < -0.15].reshape(-1, 1)
    y = thin_edge[:, 1][thin_edge[: ,1] < -0.15].reshape(-1, 1)
    ymax = np.max(y)
    idx_remove = np.where(ymax - y < 0.02)[0]
    if len(idx_remove) < 10:
        print('remove')
        X = np.delete(X, idx_remove).reshape(-1, 1)
        y = np.delete(y, idx_remove).reshape(-1, 1)

    thin_edge = np.concatenate((X, y), 1)
    return thin_edge









def RANSAC(X, y ,th):
    def is_data_valid(X_subset, y_subset):
        x = X_subset
        y = y_subset

        if abs(x[1 ] -x[0]) < 0.025:
            return False
        else:
            k = (y[1 ] -y[0] ) /(x[1 ] -x[0])

        theta = math.atan(k)

        if abs(theta) < th:
            r = True
        else:
            r = False

        return r

    ransac = linear_model.RANSACRegressor(min_samples=2, residual_threshold=0.03, is_data_valid=is_data_valid, max_trials=500)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models


    return inlier_mask, outlier_mask

class icp_alignment(object):
    def __init__(self, pcd_s ,pcd_t, flag):
        if flag is None:
            self.flag = 0
        else:
            self.flag = flag
        self.pcd_s = pcd_s
        self.pcd_t = pcd_t



    def get_fea(self ,flag_out):
        if flag_out == 0:  # 0-上楼梯及看到障碍物末端前，1-下楼梯及看到障碍物末端后
            print('SA')
            fea_t = self.get_fea_sa(self.pcd_t)
            flag_t = self.flag
            # print('flag', self.flag)
            fea_s = self.get_fea_sa(self.pcd_s)
            self.flag = flag_t

        else:
            print('SD')
            fea_t = self.get_fea_sd(self.pcd_t)
            flag_t = self.flag
            # print('flag', self.flag)
            fea_s = self.get_fea_sd(self.pcd_s)
            self.flag = flag_t
        return fea_s, fea_t



    def get_fea_sa(self, pcd):
        X0 = pcd[:, 0].reshape((-1, 1))
        Y0 = pcd[:, 1]
        th = 0.05


        inlier_mask1, outlier_mask1 = RANSAC(X0, Y0, th)  # 第一次RANSAC拟合

        y1 = Y0[inlier_mask1]
        mean_y1 = np.mean(y1)
        idx1 = np.where(abs(Y0 - mean_y1) < 0.02)[0]  # 以第一次拟合的直线的y坐标均值为中心，选取在其上下一定范围内的点作为第一条直线，避免单纯依靠RANSAC拟合的直线太细，从而使第二次拟合的直线也在这层台阶上
        x1 = X0[idx1]
        y1 = Y0[idx1]

        X1 = np.delete(X0, idx1).reshape(-1, 1)  # 删掉第一次拟合的点
        Y1 = np.delete(Y0, idx1)

        try:
            inlier_mask2, outlier_mask2 = RANSAC(X1, Y1, th)  # 第二次RANSAC拟合
            y2 = Y1[inlier_mask2]
            mean_y2 = np.mean(y2)
            idx2 = np.where(abs(Y1 - mean_y2) < 0.015)[0]
            x2 = X1[idx2]
            y2 = Y1[idx2]

            # plt.scatter(X0,Y0)
            # plt.scatter(x2,y2)
            # plt.show()



            if max(x2 ) -min(x2) < 0.03 or abs(mean_y2 -mean_y1) < 0.05:  # 第二次拟合的直线点数太少，可能导致计算误差，仅使用第一次拟合的直线进行估计
                print('第二次拟合点数目过少')
                if mean_y1 - np.min(Y0) > 0.08:
                    if self.flag == 1:
                        idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
                    else:
                        idx_fea = np.where((x1 - np.min(x1)) < 0.03)[0]
                else:
                    self.flag = 1
                    idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
                    # print('flag to 1')
                feax = x1[idx_fea].reshape(-1, 1)
                feay = y1[idx_fea].reshape(-1, 1)

            else:
                if mean_y2 > mean_y1:  # 根据两直线y轴平均坐标判断哪个在下，f1为下面的楼梯，f2为上面的
                    f2x, f2y = x2, y2
                    f1x, f1y = x1, y1
                else:
                    f2x, f2y = x1, y1
                    f1x, f1y = x2, y2
                if self.flag == 0:
                    idx_fea = np.where(f2x - np.min(f2x) < 0.03)[0]
                    feax = f2x[idx_fea].reshape(-1, 1)
                    feay = f2y[idx_fea].reshape(-1, 1)
                else:
                    idx_fea = np.where(np.max(f1x) - f1x < 0.03)[0]
                    feax = f1x[idx_fea].reshape(-1, 1)
                    feay = f1y[idx_fea].reshape(-1, 1)
                self.flag = 0
                # print('flag to 0')
            fea = np.concatenate((feax, feay), 1)
        except:
            print('第二次拟合失败')
            if mean_y1 -np.min(Y0) > 0.05:
                if self.flag == 1:
                    idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
                else:
                    idx_fea = np.where((x1 - np.min(x1)) < 0.03)[0]
            else:
                self.flag = 1
                idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
                # print('flag to 1')
            feax = x1[idx_fea].reshape(-1, 1)
            feay = y1[idx_fea].reshape(-1, 1)
            fea = np.concatenate((feax, feay), 1)
        return fea


    def get_fea_sd(self, pcd):
        X = pcd[:, 0].reshape((-1, 1))
        y = pcd[:, 1]
        th = 0.05

        ymax = np.max(y)
        X0 = X[ymax - y < 0.25]
        Y0 = y[ymax - y < 0.25]


        inlier_mask1, outlier_mask1= RANSAC(X0, Y0, th)  # 第一次RANSAC拟合

        y1 = Y0[inlier_mask1]
        mean_y1 = np.mean(y1)
        idx1 = np.where(abs(Y0 - mean_y1) < 0.02)[0]  # 以第一次拟合的直线的y坐标均值为中心，选取在其上下一定范围内的点作为第一条直线，避免单纯依靠RANSAC拟合的直线太细，从而使第二次拟合的直线也在这层台阶上
        x1 = X0[idx1]
        y1 = Y0[idx1]

        X1 = np.delete(X0, idx1).reshape(-1, 1)  # 删掉第一次拟合的点
        Y1 = np.delete(Y0, idx1)
        try:
            inlier_mask2, outlier_mask2 = RANSAC(X1, Y1, th)  # 第二次RANSAC拟合
            y2 = Y1[inlier_mask2]
            mean_y2 = np.mean(y2)
            idx2 = np.where(abs(Y1 - mean_y2) < 0.015)[0]
            x2 = X1[idx2]
            y2 = Y1[idx2]

            # plt.scatter(X0,Y0)
            # plt.scatter(x2,y2)
            # plt.show()

            if np.max(x2 ) -np.min(x2) < 0.03 or abs(mean_y2 -mean_y1) < 0.05:  # 第二次拟合的直线点数太少，可能导致计算误差，仅使用第一次拟合的直线进行估计
                print('第二次拟合点数目过少')
                if np.max(x1) - np.min(x1) > 0.35:  # 第一级or最后一级
                    if mean_y1 - np.min(Y0) < 0.05:  # 最后一级
                        idx_fea = np.where((np.max(Y0) - Y0) < 0.03)[0]
                    else:  # 第一级
                        idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
                else:
                    idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]

                self.flag = 1
                # print('flag to 1')
                feax = x1[idx_fea].reshape(-1, 1)
                feay = y1[idx_fea].reshape(-1, 1)

            else:
                if mean_y2 > mean_y1:  # 根据两直线y轴平均坐标判断哪个在下，f1为下面的楼梯，f2为上面的
                    f2x, f2y = x2, y2
                    f1x, f1y = x1, y1
                else:
                    f2x, f2y = x1, y1
                    f1x, f1y = x2, y2
                if self.flag == 0:
                    idx_fea = np.where(np.max(f2x) - f2x < 0.03)[0]
                    feax = f2x[idx_fea].reshape(-1, 1)
                    feay = f2y[idx_fea].reshape(-1, 1)
                else:
                    idx_fea = np.where(np.max(f1x) - f1x < 0.03)[0]
                    feax = f1x[idx_fea].reshape(-1, 1)
                    feay = f1y[idx_fea].reshape(-1, 1)
                self.flag = 0
                # print('flag to 0')
            fea = np.concatenate((feax, feay), 1)
        except:
            print('第二次拟合失败')
            if np.max(x1) - np.min(x1) > 0.35:  # 第一级or最后一级
                if mean_y1 -np.min(Y0) < 0.05:  # 最后一级
                    idx_fea = np.where((np.max(Y0) - Y0) < 0.03)[0]
                    feax = X0[idx_fea].reshape(-1, 1)
                    feay = Y0[idx_fea].reshape(-1, 1)
                    fea = np.concatenate((feax, feay), 1)
                    return fea
                else  :  # 第一级
                    idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
            else:
                if ymax - Y0[-1] < 0.05:
                    self.flag = 1
                    # print('flag to 1')
                else:
                    print('OB')
                idx_fea = np.where((np.max(x1) - x1) < 0.03)[0]
            feax = x1[idx_fea].reshape(-1, 1)
            feay = y1[idx_fea].reshape(-1, 1)
            fea = np.concatenate((feax, feay), 1)
        return fea

    def icp(self, pcd_s, pcd_t, max_iterate=20):

        a = min(len(pcd_s[:, 0]), len(pcd_t[:, 0]))
        pcd_s = pcd_s[0:a, :]
        pcd_t = pcd_t[0:a, :]
        src = np.array([pcd_s], copy=True).astype(np.float32)
        dst = np.array([pcd_t], copy=True).astype(np.float32)

        knn = cv2.ml.KNearest_create()
        responses = np.array(range(len(pcd_t[:, 0]))).astype(np.float32)

        knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)

        xmove, ymove = 0, 0

        max_dist = sys.maxsize

        scale_x = np.max(pcd_s[:, 0]) - np.min(pcd_s[:, 0])
        scale_y = np.max(pcd_s[:, 1]) - np.min(pcd_s[:, 1])

        scale = max(scale_x, scale_y)

        for i in range(max_iterate):

            ret, results, neighbours, dist = knn.findNearest(dst[0], 1)

            indeces = results.astype(np.int32)

            indeces = del_miss(indeces, dist, max_dist)

            x_i = src[0, indeces, 0]
            y_i = src[0, indeces, 1]
            x_j = dst[0, indeces, 0]
            y_j = dst[0, indeces, 1]

            dist_x = np.mean(x_i - x_j)
            dist_y = np.mean(y_i - y_j)

            dst[0, :, 0] += dist_x
            dst[0, :, 1] += dist_y
            xmove += dist_x
            ymove += dist_y

            if (is_converge(dist_x, dist_y, scale)):
                break

        return xmove, ymove

    def alignment(self):
        t_ymax = np.max(self.pcd_t[:, 1])
        s_ymax = np.max(self.pcd_s[:, 1])
        if t_ymax - self.pcd_t[0, 1] < 0.05 or s_ymax - self.pcd_s[0, 1] < 0.05:
            flag_out = 0
        else:
            flag_out = 1
        fea_s, fea_t = self.get_fea(flag_out)
        t0 = datetime.datetime.now()
        # plt.subplot(211)
        # plt.scatter(self.pcd_s[:,0],self.pcd_s[:,1])
        # plt.scatter(fea_s[:, 0], fea_s[:, 1], linewidths=5, color='g')
        # plt.subplot(212)
        # plt.scatter(self.pcd_t[:,0],self.pcd_t[:,1])
        # plt.scatter(fea_t[:, 0], fea_t[:, 1], linewidths=5, color='m')
        # plt.show()

        xmove, ymove = self.icp(fea_s, fea_t)
        if abs(xmove) > 0.1 or abs(ymove) > 0.1:
            fea_s, fea_t = self.get_fea(flag_out)
            print(flag_out)
            xmove, ymove = self.icp(fea_s, fea_t)
        if abs(xmove) > 0.1 or abs(ymove) > 0.1:
            self.flag = 1 - self.flag
            fea_s, fea_t = self.get_fea(flag_out)
            xmove, ymove = self.icp(fea_s, fea_t)
        t1 = datetime.datetime.now()
        print("#=====FeatureAlignCCH:{}=====#".format(
            (t1 - t0).total_seconds() * 1000))
        return xmove, ymove, self.flag


def cal_move(fea_i, fea_j):
    mean_xi = np.mean(fea_i[:, 0])
    mean_yi = np.mean(fea_i[:, 1])

    mean_xj = np.mean(fea_j[:, 0])
    mean_yj = np.mean(fea_j[:, 1])

    x_move = mean_xi - mean_xj
    y_move = mean_yi - mean_yj

    return x_move, y_move


def cal_speed(x_move, y_move, t):  # 计算相机速度，由于时间间隔较短（约30ms），认为在相邻两帧之间做匀速运动

    x_speed = x_move / t
    y_speed = y_move / t

    return x_speed, y_speed


#   SA1 110-550 2 110-525 3 80-490 4 110-500 5 90-480
#   SD1 110-600 2 150-600 3 120-550 4 130-570 5 110-480 +100
#   OB1 150-300 2 150-340 3 160-315 4 150-290 5 150-325


def main():
    x = 0
    y = 0
    trajectory = []
    start = 150
    end = 340
    pcd = io.loadmat(dst + 'pcd.mat')
    pcd = pcd['pcd_data_new'][0]
    time_vec = np.load(dst + 'time.npy')
    time_vec = time_vec.reshape(len(time_vec))

    flag = 0

    start_time = time_vec[start]  # 下楼梯起始点
    # plt.ion()
    # t = []
    # for i in range(start, end):
    #     j = i + 1
    #     t1 = time_vec[j] - start_time
    #     t.append(t1)
    #
    #     t0 = time.time()
    #     pcd_2d_i = pcd[i]
    #     pcd_2d_j = pcd[j]
    #
    #
    #
    #     n = min(len(pcd_2d_i[:, 0]), len(pcd_2d_j[:, 0]))
    #
    #     pcd_2d_i = pcd_2d_i[0:n, :]
    #     pcd_2d_j = pcd_2d_j[0:n, :]
    #     pcd_2d_i = thin(pcd_2d_i)
    #     pcd_2d_j = thin(pcd_2d_j)
    #
    #     regis = icp_alignment(pcd_2d_i, pcd_2d_j,flag)
    #     xmove, ymove,flag = regis.alignment()
    #     print('cost',time.time()-t0)
    #     if time.time()-t0>0.05:
    #         print('--------------------------------',time.time()-t0)
    #         print(i)
    #
    #     if abs(xmove)>0.1 or abs(ymove)>0.05:
    #         print('寄咯')
    #         print(flag)
    #         print(i)
    #         break
    #
    #
    #
    #     x += xmove
    #     y += ymove
    #
    #     # x_speed,y_speed = cal_speed(xmove,ymove,time_vec[j]-time_vec[i])    #计算相机移动速度
    #     # print('xspeed', x_speed)
    #     # print('yspeed', y_speed)
    #
    #     trajectory.append([x,y])
    #     #print('x,y',[x,y])
    #
    #     plt.clf()
    #     ax = plt.subplot(311)
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     plt.scatter(pcd_2d_i[:, 0], pcd_2d_i[:, 1])
    #     plt.scatter(pcd_2d_j[:, 0], pcd_2d_j[:, 1])
    #     labelss = plt.legend(['Last frame', 'This frame'], ncol=2, loc='lower center', bbox_to_anchor=(0.9, 0.8),
    #                          fontsize=10).get_texts()
    #     plt.title('Before alignment')
    #
    # #
    #     pcd_2d_j += [xmove, ymove]
    #     # if abs(xmove) > 0.03 or abs(ymove) > 0.02:
    #     #     print('--------------------------------------------')
    #     #     print(i)
    #     #     print(xmove)
    #     #     print(ymove)
    #
    #     bx = plt.subplot(312)
    #     bx.xaxis.set_visible(False)
    #     bx.yaxis.set_visible(False)
    #     plt.scatter(pcd_2d_i[:, 0], pcd_2d_i[:, 1])
    #     plt.scatter(pcd_2d_j[:, 0], pcd_2d_j[:, 1])
    #     plt.title('After alignment')
    #     labelss = plt.legend(['Last frame', 'This frame'], ncol=2, loc='lower center', bbox_to_anchor=(0.9, 0.8),
    #                          fontsize=10).get_texts()
    #     plt.subplot(313)
    #     #plt.plot(np.array(trajectory)[:,0],np.array(trajectory)[:,1])
    #     plt.plot(np.array(t)[:],np.array(trajectory)[:,1])
    #     plt.title('Camera trajectory')
    #     plt.xlabel('t (s)')
    #     plt.ylabel('z (m)')
    #     plt.plot(np.array(t)[:],np.array(trajectory)[:,1])
    #     #plt.savefig('fig_sd/{}.png'.format(p))
    #     plt.pause(0.05)
    #     plt.ioff()
    # plt.clf()                #保存图像、轨迹
    # plt.ioff()
    # io.savemat(dst_tra + 'tra_sa.mat', {'trajectory': trajectory})
    # io.savemat(dst_tra + 't_sa.mat', {'t': t})
    # plt.plot(np.array(trajectory)[:,0],np.array(trajectory)[:,1])
    # plt.xlabel('x (m)')
    # plt.ylabel('z (m)')
    # plt.title('x-z')
    # plt.show()
    # plt.plot(np.array(t)[:],np.array(trajectory)[:,1])
    # plt.xlabel('t (s)')
    # plt.ylabel('z (m)')
    # plt.title('t-z')
    # plt.show()
    # plt.plot(np.array(t)[:],np.array(trajectory)[:,0])
    # plt.xlabel('t (s)')
    # plt.ylabel('x (m)')
    # plt.title('t-x')
    # plt.show()

    ## #以下为单帧调试
    # i = 469
    # j = i+1
    # pcd_i = pcd[i]
    # pcd_j = pcd[j]
    i = 249

    pcd_i = np.load('{}/{}_i.npy'.format(dst_pcd, i))
    pcd_j = np.load('{}/{}_j.npy'.format(dst_pcd, i))

    n = min(len(pcd_i[:, 0]), len(pcd_j[:, 0]))

    pcd_2d_i = pcd_i[0:n, :]
    pcd_2d_j = pcd_j[0:n, :]
    t1 = time.time()

    # 0.007s
    pcd_2d_i = thin(pcd_2d_i)
    pcd_2d_j = thin(pcd_2d_j)

    regis = icp_alignment(pcd_2d_i, pcd_2d_j, flag)
    xmove, ymove, flag = regis.alignment()

    print(time.time() - t1)
    print('xmove', xmove)
    print('ymove', ymove)
    #
    #
    plt.subplot(211)
    plt.scatter(pcd_2d_i[:, 0], pcd_2d_i[:, 1])
    plt.scatter(pcd_2d_j[:, 0], pcd_2d_j[:, 1])
    # plt.scatter(pcd[i-1][:,0],pcd[i-1][:,1])

    pcd_2d_j += [xmove, ymove]

    plt.subplot(212)
    plt.scatter(pcd_2d_i[:, 0], pcd_2d_i[:, 1])
    plt.scatter(pcd_2d_j[:, 0], pcd_2d_j[:, 1])
    plt.show()


if __name__ == '__main__':
    dst_tra = 'tra_215/OB/2/'
    dst = 'mat_215/OB/2/'
    dst_pcd = 'online'
    main()


