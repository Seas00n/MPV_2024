#!/usr/bin/env python3

# 读取已有的IMU和RGBD数据，并转换为二值图像

from datetime import datetime
import time
import sys
import os
import numpy as np
import torch
from scipy import stats
import open3d as o3d
import torchvision.transforms as transforms
from PIL import Image
import argparse
import configparser
import glob
import shutil
import cv2
from openni import openni2
from openni import _openni2 as c_api
from Utils import Plot, Algo, IO
import matplotlib.pyplot as plt
from IMUReader import *
import math
from sklearn import linear_model, datasets



dst = 'data_experiment/data/depth_out'


mirroring = False
compression = False
width_stream = 640
height_stream = 480
fs = 30 # Hz
capture_time = 60000
scale = 100
transform = transforms.Compose([
    transforms.Scale(scale),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



def rm_all_files(dst):
    files = glob.glob('{}/*'.format(dst))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

clear_dst = False  # 为True时会清除数据！！
if clear_dst:
    print('!!!!!!!!!!!!!!!Remove all files in {}!!!!!!!!!!!!!!!!!!!!!'.format(dst))
    rm_all_files(dst)






def classification_online(dev, dst):
    E = torch.load(
        'checkpoint/model_epoch49_G.pt')
    C = torch.load(
        'checkpoint/model_epoch49_C1.pt')
    E.cuda()
    C.cuda()
    pred_1 = np.zeros(8)


    depth_stream = dev.create_depth_stream()
    depth_stream.set_mirroring_enabled(mirroring)
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                   resolutionX=width_stream,
                                                   resolutionY=height_stream,
                                                   fps=fs))
    dev.set_image_registration_mode(True)
    dev.set_depth_color_sync_enabled(True)
    depth_stream.start()    #开相机，这里只获取深度信息进行测试

    current_date = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3]

    device_vec, callback_vec, control_vec,port_vec = IMU_Getready()  #开IMU，IMU_Getready()代码在IMUReader中

    if not os.path.exists(dst):
        os.mkdir(dst)
        print("Directory ", dst, " Created ")

    rec = openni2.Recorder((dst + '/' + current_date + ".oni").encode('utf-8'))
    rec.attach(depth_stream, compression)
    rec.start()  #开始录制视频
    start = time.time()
    print("Recording started.. press ctrl+C to stop")

    try:
        while True:
            if (time.time() - start) > capture_time:
                break

            current_time = time.time()
            captured_data_vec = capture_one_frame(callback_vec)
            count = 0

            while (captured_data_vec is None) or (count <= 2):
                count += 1
                captured_data_vec = capture_one_frame(
                    callback_vec)  # IMU数据读取，为避免IMU数据有延迟，与RGBD图像不同步，此处多读几次IMU数据，实现近似同步，capture_one_frame代码在IMUReader中

            frame_depth = depth_stream.read_frame()
            frame_depth_data = frame_depth.get_buffer_as_uint16()
            depth_array = np.ndarray((frame_depth.height, frame_depth.width), dtype=np.uint16,
                                     buffer=frame_depth_data)  # 深度图像获取，格式转换


            depth_name = '{}/{:.3f}.png'.format(dst, current_time)
            cv2.imwrite(depth_name, depth_array)
            pcd = IO.read_depth_pcd(depth_name)  # 深度图像转成点云
            img, pcd_2d = pcd_to_binary_image(pcd, captured_data_vec)

            image = img
            cv2.imshow('binaryimage', image)  # 显示二值图像
            cv2.waitKey(1)

            t1 = time.time()
            pred = classification(E, C, img)                   #单帧分类结果
            pred_1 = fifo_data_vec(pred_1, pred)
            pred_out = stats.mode(pred_1)[0]                   #众数滤波后的分类结果
            t2 = time.time()
            print('classification result:', pred_out)
            # print('classification time:', t2 - t1)

            width, height, theta = get_W_H_theta(pcd_2d[:, 1], pcd_2d[:, 0], pred_out)  #地形特征估计
            t3 = time.time()
            # print('estimation time', t3 - t2)
            print('total time',t3 - current_time)
            print('width', width)
            print('height', height)
            print('theta', theta)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    rec.stop()
    depth_stream.stop()
    close_device(device_vec, control_vec, port_vec, callback_vec)


def classification(E,C,img):


    img = img.reshape(1, 100, 100)
    im = np.uint8(np.asarray(img))
    im = np.vstack([im, im, im])

    im = im.transpose((1, 2, 0))
    img = transform(Image.fromarray(im))

    img = img.reshape(1, 3, 100, 100).cuda()


    with torch.no_grad():
        feat = E(img)
        output = C(feat)

    pred = output.data.max(1)[1].cpu().numpy()
    return pred





def pcd_to_binary_image(pcd, captured_data_vec):   #利用IMU数据把点云转换为二值图像
    pcd_array = np.asarray(pcd.points)   #将点云的格式转换为ndarray
    x = pcd_array[:, 0]
    y = pcd_array[:, 1]
    z = pcd_array[:, 2]
    y1 = y[np.logical_and(abs(x) < 0.02, abs(y) < 1)]
    z1 = z[np.logical_and(abs(x) < 0.02, abs(y) < 1)]   #降维，abs(x)的值表示降维时保留的点在x轴上的坐标范围，越大保留的点越多
    img = np.zeros([100, 100])


    eular = captured_data_vec[0:3]           #IMU的欧拉角
    theta = (-(eular[0] + 95) / 180) * np.pi   #加个补偿，转成弧度

    pcd_y = (y1 * np.cos(theta) - z1 * np.sin(theta))     #坐标系转换
    pcd_z = (z1 * np.cos(theta) + y1 * np.sin(theta))

    if np.any(pcd_y) :
        pcd_ymin = min(pcd_y)
        pcd_zmin = min(pcd_z)
        pcd_y -= pcd_ymin
        pcd_z -= pcd_zmin
        pcd_y += 1 - max(pcd_y)
        if min(pcd_y) < 0.4:
            pcd_y += 0.4-min(pcd_y)


        p = pcd_y[np.logical_and(abs(pcd_y) < 1, abs(pcd_z) < 1)]
        q = pcd_z[np.logical_and(abs(pcd_y) < 1, abs(pcd_z) < 1)]


        for i in range(len(q)):
            if q[i] < 1 and q[i] > 0.01 and p[i] < 1 and p[i] > 0.01: #二值图像生成
                q_int = int(100 * q[i])
                p_int = int(100 * p[i])

                img[p_int, q_int] = 1

        img = img * 255

        pcd_2d = np.zeros([len(p), 2])   #用于估计特征的二维点云
        pcd_2d[:, 0] = p
        pcd_2d[:, 1] = q

    else:
        pcd_2d = np.zeros(len(x),2)
    return img,pcd_2d


def fifo_data_vec(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat

def main():
    try:
        print(sys.platform)
        if sys.platform == "win32":
            libpath = "lib/Windows"
        else:
            libpath = "lib/Linux"
        # abs_lib_path = os.path.join(currentdir,libpath)
        print("library path is: ", libpath)
        openni2.initialize(libpath)
        print("OpenNI2 initialized \n")
    except Exception as ex:
        print("ERROR OpenNI2 not initialized",ex," check library path..\n")
        return
    try:
        dev = openni2.Device.open_any()
    except Exception as ex:
        print("ERROR Unable to open the device: ",ex," device disconnected? \n")
        return

    classification_online(dev, dst)

    try:
        openni2.unload()
        print("Device unloaded \n")
    except Exception as ex:
        print("Device not unloaded: ",ex, "\n")



def get_W_H_theta(X, y, flag):
    X = X.reshape((-1, 1))
    # y = y.reshape((-1, 1))
    try:
        if flag == 0 or flag == 3 or flag == 4:
            return get_theta(X, y)
        elif flag == 1 or flag == 2:
            return get_W_H(X, y)
    except:
        print('failed to calculate the pointCloud2')
        return 0, 0, 0


def get_theta(X, y):
    X = X.reshape(-1)
    x_ = np.mean(X)
    y_ = np.mean(y)
    temp = np.sum((X - x_) * (y - y_)) / np.sum((X - x_) ** 2)
    theta = math.atan(temp)
    theta = abs(theta / (2 * 3.1415926) * 360)
    width, height = 0, 0

    return width, height, theta



def RANSAC(X, y):
    def is_data_valid(X_subset, y_subset):
        x = X_subset

        y = y_subset

        if abs(x[1]-x[0]) < 0.05:
            return False
        else:
            k = (y[1]-y[0])/(x[1]-x[0])

        theta = math.atan(k)
        if abs(theta) < 0.05:
            return True
        else:
            return False

    ransac = linear_model.RANSACRegressor(min_samples=2, residual_threshold=0.01, is_data_valid=is_data_valid, max_trials=400)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max(),0.01)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    return inlier_mask, outlier_mask, line_y_ransac, line_X

def get_W_H(X, y):
    # inlier_x, inlier_y = [], []
    inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X, y)


    # plt.subplot(311)
    # plt.scatter(X,y)
    # plt.subplot(312)
    # plt.scatter(X[inlier_mask1],y[inlier_mask1])
    # plt.subplot(313)
    # plt.plot(line_X1,line_y_ransac1)
    # plt.show()


    width = np.max(X[inlier_mask1]) - np.min(X[inlier_mask1])
    height = np.mean(y[inlier_mask1])
    #
    # inlier_x.extend(X[inlier_mask1])
    # inlier_y.extend(y[inlier_mask1])

    X = X[outlier_mask1]
    y = y[outlier_mask1]
    inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X, y)
    height = abs(height-(np.mean(y[inlier_mask2])))
    if height > 0.3:
        height /= 2
    theta = 0
    return width, height, theta


if __name__ == '__main__':

    main()






