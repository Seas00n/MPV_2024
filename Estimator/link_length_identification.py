import numpy as np
import matplotlib.pyplot as plt
from Environment.Environment import *
from Environment.Plot_ import *

import datetime
import random
import sys


sys.path.append("/home/yuxuan/Project/MPV_2024/")
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import lcm
import time
import PIL
import os
from scipy import interpolate
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *

from pydrake.solvers import MathematicalProgram, Solve

env = Environment()
down_sample_rate = 5

str = input("按回车开始")
if __name__ == "__main__":
    imu_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
    num_points = int(38528 / down_sample_rate) + 1
    pcd_buffer = np.memmap("../Sensor/RoyaleSDK/pcd_buffer.npy", dtype='float32', mode='r', shape=(num_points, 3))

    ax = plt.axes()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.ion()
    num_frame = 600

    t0 = time.time()
    imu_initial_buffer = []
    count = 0
    while time.time() - t0 < 0.5:
        imu_data = imu_buffer[:]
        imu_initial_buffer.append(imu_data)
        count += 1
        time.sleep(0.001)
    imu_initial = np.mean(np.array(imu_initial_buffer), axis=0)

    link_default = 0.3823
    foot_height = 0.086

    knee_angle_list = []
    ground_height_list = []
    try:
        for i in range(num_frame):
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            plt.cla()
            pcd_data_temp = pcd_buffer[:]
            imu_data = imu_buffer[:]
            eular_angle = imu_data[7:10]
            env.pcd_to_binary_image(pcd_data_temp, eular_angle)
            env.thin()
            env.classification_from_img()

            pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
            pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
            line_theta = pcd_os.env_rotate
            pcd_os.show_(ax, pcd_color='r', id=int(i), downsample=3)

            knee_angle = (eular_angle[0] - imu_initial[7]) * np.pi / 180
            line_x = [0, link_default * np.cos(np.pi * 3 / 2 - knee_angle)]
            line_y = [0, link_default * np.sin(np.pi * 3 / 2 - knee_angle)]
            foot_x = [link_default * np.cos(np.pi * 3 / 2 - knee_angle), link_default * np.cos(np.pi * 3 / 2 - knee_angle)+0.1]
            foot_y = [link_default * np.sin(np.pi * 3 / 2 - knee_angle)-foot_height, link_default * np.sin(np.pi * 3 / 2 - knee_angle)-foot_height]
            ax.plot(line_x, line_y, color='g', linewidth=5)
            ax.plot(foot_x, foot_y, color='b', linewidth=8)

            ground_height = np.mean(pcd_os.pcd_new[:, 1])

            knee_angle_list.append(knee_angle)
            ground_height_list.append(ground_height)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.draw()
            plt.pause(0.025)

    except KeyboardInterrupt:
        pass

    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(2, "x")

    A = np.ones((len(knee_angle_list), 2))
    for i in range(len(knee_angle_list)):
        A[i, 1] = np.cos(knee_angle_list[i])
    b = -1 * np.array(ground_height_list).reshape((len(ground_height_list), 1))
    cost = prog.Add2NormSquaredCost(A=A, b=b, vars=x)
    constraint1 = prog.AddLinearConstraint(x[0] >= 0.05)
    constraint2 = prog.AddLinearConstraint(x[0] <= 0.14)
    constraint3 = prog.AddLinearConstraint(x[1] >= 0.3)
    result = Solve(prog)
    print("标定结果")
    print(result.GetSolution(x))
