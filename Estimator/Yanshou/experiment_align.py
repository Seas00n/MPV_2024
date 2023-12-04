import sys

sys.path.append("//")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL
import os
from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *
from Environment.Plot_ import *
from Estimator.fusion_algo import StateKalmanFilter
from Utils.IO import fifo_data_vec
from Utils.pcd_os_fast_plot import *

down_sample_rate = 1
if down_sample_rate % 2 == 1:
    num_points = int(38528 / down_sample_rate) + 1
    if num_points > 38528:
        num_points = 38527
else:
    num_points = int(38528 / down_sample_rate)

idx_exp = 1
file_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(idx_exp)
moca_data = np.load(file_path + "Moca{}_smooth.npy".format(idx_exp))
align_file_path = "idx_align{}.npy".format(idx_exp)
time_moca = moca_data[:, 1]
time_moca -= time_moca[0]
frame_moca = moca_data[:, 0]
idx_stair = np.arange(2, 2 + 3 * 8)
idx_cam = np.arange(26, 29)
idx_knee = np.arange(29, 32)
idx_ankle = np.arange(32, 35)
idx_heel = np.arange(35, 38)
idx_toe = np.arange(38, 41)
knee = moca_data[:, idx_knee]
heel = moca_data[:, idx_heel]
toe = moca_data[:, idx_toe]

num_frame_vio = int((len(os.listdir(file_path)) - 2) / 3)
imu_buffer = []
time_buffer = []
for i in range(num_frame_vio):
    imu_buffer.append(np.load(file_path + "{}_imu.npy".format(i + 1), allow_pickle=True))
    time_buffer.append(np.load(file_path + "{}_time.npy".format(i + 1), allow_pickle=True))
imu_data = np.array(imu_buffer)
time_vio = np.array(time_buffer)
time_vio -= time_vio[0]
frame_vio = np.arange(0, np.shape(imu_data)[0])
idx_acc = np.arange(1, 4)
idx_omega = np.arange(4, 7)
idx_eular = np.arange(7, 10)
acc = imu_data[:, idx_acc]
omega = imu_data[:, idx_omega]
eular = imu_data[:, idx_eular]

idx_align = np.load(align_file_path, allow_pickle=True)
# idx_align = [0, 0]
print(idx_align)
counter = 0


fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
def on_press(event):
    global counter, idx_align
    global ax
    if counter == 0:
        idx_align[0] = int(event.xdata)
        counter += 1
        print(idx_align)
    elif counter == 1:
        idx_align[1] = int(event.xdata)
        print(idx_align)
        counter += 1
    print("my position:", event.button, event.xdata, event.ydata)


ax1.plot(frame_moca, knee)
ax1.scatter(frame_moca[idx_align[0]], knee[idx_align[0], 0])
acc_norm = np.linalg.norm(acc * 100, axis=1)
ax2.plot(frame_vio, acc_norm)
ax2.plot(frame_vio, eular)
ax2.scatter(frame_vio[idx_align[1]], acc_norm[idx_align[1]])
fig.canvas.mpl_connect('button_press_event', on_press)
plt.show()

input("存储")
np.save(align_file_path, np.array(idx_align))
