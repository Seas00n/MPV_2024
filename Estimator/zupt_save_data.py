import time

import numpy as np

from fusion_algo import *
from Utils.Algo import *
from Utils.IO import *

import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
from filterpy.kalman import FixedLagSmoother, KalmanFilter

from scipy import signal
from matplotlib import pyplot as plt

import sys

sys.path.append("/home/yuxuan/Project/MPV_2024/")



use_data_set = True
num_frame = 600
freq = 40

if not use_data_set:
    str = input("按回车开始")

def low_pass_filter(mag_acc, acc_mag_buffer, acc_mag_lp_buffer, low_freq=12, freq=40):
    acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
    acc_mag_lp = jason_rewrite_of_tons_lowpass_filter(cutoff_freq=low_freq,
                                                      sample_time=1 / freq,
                                                      x0=acc_mag_buffer[2],
                                                      x1=acc_mag_buffer[1],
                                                      x2=acc_mag_buffer[0],
                                                      y1=acc_mag_lp_buffer[1],
                                                      y2=acc_mag_lp_buffer[0])
    acc_mag_lp_buffer = fifo_data_vec(acc_mag_lp_buffer, acc_mag_lp)
    mag_acc = acc_mag_lp
    return mag_acc, acc_mag_buffer, acc_mag_lp_buffer


def save_data():
    imu_knee_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
    imu_ankle_buffer = np.memmap("../Sensor/IM948/imu_ankle.npy", dtype="float32", mode='r', shape=(14,))
    imu_knee_save = []
    imu_ankle_save = []
    for i in range(900):
        imu_knee_data = np.copy(imu_knee_buffer[:])
        imu_ankle_data = np.copy(imu_ankle_buffer[:])
        accx = imu_ankle_data[1]
        accy = imu_ankle_data[2]
        accz = imu_ankle_data[3]
        mag_acc = np.sqrt(accx ** 2 + accy ** 2 + accz ** 2)
        print(mag_acc)
        imu_knee_save.append(imu_knee_data)
        imu_ankle_save.append(imu_ankle_data)
        print(i)
        time.sleep(1 / 40)

    imu_knee_save = np.array(imu_knee_save)
    imu_ankle_save = np.array(imu_ankle_save)
    np.save("imu_knee_zupt2.npy", imu_knee_save)
    np.save("imu_ankle_zupt2.npy", imu_ankle_save)


if __name__ == "__main__":
    save_data()