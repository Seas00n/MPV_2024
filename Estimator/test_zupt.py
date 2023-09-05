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



use_data_set = False
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
    #
    # imu_knee_buffer = np.memmap("../Sensor/IM948/imu_knee.npy", dtype='float32', mode='r', shape=(14,))
    # imu_ankle_buffer = np.memmap("../Sensor/IM948/imu_ankle.npy", dtype="float32", mode='r', shape=(14,))
    #
    # count = 0
    # if use_data_set:
    #     imu_ankle_data_all = np.load("imu_ankle_zupt2.npy")
    #     imu_knee_data_all = np.load("imu_knee_zupt2.npy")
    #
    # t0 = time.time()
    # imu_knee_initial_buffer = []
    # imu_ankle_initial_buffer = []
    # mahony = ahrs.filters.Mahony(Kp=1, Ki=0, KpInit=1, frequency=freq)  # type: ahrs.filters.mahony
    # while time.time() - t0 < 1:
    #     if not use_data_set:
    #         imu_knee_data = imu_knee_buffer[:]
    #         imu_ankle_data = imu_ankle_buffer[:]
    #     else:
    #         imu_ankle_data = imu_ankle_data_all[count, :]
    #         imu_knee_data = imu_knee_data_all[count, :]
    #         count += 1
    #     imu_knee_initial_buffer.append(imu_knee_data)
    #     imu_ankle_initial_buffer.append(imu_ankle_data)
    #     time.sleep(0.02)
    # q_prior = np.array([1, 0, 0, 0], dtype=np.float64)
    # for i in range(len(imu_ankle_initial_buffer)):
    #     q_prior = mahony.updateIMU(q_prior, gyr=imu_ankle_initial_buffer[i][4:7] * np.pi / 180,
    #                                acc=imu_ankle_initial_buffer[i][1:4])
    # imu_knee_initial = np.mean(np.array(imu_knee_initial_buffer), axis=0)
    # imu_ankle_initial = np.mean(np.array(imu_ankle_initial_buffer), axis=0)
    #
    # foot_vx_list = [0]
    # foot_vy_list = [0]
    # foot_vz_list = [0]
    # foot_vxzupt_list = [0]
    # foot_vyzupt_list = [0]
    # foot_vzzupt_list = [0]
    # foot_px_list = [0]
    # foot_py_list = [0]
    # foot_pz_list = [0]
    # foot_pxzupt_list = [0]
    # foot_pyzupt_list = [0]
    # foot_pzzupt_list = [0]
    # foot_accmag_list = [0]
    # knee_accmag_list = [0]
    # foot_vmag_list = [0]
    # foot_vmagnew_list = [0]
    # time_buffer = []
    # foot_stationary_buffer = [1]
    # driftRateBuffer = [0]
    #
    # acc_mag_buffer = [0, 0, 0]
    # acc_mag_lp_buffer = [0, 0, 0]
    # acc_mag_knee_buffer = [0, 0, 0]
    # acc_mag_knee_lp_buffer = [0, 0, 0]
    #
    # if use_data_set:
    #     t0 = imu_ankle_data_all[count, 0]
    #     count += 1
    #     num_frame = len(imu_ankle_data_all) - count
    # else:
    #     t0 = imu_ankle_buffer[0]
    # time_buffer.append(t0)
    #
    # is_in_stand_phase = True
    # step_state_buffer = np.array([1])
    # step_period_buffer = [np.array([t0, 0, 0, 0])]
    # driftRate = np.array([0, 0, 0])
    #
    # fig, axes = plt.subplots(2, 1, sharex=True)
    #
    # try:
    #     for i in range(num_frame):
    #         print("----------------------------Frame[{}]------------------------".format(i))
    #         if use_data_set:
    #             time = imu_ankle_data_all[count, 0]
    #             time_buffer.append(time)
    #             accx = imu_ankle_data_all[count, 1] - imu_ankle_initial[1]
    #             accy = imu_ankle_data_all[count, 2] - imu_ankle_initial[2]
    #             accz = imu_ankle_data_all[count, 3] - imu_ankle_initial[3]
    #             gyrx = imu_ankle_data_all[count, 4] - imu_ankle_initial[4]
    #             gyry = imu_ankle_data_all[count, 5] - imu_ankle_initial[5]
    #             gyrz = imu_ankle_data_all[count, 6] - imu_ankle_initial[6]
    #             qw = imu_ankle_data_all[count, 10]
    #             qx = imu_ankle_data_all[count, 11]
    #             qy = imu_ankle_data_all[count, 12]
    #             qz = imu_ankle_data_all[count, 13]
    #             accx_knee = imu_knee_data_all[count, 1] - imu_knee_initial[1]
    #             accy_knee = imu_knee_data_all[count, 2] - imu_knee_initial[2]
    #             accz_knee = imu_knee_data_all[count, 3] - imu_knee_initial[3]
    #             count += 1
    #         else:
    #             time = imu_ankle_buffer[0]
    #             time_buffer.append(time)
    #             accx = imu_ankle_buffer[1]
    #             accy = imu_ankle_buffer[2]
    #             accz = imu_ankle_buffer[3]
    #             gyrx = imu_ankle_buffer[4]
    #             gyry = imu_ankle_buffer[5]
    #             gyrz = imu_ankle_buffer[6]
    #             accx_knee = imu_knee_buffer[1]
    #             accy_knee = imu_knee_buffer[2]
    #             accz_knee = imu_knee_buffer[3]
    #             qw = imu_ankle_buffer[10]
    #             qx = imu_ankle_buffer[11]
    #             qy = imu_ankle_buffer[12]
    #             qz = imu_ankle_buffer[13]
    #
    #         dt = time_buffer[-1] - time_buffer[-2]
    #         #
    #         # # bandwidth filter
    #         mag_acc = np.sqrt(accx ** 2 + accy ** 2 + accz ** 2)
    #         low_freq = 12
    #
    #         if i == 0:
    #             acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
    #             acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
    #             acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
    #         mag_acc, acc_mag_buffer, acc_mag_lp_buffer = low_pass_filter(mag_acc, acc_mag_buffer, acc_mag_lp_buffer,
    #                                                                      low_freq=low_freq, freq=freq)
    #
    #         mag_acc_knee = np.sqrt(accx_knee ** 2 + accy_knee ** 2 + accz_knee ** 2)
    #         if i == 0:
    #             acc_mag_knee_buffer = fifo_data_vec(acc_mag_knee_buffer, mag_acc_knee)
    #             acc_mag_knee_buffer = fifo_data_vec(acc_mag_knee_buffer, mag_acc_knee)
    #             acc_mag_knee_buffer = fifo_data_vec(acc_mag_knee_buffer, mag_acc_knee)
    #         mag_acc_knee, acc_mag_knee_buffer, acc_mag_knee_lp_buffer = low_pass_filter(mag_acc_knee,
    #                                                                                     acc_mag_knee_buffer,
    #                                                                                     acc_mag_knee_lp_buffer,
    #                                                                                     low_freq=low_freq, freq=freq)
    #
    #         is_stationary = mag_acc < 0.7
    #         step_state_buffer = np.append(step_state_buffer, int(is_stationary))
    #
    #         if is_stationary:
    #             mahony.Kp = 0.7
    #         else:
    #             mahony.Kp = 0
    #         gyr = np.array([gyrx, gyry, gyrz]) * np.pi / 180
    #         acc = np.array([accx, accy, accz])
    #         # quat = mahony.updateIMU(q_prior, gyr=gyr, acc=acc) #测试效果不好
    #         quat = np.array([qw, qx, qy, qz]).reshape((4, 1))
    #
    #         acc_in_world = q_rot(quat, np.array([accx, accy, accz]))
    #         acc_in_world = np.array(acc_in_world).reshape((3,))
    #
    #         vx = foot_vx_list[-1] + acc_in_world[0] * dt
    #         vy = foot_vy_list[-1] + acc_in_world[1] * dt
    #         vz = foot_vz_list[-1] + acc_in_world[2] * dt
    #
    #         if is_stationary:
    #             vx = 0
    #             vy = 0
    #             vz = 0
    #
    #         step_period_buffer.append(np.array([time_buffer[-1], vx, vy, vz]))
    #
    #         vx_new = vy_new = vz_new = 0
    #         driftVel = np.zeros((3,))
    #         if is_in_stand_phase:
    #             idx_stationaryEnd = np.where(np.diff(step_state_buffer) == -1)[0]
    #             if np.shape(idx_stationaryEnd)[0] > 0:
    #                 if idx_stationaryEnd[-1] > 5:
    #                     print("Switch to Swing")
    #                     step_state_buffer = step_state_buffer[-1]
    #                     step_period_buffer = [step_period_buffer[-1]]
    #                     is_in_stand_phase = False
    #         else:
    #             idx_stationaryStart = np.where(np.diff(step_state_buffer) == 1)[0]
    #             driftVel = (len(step_state_buffer) - 1) * driftRate
    #             if np.shape(idx_stationaryStart)[0] > 0:
    #                 if idx_stationaryStart[-1] > 5:
    #                     print("Switch to Stand")
    #                     idx_stationaryStart = idx_stationaryStart[-1]
    #                     driftVelResidual = np.array([vx, vy, vz]) - driftVel
    #                     driftVel = step_period_buffer[idx_stationaryStart][1:]  # - step_period_buffer[0][1:]
    #                     k = 1
    #                     driftRate = (k * driftVel + (1 - k) * driftVelResidual) / idx_stationaryStart
    #                     # more complex drift fit
    #
    #                     axes[1].scatter(i, np.linalg.norm(step_period_buffer[idx_stationaryStart][1:]), color='r')
    #                     axes[1].scatter(i - idx_stationaryStart, np.linalg.norm(step_period_buffer[0][1:]), color='g')
    #                     step_state_buffer = step_state_buffer[-1]
    #                     step_period_buffer = [step_period_buffer[-1]]
    #                     is_in_stand_phase = True
    #
    #         if not is_in_stand_phase:
    #             # axes[0].scatter(i, vx - driftVel[0], color='c', linewidth=1)
    #             vx_new = vx - driftVel[0]
    #             vy_new = vy - driftVel[1]
    #             vz_new = vz - driftVel[2]
    #         else:
    #             vx_new = 0
    #             vy_new = 0
    #             vz_new = 0
    #
    #         driftRateBuffer.append(np.sqrt(np.linalg.norm(driftRate)))
    #         foot_stationary_buffer.append(int(is_in_stand_phase))
    #         foot_accmag_list.append(mag_acc)
    #         knee_accmag_list.append(mag_acc_knee)
    #         foot_vx_list.append(vx)
    #         foot_vy_list.append(vy)
    #         foot_vz_list.append(vz)
    #         foot_vxzupt_list.append(vx_new)
    #         foot_vyzupt_list.append(vy_new)
    #         foot_vzzupt_list.append(vz_new)
    #         foot_vmagnew_list.append(np.sqrt(vx_new ** 2 + vy_new ** 2 + vz_new ** 2))
    #         foot_vmag_list.append(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
    #         px = foot_px_list[-1]
    #         py = foot_py_list[-1]
    #         pz = foot_pz_list[-1]
    #         px = px + foot_vx_list[-1] * dt
    #         py = py + foot_vy_list[-1] * dt
    #         pz = pz + foot_vz_list[-1] * dt
    #         foot_px_list.append(px)
    #         foot_py_list.append(py)
    #         foot_pz_list.append(pz)
    #         px = foot_pxzupt_list[-1]
    #         py = foot_pyzupt_list[-1]
    #         pz = foot_pzzupt_list[-1]
    #         px = px + foot_vxzupt_list[-1] * dt
    #         py = py + foot_vyzupt_list[-1] * dt
    #         pz = pz + foot_vzzupt_list[-1] * dt
    #         foot_pxzupt_list.append(px)
    #         foot_pyzupt_list.append(py)
    #         foot_pzzupt_list.append(pz)
    #
    # except KeyboardInterrupt:
    #     pass
    #
    # idx = np.arange(len(time_buffer))
    # axes[0].plot(idx, np.array(foot_stationary_buffer) * 1.5, label='stationary window')
    # axes[0].plot(idx, np.array(foot_accmag_list), label='mag of acc')
    #
    # # axes[0].plot(idx, np.array(knee_accmag_list), label="mag of acc knee")
    # axes[0].legend()
    # axes[1].plot(idx, np.array(foot_stationary_buffer) * 1.5, label='stationary window')
    # axes[1].plot(idx, np.array(foot_vx_list), label='vz before zupt')
    # axes[1].plot(idx, np.array(foot_vxzupt_list), label='vz after zupt')
    # # axes[1].plot(idx, np.array(foot_pz_list), label = 'pz before zupt')
    # # axes[1].plot(idx, np.array(foot_pzzupt_list), label='pz after zupt')
    # axes[1].legend()
    # plt.show()
