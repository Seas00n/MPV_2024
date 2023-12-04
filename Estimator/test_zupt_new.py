import time

from Utils.Algo import *
from Utils.IO import *

import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot

from scipy import signal
from matplotlib import pyplot as plt

from esekf import *

import sys

sys.path.append("/home/yuxuan/Project/MPV_2024/")

freq = 40

foot_vx_list = [0]
foot_vy_list = [0]
foot_vz_list = [0]
foot_vx_zupt_list = [0]
foot_vy_zupt_list = [0]
foot_vz_zupt_list = [0]
foot_px_list = [0]
foot_py_list = [0]
foot_pz_list = [0]
foot_px_zupt_list = [0]
foot_py_zupt_list = [0]
foot_pz_zupt_list = [0]
time_buffer = [0]
foot_accmag_list = [0]
foot_stationary_buffer = [1]

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


def load_imu_parameters(sigma_a_n, sigma_w_n, sigma_a_b, sigma_w_b):
    params = ImuParameters()
    params.frequency = 40
    # params.frequency = yml['IMU.frequency']
    # params.sigma_a_n = yml['IMU.acc_noise_sigma']  # m/sqrt(s^3)
    # params.sigma_w_n = yml['IMU.gyro_noise_sigma']  # rad/sqrt(s)
    # params.sigma_a_b = yml['IMU.acc_bias_sigma']  # m/sqrt(s^5)
    # params.sigma_w_b = yml['IMU.gyro_bias_sigma']  # rad/sqrt(s^3)
    params.sigma_a_n = sigma_a_n  # m/sqrt(s^3)
    params.sigma_w_n = sigma_w_n  # rad/sqrt(s)
    params.sigma_a_b = sigma_a_b  # m/sqrt(s^5)
    params.sigma_w_b = sigma_w_b  # rad/sqrt(s^3)
    return params


if __name__ == "__main__":

    fig, axes = plt.subplots(3, 1, sharex=False)
    imu_ankle_data_all = np.load("imu_ankle_zupt3.npy")
    imu_knee_data_all = np.load("imu_knee_zupt3.npy")
    imu_ankle_parameters = load_imu_parameters(sigma_a_n=4.3869000987044300e-03,
                                               sigma_w_n=2.1996304018440254e-02,
                                               sigma_a_b=3.4394710418253510e-04,
                                               sigma_w_b=6.7625705146351529e-04)
    imu_knee_parameters = load_imu_parameters(sigma_a_n=1.8890224776446796e-02,
                                              sigma_w_n=2.4505983414289048e-01,
                                              sigma_a_b=1.7919202879752155e-03,
                                              sigma_w_b=.7908881261317332e-03)

    count = 0

    # 前50数据求mean
    initial_count = 50
    imu_knee_initial_buffer = []
    imu_ankle_initial_buffer = []


    while count < initial_count:
        count += 1
        if count > 10:
            imu_ankle_data = imu_ankle_data_all[count, :]
            imu_knee_data = imu_knee_data_all[count, :]
            imu_knee_initial_buffer.append(imu_knee_data)
            imu_ankle_initial_buffer.append(imu_ankle_data)
    imu_knee_0 = np.mean(np.array(imu_knee_initial_buffer), axis=0)
    imu_ankle_0 = np.mean(np.array(imu_ankle_initial_buffer), axis=0)
    g_0 = np.linalg.norm(imu_ankle_0[1:4])

    init_nominal_state_ankle = np.zeros((19,))
    init_nominal_state_ankle[0:3] = 0  # init position
    init_nominal_state_ankle[3:7] = imu_ankle_initial_buffer[0][10:14]  # init quat
    init_nominal_state_ankle[7:10] = 0  # init vel
    init_nominal_state_ankle[10:13] = 0  # init ba
    init_nominal_state_ankle[13:16] = 0  # init bg
    init_nominal_state_ankle[16:19] = np.array([0, 0, -9.81])
    estimator_ankle = ESEKF(init_nominal_state=init_nominal_state_ankle,
                            imu_parameters=imu_ankle_parameters)

    sigma_measurement_p = 0.02
    sigma_measurement_q = 0.015
    sigma_measurement = np.eye(6)
    sigma_measurement[0:3, 0:3] *= sigma_measurement_p ** 2
    sigma_measurement[3:6, 3:6] *= sigma_measurement_q ** 2


    # ESEKF 暖机
    estimator_ankle.last_predict_time = imu_ankle_data_all[0][0]
    for i in range(len(imu_ankle_initial_buffer)):
        time = imu_ankle_initial_buffer[i][0]
        accx = imu_ankle_initial_buffer[i][1]
        accy = imu_ankle_initial_buffer[i][2]
        accz = imu_ankle_initial_buffer[i][3]
        gyrx = imu_ankle_initial_buffer[i][4] + 1e-8
        gyry = imu_ankle_initial_buffer[i][5] + 1e-8
        gyrz = imu_ankle_initial_buffer[i][6] + 1e-8
        estimator_ankle.predict(np.array([time, gyrx*np.pi/180, gyry*np.pi/180, gyrz*np.pi/180, accx, accy, accz]))
        gt_pose = np.zeros((7,))
        gt_pose[0:3] = 0  # init position
        gt_pose[3:] = imu_ankle_initial_buffer[i][10:]  # init quat
        estimator_ankle.update(gt_pose, sigma_measurement)
        frame_pos = estimator_ankle.nominal_state[0:3]
        frame_quat = estimator_ankle.nominal_state[3:7]
        frame_vel = estimator_ankle.nominal_state[7:10]
        axes[2].scatter(i, np.linalg.norm(imu_ankle_0[1:4]), color='g')
        axes[2].scatter(i, np.linalg.norm(imu_ankle_initial_buffer[i][1:4]), color='r')




    acc_mag_buffer = [0, 0, 0]
    acc_mag_lp_buffer = [0, 0, 0]
    acc_mag_knee_buffer = [0, 0, 0]
    acc_mag_knee_lp_buffer = [0, 0, 0]

    num_frame = len(imu_ankle_data_all) - count
    t0 = imu_ankle_data_all[count, 0]
    time_buffer[0] = t0
    is_in_stand_phase = True
    step_state_buffer = np.array([1])
    step_period_buffer = [np.array([t0, 0, 0, 0])]
    driftRate = np.array([0, 0, 0])

    try:
        for i in range(num_frame):
            print("----------------------------Frame[{}]------------------------".format(i))
            time = imu_ankle_data_all[count, 0]
            time_buffer.append(time-t0)
            accx = imu_ankle_data_all[count, 1]
            accy = imu_ankle_data_all[count, 2]
            accz = imu_ankle_data_all[count, 3]
            gyrx = imu_ankle_data_all[count, 4]
            gyry = imu_ankle_data_all[count, 5]
            gyrz = imu_ankle_data_all[count, 6]
            qw = imu_ankle_data_all[count, 10]
            qx = imu_ankle_data_all[count, 11]
            qy = imu_ankle_data_all[count, 12]
            qz = imu_ankle_data_all[count, 13]
            accx_knee = imu_knee_data_all[count, 1]
            accy_knee = imu_knee_data_all[count, 2]
            accz_knee = imu_knee_data_all[count, 3]
            count += 1

            dt = time_buffer[-1] - time_buffer[-2]

            mag_acc = np.abs(np.sqrt(accx ** 2 + accy ** 2 + accz ** 2)-g_0)
            low_freq = 12

            if i == 0:
                acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
                acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
                acc_mag_buffer = fifo_data_vec(acc_mag_buffer, mag_acc)
            mag_acc, acc_mag_buffer, acc_mag_lp_buffer = low_pass_filter(mag_acc, acc_mag_buffer, acc_mag_lp_buffer,
                                                                         low_freq=low_freq, freq=freq)



            is_stationary = mag_acc < 0.2
            step_state_buffer = np.append(step_state_buffer, int(is_stationary))

            estimator_ankle.predict(np.array([time, gyrx*np.pi/180, gyry*np.pi/180, gyrz*np.pi/180, accx, accy, accz]))


            frame_pos_without_update = estimator_ankle.nominal_state[0:3]
            frame_quat_without_update = estimator_ankle.nominal_state[3:7]
            frame_vel_without_update = estimator_ankle.nominal_state[7:10]

            step_period_buffer.append(np.array([time_buffer[-1],
                                                frame_vel_without_update[0],
                                                frame_vel_without_update[1],
                                                frame_vel_without_update[2]]))

            if is_in_stand_phase:
                idx_stationaryEnd = np.where(np.diff(step_state_buffer) == -1)[0]
                if np.shape(idx_stationaryEnd)[0] > 0:
                    if idx_stationaryEnd[-1] > 10:
                        print("Switch to Swing")
                        idx_stationaryEnd = idx_stationaryEnd[-1]
                        step_state_buffer = step_state_buffer[-1]
                        step_period_buffer = [step_period_buffer[-1]]
                        is_in_stand_phase = False
                if is_in_stand_phase:
                    gt_pose = np.zeros((7,))
                    gt_pose[0:3] = np.array([foot_px_zupt_list[-1], foot_py_zupt_list[-1], foot_pz_zupt_list[-1]])  # init position
                    gt_pose[3:] = np.array([qw, qx, qy, qz])  # init quat
                    estimator_ankle.update(gt_pose, sigma_measurement)
            else:
                idx_stationaryStart = np.where(np.diff(step_state_buffer) == 1)[0]
                driftVel = (len(step_state_buffer) - 1) * driftRate
                if np.shape(idx_stationaryStart)[0] > 0:
                    if idx_stationaryStart[-1] > 10:
                        print("Switch to Stand")
                        idx_stationaryStart = idx_stationaryStart[-1]
                        driftVel = step_period_buffer[idx_stationaryStart][1:]  # - step_period_buffer[0][1:]
                        driftRate = driftVel / idx_stationaryStart
                        # axes[1].scatter(i, np.linalg.norm(step_period_buffer[idx_stationaryStart][1:]), color='r')
                        # axes[1].scatter(i - idx_stationaryStart, np.linalg.norm(step_period_buffer[0][1:]), color='g')
                        axes[1].scatter(i, step_period_buffer[idx_stationaryStart][3], color='r')
                        axes[1].scatter(i - idx_stationaryStart, step_period_buffer[0][3], color='g')
                        step_state_buffer = step_state_buffer[-1]
                        step_period_buffer = [step_period_buffer[-1]]
                        is_in_stand_phase = True

            frame_pos = estimator_ankle.nominal_state[0:3]
            frame_quat = estimator_ankle.nominal_state[3:7]
            frame_vel = estimator_ankle.nominal_state[7:10]

            foot_vx_list.append(frame_vel_without_update[0])
            foot_vy_list.append(frame_vel_without_update[1])
            foot_vz_list.append(frame_vel_without_update[2])
            foot_vx_zupt_list.append(frame_vel[0])
            foot_vy_zupt_list.append(frame_vel[1])
            foot_vz_zupt_list.append(frame_vel[2])
            foot_px_list.append(frame_pos_without_update[0])
            foot_py_list.append(frame_pos_without_update[1])
            foot_pz_list.append(frame_pos_without_update[2])
            foot_px_zupt_list.append(frame_pos[0])
            foot_py_zupt_list.append(frame_pos[1])
            foot_pz_zupt_list.append(frame_pos[2])
            foot_accmag_list.append(mag_acc)
            foot_stationary_buffer.append(int(is_in_stand_phase))
    except KeyboardInterrupt:
        pass
    idx = np.arange(len(time_buffer))
    axes[0].plot(idx, np.array(foot_stationary_buffer) * 1.5, label='stationary window')
    axes[0].plot(idx, np.array(foot_accmag_list), label='mag of acc')
    axes[0].legend()
    axes[1].plot(idx, np.array(foot_stationary_buffer) * 1.5, label='stationary window')
    axes[1].plot(idx, np.array(foot_vz_list), label='vz before zupt')
    axes[1].plot(idx, np.array(foot_vz_zupt_list), label='vz after zupt')
    axes[1].legend()
    plt.show()