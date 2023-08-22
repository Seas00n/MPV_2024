import numpy as np
import numpy.linalg as la
# import transformations as tr
import math
import time
from scipy import interpolate


class ImuParameters:
    def __init__(self):
        # self.sigma_a_n = 6.87e-05     # acc noise.   m/(s*sqrt(s)), continuous noise sigma
        # self.sigma_a_b = 2.18e-02  # acc bias     m/sqrt(s^5), continuous bias sigma
        self.sigma_a_n = 0.001221  # m/sqrt(s^3)
        self.sigma_a_b = 0.000048


class ESEKF(object):
    def __init__(self, init_nominal_state: np.array, imu_parameters: ImuParameters):
        """
        :param init_nominal_state: [ p, q, v, a_b, w_b, g ], a 19x1 or 1x19 vector
        :param imu_parameters: imu parameters
        """
        self.nominal_state = init_nominal_state[:8]
        self.imu_parameters = imu_parameters

        self.eular = init_nominal_state[8:11]

        # initialize noise covariance matrix
        self.noise_covar = np.zeros((4, 4))
        # assume the noises (especially sigma_a_n) are isotropic so that we can precompute self.noise_covar and save it.

        self.noise_covar[0:2, 0:2] = (imu_parameters.sigma_a_n ** 2) * np.eye(2)  # a只需要两个
        self.noise_covar[2:4, 2:4] = (imu_parameters.sigma_a_b ** 2) * np.eye(2)
        # self.noise_covar[0,0] = (imu_parameters.sigma_ax_n**2)
        # self.noise_covar[1,1] = (imu_parameters.sigma_az_n**2)
        # self.noise_covar[2,2] = (imu_parameters.sigma_ax_b**2)  #a只需要两个
        # self.noise_covar[3,3] = (imu_parameters.sigma_az_b**2)

        self.error_covar = np.zeros([8, 8])
        # ----------------------------------------------------------------------
        positionCov = 0.0001
        velocityCov = 0.1
        accelBiasCov = 0.001
        gravityCov = 100

        self.error_covar[0:2, 0:2] = positionCov * np.eye(2)
        self.error_covar[2:4, 2:4] = velocityCov * np.eye(2)
        self.error_covar[4:6, 4:6] = accelBiasCov * np.eye(2)
        self.error_covar[6:8, 6:8] = gravityCov * np.eye(2)
        # ----------------------------------------------------------------------
        self.last_predict_time = 0
        # ZUPT------------------

        self.station_window = np.zeros(7)
        self.station_flag = 0
        self.sta_flag1 = 0

        self.P = np.zeros((4, 4))

        self.p_window = np.zeros((20, 2))
        self.v_window = np.zeros((20, 2))

        self.num = 0

    def predict(self, imu_measurement: np.array, t):
        """
        :param imu_measurement: [t, w_m, a_m]
        :return:
        """
        # if self.last_predict_time == t:
        #     return
        # we predict error_covar first, because __predict_nominal_state will change the nominal state.
        self.__predict_error_covar(imu_measurement, t)
        self.__predict_nominal_state(imu_measurement, t)
        self.station_detect(imu_measurement)

        # self.ZUPT(imu_measurement,t)
        self.last_predict_time = t  # update timestamp

    def station_detect(self, imu_measurement):
        acc_stationary_threshold_H = 11
        acc_stationary_threshold_L = 9
        gyro_stationary_threshold = 0.15
        acc_s = imu_measurement[3:6]
        gyro_s = imu_measurement[6:9]

        acc_mag = np.sqrt(acc_s[0] ** 2 + acc_s[1] ** 2 + acc_s[2] ** 2)
        gyro_mag = np.sqrt((gyro_s[0]) ** 2 + (gyro_s[1]) ** 2 + (gyro_s[2]) ** 2)

        # plt.plot(time_vec,acc_mag)
        # plt.show()
        # plt.plot(time_vec,gyro_mag)
        # plt.show()
        stationary_acc_H = (acc_mag < acc_stationary_threshold_H)
        stationary_acc_L = (acc_mag > acc_stationary_threshold_L)
        stationary_acc = np.logical_and(stationary_acc_H, stationary_acc_L)  # C1
        stationary_gyro = (gyro_mag < gyro_stationary_threshold)  # C2
        stationary = np.logical_and(stationary_acc, stationary_gyro)
        self.station_window = fifo_data_vec(self.station_window, stationary)
        if np.sum(self.station_window) > 3:
            self.station_flag = 1
        else:
            self.station_flag = 0

    def ZUPT(self, imu_measurement, t):
        p = self.nominal_state[0:2].reshape(-1, 1)
        v = self.nominal_state[2:4].reshape(-1, 1)
        dt = t - self.last_predict_time
        # 0.6,0.1
        sigma_a = 1
        sigma_v = 0.05
        R = np.diag([sigma_v, sigma_v]) ** 2

        F = np.eye(4)
        F[0:2, 2:4] = np.eye(2) * dt

        H = np.block([
            [np.zeros((2, 2)), np.eye(2)],
        ])

        Q = (np.diag([0, 0, sigma_a, sigma_a]) * dt) ** 2
        self.P = F @ self.P @ F.T + Q

        if self.station_flag:
            K = self.P @ H.T @ la.inv(H @ self.P @ H.T + R)
            delta_x = K @ v
            self.P = (np.eye((4)) - K @ H) @ self.P
            pos_error = delta_x[0:2]  # [:,np.newaxis]
            vel_error = delta_x[2:4]
            # print('-------------------')
            # print(t)
            # print(pos_error)
            # print(vel_error)

            if self.sta_flag1 == 0:
                print('pause')
                self.p_pause = p
                self.v_pause = np.zeros(2)
                self.sta_flag1 = 1
            p = self.p_pause
            v = self.v_pause

            # if self.sta_flag1 == 0:
            #
            #     # self.p_des = p-pos_error
            #     # self.v_des = v-vel_error
            #     p_now = p
            #     v_now = v
            #     self.p_window = np.linspace(p_now, p - pos_error,30)
            #     self.v_window = np.linspace(v_now, v - vel_error,30)
            #     print(p - pos_error)
            #
            #     self.sta_flag1 = 1
            #
            # if self.num<30:
            #     p = self.p_window[self.num,:]
            #     v = self.v_window[self.num,:]
            #
            #     self.num += 1
            # else:
            #     p = p-pos_error
            #     v = v-vel_error

            # p = self.p_pause
            # v = self.v_pause

            # p = p-pos_error
            # v = v-vel_error
            self.nominal_state[:2] = p.reshape(2, )
            self.nominal_state[2:4] = v.reshape(2, )
        else:
            self.sta_flag1 = 0
            self.num = 0

    def __predict_nominal_state(self, imu_measurement: np.array, t):

        p = self.nominal_state[0:2].reshape(-1, 1)
        v = self.nominal_state[2:4].reshape(-1, 1)
        a_b = self.nominal_state[4:6].copy().reshape(-1, 1)
        g = self.nominal_state[6:8].copy().reshape(-1, 1)
        dt = t - self.last_predict_time  # --------------------------------
        a_b1 = np.zeros([3, 1])

        angle_last = self.eular
        angle = imu_measurement[0:3]
        # angle[0] = -angle[0]

        R = rotation_matrix(angle_last)
        R_half_next = rotation_matrix((angle_last + angle) / 2)
        R_next = rotation_matrix(angle)

        a_m = np.array([imu_measurement[3:6]]).reshape(-1, 1).copy()
        a_b1[1:3] = a_b
        a_m -= a_b1

        # use RK4 method to integration velocity and position.
        # integrate velocity first.
        a_k4 = (R_next @ a_m)[1:3] + g
        a_k1 = (R @ a_m)[1:3] + g
        a_k2 = (R_half_next @ a_m)[1:3] + g
        # v_k3 = R_half_next @ a_m + g  # yes. v_k2 = v_k3.
        a_k3 = a_k2

        #
        # #
        # v_next = v + dt * (a_k1 + 2 * a_k2 + 2 * a_k3 + a_k4) / 6
        # v_half = (v_next+v)/2
        #
        # v_k1 = v
        # v_k2 = v_half + 0.5 * dt * a_k1  # k2 = v_next_half = v + 0.5 * dt * v' = v + 0.5 * dt * v_k1(evaluate at t0)
        # v_k3 = v_half + 0.5 * dt * a_k2  # v_k2 is evaluated at t0 + 0.5*delta
        # v_k4 = v_next + 1 * dt * a_k3
        # p_next = p + dt * (v_k1 + 2 * v_k2 + 2 * v_k3 + v_k4) / 6
        v_next = v + dt * a_k4
        p_next = p + dt * v + + 0.5 * a_k4 * (dt ** 2)

        self.nominal_state[:2] = p_next.reshape(2, )
        self.nominal_state[2:4] = v_next.reshape(2, )
        self.eular = angle
        self.a_m_pre = a_m
        # print('pnext',p_next)

    def __predict_error_covar(self, imu_measurement: np.array, t):

        a_m = np.array([imu_measurement[3], imu_measurement[5]]).reshape(-1, 1).copy()
        a_b = self.nominal_state[4:6].reshape(-1, 1)

        angle = self.eular

        G = np.zeros((8, 4))
        G[2:4, 0:2] = np.eye(2)  # -np.eye(2)
        G[4:6, 2:4] = np.eye(2)

        R = rotation_matrix(angle)[1:3, 1:3]

        # use 3rd-order truncated integration to compute transition matrix Phi.

        dt = t - self.last_predict_time

        # F = np.zeros((8, 8))
        # F[0:2, 2:4] = np.eye(2)
        # F[4:6, 6:8] = -R
        #
        # Fdt = F * dt
        # Fdt2 = Fdt @ Fdt
        # Fdt3 = Fdt2 @ Fdt
        # Phi = np.eye(8) + Fdt + 0.5 * Fdt2 + (1. / 6.) * Fdt3

        Fdt = np.zeros((8, 8))
        Fdt[0:2, 2:4] = np.eye(2) * dt
        Fdt[2:4, 4:6] = -R * dt
        Fdt[2:4, 6:8] = np.eye(2) * dt
        Phi = np.eye(8) + Fdt

        """
        use trapezoidal integration to integrate noise covariance:
          Qd = 0.5 * dt * (Phi @ self.noise_covar @ Phi.T + self.noise_covar)
          self.error_covar = Phi @ self.error_covar @ Phi.T + Qd

        operations above can be merged to the below for efficiency.
        """

        # Qc_dt = 0.5*dt*G @ self.noise_covar @ G.T   #
        # self.error_covar = Phi @ (self.error_covar + Qc_dt) @ Phi.T + Qc_dt

        Qc_dt = G @ self.noise_covar @ G.T

        Qc_dt = dt * Qc_dt  #
        Qc_dt[2, 2] *= dt
        Qc_dt[3, 3] *= dt
        self.error_covar = Phi @ self.error_covar @ Phi.T + Qc_dt
        # self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

    def update(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        :param gt_measurement: [p, q], a 7x1 or 1x7 vector
        :param measurement_covar: a 6x6 symmetrical matrix = diag{sigma_p^2, sigma_theta^2}
        :return:
        ground_truth - nominal_state = delta = H @ error_state + noise
        """
        H = np.zeros((2, 8))
        H[0:2, 0:2] = np.eye(2)
        # H[2:4, 2:4] = np.eye(2)

        PHt = self.error_covar @ H.T  # 18x6

        # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
        K = PHt @ la.inv(H @ PHt + measurement_covar)  # 18x6

        # print('self.error_covar', self.error_covar)

        # update error covariance matrix
        self.error_covar = (np.eye(8) - K @ H) @ self.error_covar
        # force the error_covar to be a symmetrical matrix
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        # compute the measurements according to the nominal state and ground-truth state.

        gt_p = gt_measurement[0:2]
        # print('ground truth',gt_p)

        delta = np.zeros(2)
        delta[0:2] = gt_p - self.nominal_state[0:2]
        # compute state errors.
        errors = K @ delta
        # inject errors to the nominal state
        self.nominal_state[0:2] += errors[0:2]  # update position
        self.nominal_state[2:] += errors[2:]  # update the rest.

        """
        reset errors to zero and modify the error covariance matrix.
        we do nothing to the errors since we do not save them.
        but we need to modify the error_covar according to P = GPG^T
        """

        G = np.eye(8)
        self.error_covar = G @ self.error_covar @ G.T


def fifo_data_vec(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat


def rotation_matrix(eular):
    eular = eular * np.pi / 180

    Rz = np.array([np.cos(eular[2]), -np.sin(eular[2]), 0, np.sin(eular[2]), np.cos(eular[2]), 0, 0, 0, 1]).reshape(
        3, 3)
    Ry = np.array([np.cos(eular[1]), 0, np.sin(eular[1]), 0, 1, 0, -np.sin(eular[1]), 0, np.cos(eular[1])]).reshape(
        3, 3)
    Rx = np.array([1, 0, 0, 0, np.cos(eular[0]), -np.sin(eular[0]), 0, np.sin(eular[0]), np.cos(eular[0])]).reshape(
        3, 3)

    R = Rz.dot(Ry)
    R = R.dot(Rx)

    return R

    # def __update_legacy(self, gt_measurement: np.array, measurement_covar: np.array):
    #     """
    #     An old implementation of the updating procedure.
    #     :param gt_measurement: [p, q], a 7x1 or 1x7 vector
    #     :param measurement_covar: a 7x7 symmetrical matrix
    #     :return:
    #     """
    #     """
    #      Hx = dh/dx = [[I, 0, 0, 0, 0, 0]
    #                    [0, I, 0, 0, 0, 0]]
    #     """
    #     Hx = np.zeros((7, 19))
    #     Hx[0:3, 0:3] = np.eye(3)
    #     Hx[3:7, 3:7] = np.eye(4)
    #
    #     """
    #      X = dx/d(delta_x) = [[I_3, 0, 0],
    #                           [0, Q_d_theta, 0],
    #                           [0, 0, I_12]
    #     """
    #     X = np.zeros((19, 18))
    #     q = self.nominal_state[3:7]
    #     X[0:3, 0:3] = np.eye(3)
    #     X[3:7, 3:6] = 0.5 * np.array([[-q[1], -q[2], -q[3]],
    #                                   [q[0], -q[3], q[2]],
    #                                   [q[3], q[0], -q[1]],
    #                                   [-q[2], q[1], q[0]]])
    #     X[7:19, 6:18] = np.eye(12)
    #
    #     H = Hx @ X                      # 7x18
    #     PHt = self.error_covar @ H.T    # 18x7
    #     # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
    #     K = PHt @ la.inv(H @ PHt + measurement_covar)
    #
    #     # update error covariance matrix
    #     self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
    #     # force the error_covar to be a symmetrical matrix
    #     self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)
    #
    #     """
    #     compute errors.
    #     restrict quaternions in measurement and state have positive real parts.
    #     this is necessary for errors computation since we subtract quaternions directly.
    #     """
    #     if gt_measurement[3] < 0:
    #         gt_measurement[3:7] *= -1
    #     # NOTE: subtracting quaternion directly is tricky. that's why we abandon this implementation.
    #     errors = K @ (gt_measurement.reshape(-1, 1) - Hx @ self.nominal_state.reshape(-1, 1))
    #
    #     # inject errors to the nominal state
    #     self.nominal_state[0:3] += errors[0:3, 0]  # update position
    #     dq = tr.quaternion_about_axis(la.norm(errors[3:6, 0]), errors[3:6, 0])
    #     # print(dq)
    #     self.nominal_state[3:7] = tr.quaternion_multiply(q, dq)  # update rotation
    #     self.nominal_state[3:7] /= la.norm(self.nominal_state[3:7])
    #     if self.nominal_state[3] < 0:
    #         self.nominal_state[3:7] *= 1
    #     self.nominal_state[7:] += errors[6:, 0]  # update the rest.
    #
    #     """
    #     reset errors to zero and modify the error covariance matrix.
    #     we do nothing to the errors since we do not save them.
    #     but we need to modify the error_covar according to P = GPG^T
    #     """
    #     G = np.eye(18)
    #     G[3:6, 3:6] = np.eye(3) - tr.skew_matrix(0.5 * errors[3:6, 0])
    #     self.error_covar = G @ self.error_covar @ G.T

# A = np.memmap('six_force_data.npy', dtype='float64', mode='w+', shape=(3,))
# t = np.linspace(1,100,10000)
# for i in range(10000):
#     a = np.sin(t[i])
#     b = np.cos(t[i])
#     c = np.tan(t[i])
#     A[:] = [a,b,c]
#     time.sleep(0.01)
#     print(A[:])

