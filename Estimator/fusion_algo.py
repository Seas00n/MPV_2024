import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class StateKalmanFilter(object):
    def __init__(self):
        self.knee_pos = [0, 0]
        self.knee_vel = [0, 0]
        self.knee_angle = 0
        self.ankle_angle = 0
        self.phi = 0
        self.dt = 0
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
        self.Q = np.diag([0.1, 0.1, 0.05, 0.05, 0.05, 80, 1, 80, 100, 1, 80, 100])
        self.R = np.diag([0.1, 0.1, 0.01, 0.1, 0.1, 0.1])
        self.P0 = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # State vector [px, py, vx, ax, phi, dot_phi , qk, wk, ak, qa, wa, aa]
        self.kf = KalmanFilter(dim_x=12, dim_z=6)
        self.kf.x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.kf.F = self.transtion_matrix(self.phi, self.dt)
        self.kf.H = self.H
        self.kf.R = self.R

    def cal_phi(self):
        self.phi = math.atan2(self.knee_pos[1], self.knee_pos[0])

    def transtion_matrix(self, phi, dt):
        F = np.array([[1, 0, math.cos(phi) * dt, math.cos(phi) * 0.5 * dt ** 2, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, math.sin(phi) * dt, math.sin(phi) * 0.5 * dt ** 2, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, dt, 0.5 * dt ** 2, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0.5 * dt ** 2],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return F

    def prediction(self, dt):
        # state_vec: knee_px knee_py knee_vel knee_q ankle_q
        self.dt = dt
        self.kf.F = self.transtion_matrix(self.phi, dt)
        self.kf.predict()

    def update(self, state_measure):
        knee_pos = [state_measure[0], state_measure[1]]
        knee_vel = [state_measure[2], state_measure[3]]
        knee_angle = state_measure[4]
        ankle_angle = state_measure[5]
        phi = math.atan2(state_measure[2], state_measure[3])
        z = np.array([knee_pos[0], knee_pos[1],
                      math.sqrt(knee_vel[0] ** 2 + knee_vel[1] ** 2), phi,
                      knee_angle, ankle_angle])
        self.kf.update(z)
        state_vec = self.kf.x
        # State vector [px, py, vx, ax, phi, dot_phi , qk, wk, ak, qa, wa, aa]
        self.knee_pos = [state_vec[0], state_vec[1]]
        self.knee_angle = state_vec[6]
        self.knee_angle = state_vec[9]
        self.phi = state_vec[4]
        self.knee_vel = [state_vec[2], math.atan(self.phi)*state_vec[2]]


    def model_prediction(self, num, dt):
        F = self.transtion_matrix(self.phi, dt)
        prediction_state = np.zeros((num, 5))
        state = self.kf.x
        # State vector [px 0, py 1, vx 2, ax 3, phi 4, dot_phi 5 , qk 6, wk 7, ak 8, qa 9, wa, aa]
        for i in range(num):
            state = np.dot(F, state)
            prediction_state[i, :] = [state[0], state[1], state[6], state[9], state[4]]
        return prediction_state
