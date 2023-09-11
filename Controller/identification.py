import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as md
import cvxopt

ctrl_vec = np.load("mujoco_sim/data/ctrl.npy")
q_vec = np.load("mujoco_sim/data/q.npy")
qd_vec = np.load("mujoco_sim/data/qd.npy")
qdd_vec = np.load("mujoco_sim/data/qdd.npy")
time_vec = np.load("mujoco_sim/data/t.npy")
q_vec = q_vec[2000:, 2:5]
qd_vec = qd_vec[2000:, 2:5]
qdd_vec = qdd_vec[2000:, 2:5]
torque_vec = ctrl_vec[2000, :]
time_vec = time_vec[2000:]



