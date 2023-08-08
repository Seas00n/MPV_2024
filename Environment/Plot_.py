import cv2
import matplotlib.pyplot as plt
from Environment import Env_Type
import numpy as np


def add_type(img, env_type, id = 0):
    if env_type == Env_Type.Levelground:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    elif env_type == Env_Type.Upstair:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2)
    elif env_type == Env_Type.DownStair:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
    elif env_type == Env_Type.Upslope:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 2)
    elif env_type == Env_Type.Downslope:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
    else:
        cv2.putText(img, "{}".format(env_type), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    cv2.putText(img, "id:{}".format(id), (40, 100), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255),2)

def add_collision(ax, xc, yc, w, h, p=None):
    if p is None:
        p = [0, 0]
    ax.plot3D([-0.3, 0.3], [xc - w + p[0], xc - w + p[0]], [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1]], color='c',
              linewidth=1)
    ax.plot3D([-0.3, 0.3], [xc + p[0], xc + p[0]], [yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]], color='c', linewidth=1)
    ax.plot3D([-0.3, 0.3], [xc + p[0], xc + p[0]], [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1]], color='c', linewidth=1)
    ax.plot3D([-0.3, 0.3], [xc + w + p[0], xc + w + p[0]], [yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]], color='c',
              linewidth=1)
    ax.plot3D([0.3, 0.3, 0.3, 0.3], [xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]],
              [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1], yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]],
              color='c',
              linewidth=1)
    ax.plot3D([-0.3, -0.3, -0.3, -0.3], [xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]],
              [yc - 1.5 * h + p[1], yc - 1.5 * h + p[1], yc + 0.5 * h + p[1], yc + 0.5 * h + p[1]],
              color='c',
              linewidth=1)


def add_pcd3d(ax, pcd2d, camera_pos, linewidth=1, color='r', alpha=1):
    pcd3d = pcd2d_to_3d(pcd2d)
    xx = pcd3d[:, 0]
    yy = pcd3d[:, 1] + camera_pos[0]
    zz = pcd3d[:, 2] + camera_pos[1]
    ax.plot3D(xx[0:-1:51],
              yy[0:-1:51],
              zz[0:-1:51],
              linewidth=linewidth,
              color=color,
              alpha=alpha)


def add_camera_trajectory(ax, camera_x, camera_y, linewidth=1, color='r'):
    ax.plot3D(np.zeros(np.shape(camera_x)[0]),
              camera_x,
              camera_y,
              linewidth=linewidth,
              color=color)


def pcd2d_to_3d(pcd_2d, num_rows=5):
    num_points = np.shape(pcd_2d)[0]
    pcd_3d = np.zeros((num_points * num_rows, 3))
    pcd_3d[:, 1:] = np.repeat(pcd_2d, num_rows, axis=0)
    x = np.linspace(-0.2, 0.2, num_rows).reshape((-1, 1))
    xx = np.repeat(x, num_points, axis=1)
    # weights_diag = np.diag(np.linspace(0.0001, -0.0001, num_rows))
    weights_diag = np.diag(np.linspace(0, 0, num_rows))
    idx = np.arange(num_points)
    idx_m = np.repeat(idx.reshape((-1, 1)).T, num_rows, axis=0)
    xx = xx + np.matmul(weights_diag, idx_m)
    pcd_3d[:, 0] = np.reshape(xx.T, (-1,))
    return pcd_3d
