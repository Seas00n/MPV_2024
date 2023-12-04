import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

fig = plt.figure(figsize=(20, 10))
exp_list = [1, 4, 8, 9, 10, 11, 12, 14]
ax_list = [
    fig.add_subplot(421),
    fig.add_subplot(422),
    fig.add_subplot(423),
    fig.add_subplot(424),
    fig.add_subplot(425),
    fig.add_subplot(426),
    fig.add_subplot(427),
    fig.add_subplot(428)
]
save_path = "/media/yuxuan/My Passport/VIO_Experiment/Result/"

for i in range(len(exp_list)):
    ax = ax_list[i]
    exp_idx = exp_list[i]
    moca_data = scio.loadmat(save_path + "moca_final_{}.mat".format(exp_idx))
    vio_wyx_data = scio.loadmat(save_path + "vio_final_{}.mat".format(exp_idx))
    vio_cch_data = scio.loadmat(save_path + "vio_cch_final_{}.mat".format(exp_idx))
    line_moca = ax.plot(moca_data['cam_x'], moca_data['cam_y'])
    line_vio = ax.plot(vio_wyx_data['vio_x'], vio_wyx_data['vio_y'])
    line_cch = ax.plot(vio_cch_data['vio_x_cch'], vio_cch_data['vio_y_cch'])

plt.show()
