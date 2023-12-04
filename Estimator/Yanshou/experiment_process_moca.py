import sys

sys.path.append("/home/yuxuan/Project/MPV_2024/")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib.pyplot import MultipleLocator
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
idx_exp = 1
moca_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(idx_exp)
moca_file = moca_path + "Moca{}.mat".format(idx_exp)
moca_data = scio.loadmat(moca_file)["SA{}".format(idx_exp)]


# moca_data = np.load("/media/yuxuan/My Passport/VIO_Experiment/vsMoca/9/Moca9_smooth.npy")

def interp_interval_bilinear(data, idx):
    data_start = data[idx[0], :]
    data_end = data[idx[-1], :]
    x = np.array([idx[0], idx[-1]])
    xx = idx
    y = np.array([data_start, data_end])
    f = interpolate.interp1d(x, data[x, :], kind='linear', axis=0)
    data[idx] = f(xx)
    return data


def interp_interval_cubic(data, idx_shift, idx_smooth):
    x = np.hstack([idx_smooth[0], np.arange(idx_shift[0], idx_shift[1]), idx_smooth[1]])
    xx = np.arange(idx_smooth[0], idx_smooth[1])
    y = data[x, :]
    f = interpolate.interp1d(x, y, kind="linear", axis=0)
    data[xx] = f(xx)
    for i in range(3):
        data[xx, i] = gaussian_filter1d(data[xx, i], 3)
    return data


plot_3d = True
fig = plt.figure()

frames = moca_data[:, 0]
time = moca_data[:, 1]

idx_stair = np.arange(2, 2 + 3 * 8)
idx_cam = np.arange(26, 29)
idx_knee = np.arange(29, 32)
idx_ankle = np.arange(32, 35)
idx_heel = np.arange(35, 38)
idx_toe = np.arange(38, 41)

# 2-25
stairs = np.mean(moca_data[:, idx_stair], axis=0)
stairs_x = stairs[0::3]
stairs_y = stairs[1::3]
stairs_z = stairs[2::3]

# 26-28
cam = moca_data[:, idx_cam]

# 29-31
knee = moca_data[:, idx_knee]

# 32:34
ankle = moca_data[:, idx_ankle]
shift_idx = [980, 1080]
smooth_idx = [950, 1100]
# ankle[np.arange(shift_idx[0], shift_idx[1] + 1), 0] -= 200
# ankle[np.arange(shift_idx[0], shift_idx[1] + 1), 1] += 160
# ankle[np.arange(shift_idx[0], shift_idx[1] + 1), 2] -= 20
# ankle = interp_interval_cubic(ankle, shift_idx, smooth_idx)

# 35:37
heel = moca_data[:, idx_heel]

# 38:40
toe = moca_data[:, idx_toe]
# smooth_idx = np.arange(1140, 1485)
# toe = interp_interval(toe, smooth_idx)


if plot_3d:
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([500, 2000])
    ax.set_ylim([-750, 750])
    ax.set_zlim([200, 1700])
    ax.plot3D(stairs_x, stairs_y, stairs_z)
    ax.plot3D(cam[:, 0], cam[:, 1], cam[:, 2])
    ax.plot3D(knee[:, 0], knee[:, 1], knee[:, 2])
    ax.plot3D(ankle[:, 0], ankle[:, 1], ankle[:, 2])
    ax.plot3D(heel[:, 0], heel[:, 1], heel[:, 2])
    ax.plot3D(toe[:, 0], toe[:, 1], toe[:, 2])
else:
    ax = fig.add_subplot()
    diff_toe_y = np.gradient(toe[:, 1])
    diff_heel_y = np.gradient(heel[:, 1])
    # ax.plot(frames, np.abs(diff_toe_y))
    ax.plot(frames, ankle)
    ax.scatter(frames[smooth_idx], ankle[smooth_idx, 0])
    ax.scatter(frames[shift_idx], ankle[shift_idx, 0])

plt.show()

input("替换")
# moca_data[:, idx_toe] = toe
# moca_data[:, idx_heel] = heel
# moca_data[:, idx_ankle] = ankle
np.save(moca_path + "Moca{}_smooth.npy".format(idx_exp), moca_data)
