import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch.cuda

from lcm_msg.pcd_lcm.pcd_xyz import *
from lcm_msg.lcm_init import *
import lcm
import time
from Environment import *
import PIL
import os

totaltimestep = 50
pcd_data_int = np.zeros((38528, 3))
pcd_data_temp = np.zeros((38528, 3))


def pcd_handler(channel, data):
    global pcd_data, pcd_data_temp
    msg = pcd_xyz.decode(data)
    pcd_data_int[:, 0] = np.array(msg.pcd_x)
    pcd_data_int[:, 1] = np.array(msg.pcd_y)
    pcd_data_int[:, 2] = np.array(msg.pcd_z)
    pcd_data_temp = (pcd_data_int - 10000) / 300.0  # int16_t to float


env = Environment()

imu_buffer_path = "../Sensor/IM948/imu_buffer.npy"
data_save_path = "data/"
img_list = os.listdir(data_save_path)

imu_color = ['#D8383A', '#96C37D', '#2F7FC1']
camera_color = ['#FA7F6F', '#8ECFC9', '#82B0D2']
world_color = ['r', 'g', 'b']
body_color = ['#EF7A6D', '#63E398', '#9DC3E7']


def plot_coordinate(R, ax, length=0.5, linewidth=1, shift=[0, 0, 0], color=['r', 'g', 'b']):
    x_ = R[:, 0] * length
    y_ = R[:, 1] * length
    z_ = R[:, 2] * length
    ax.plot3D([shift[0], x_[0] + shift[0]], [shift[1], x_[1] + shift[1]], [shift[2], x_[2] + shift[2]], color=color[0],
              linewidth=linewidth)
    ax.plot3D([shift[0], y_[0] + shift[0]], [shift[1], y_[1] + shift[1]], [shift[2], y_[2] + shift[2]], color=color[1],
              linewidth=linewidth)
    ax.plot3D([shift[0], z_[0] + shift[0]], [shift[1], z_[1] + shift[1]], [shift[2], z_[2] + shift[2]], color=color[2],
              linewidth=linewidth)
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(-1.5, 0.5)
    ax.set_zlim(-0.5, 1.5)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')


def data_collect():
    global pcd_data_temp
    imu_buffer = np.memmap(imu_buffer_path, dtype='float32', mode='r',
                           shape=(12,))
    pcd_msg, pcd_lc = pcd_lcm_initialize()
    subscriber = pcd_lc.subscribe("PCD_DATA", pcd_handler)
    imu_data = np.zeros((13,))
    t0 = time.time()
    img_list = []
    pcd_2d_list = []
    try:
        for i in range(totaltimestep):
            pcd_lc.handle()
            eular_angle = [imu_buffer[6], imu_buffer[7], imu_buffer[8]]
            env.pcd_to_binary_image(pcd_data_temp, eular_angle)
            cv2.imshow("binaryimage", env.elegant_img())
            # img_save = PIL.Image.fromarray(env.img_binary)
            # img_save_name = data_save_path + "{}.png".format(i)
            # img_save.save(img_save_name, bits=1, optimize=True)
            data_temp = np.zeros((13,))
            data_temp[0:12] = imu_buffer
            data_temp[12] = time.time() - t0
            imu_data = np.vstack([imu_data, data_temp])
            img_list.append(env.img_binary)
            pcd_2d_list.append(pcd_data_temp)  # 每次的数据
            key = cv2.waitKey(1)
        np.save(data_save_path + "imu_data_.npy", imu_data)
        for k in range(len(pcd_2d_list)):
            np.save(data_save_path + "{}_.npy".format(k), pcd_2d_list[k])  # 最后存储
            print("{} in {} pcd".format(k, len(pcd_2d_list)))
    except KeyboardInterrupt:
        pass


def data_calibrate():
    data_save_path = "calibrate/"
    pcd = np.load(data_save_path + "44_.npy")
    imu_data = np.load(data_save_path + "imu_data_.npy")
    eular_angle = imu_data[45, :][6:9]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')

    # 相机坐标系下的点云数据
    ax.scatter3D(pcd[0:-1:30, 0], pcd[0:-1:30, 1], pcd[0:-1:30, 2], color='blue', linewidths=0.5)

    # 相机坐标系绘制
    X_camera = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    plot_coordinate(X_camera, ax)
    plt.show()

    # 考虑相机安装位置可以看到相机的z轴朝前，y向下，x向右
    # 相机坐标系在解剖坐标系下的朝向为
    R_body_camera = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    plot_coordinate(R_body_camera, ax)
    pcd_in_body = np.matmul(R_body_camera, pcd.T).T
    ax.scatter3D(pcd_in_body[0:-1:30, 0], pcd_in_body[0:-1:30, 1], pcd_in_body[0:-1:30, 2], color='blue',
                 linewidths=0.5)
    plt.show()
    # imu 在世界坐标系下的位姿
    R_world_imu = Rotation.from_euler('xyz', [eular_angle[0], eular_angle[1], eular_angle[2]], degrees=True).as_matrix()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    plot_coordinate(R_world_imu, ax, length=0.2, linewidth=2, shift=[0.02, 0.02, 0])
    # 考虑到安装情况，相机到IMU的坐标变换矩阵为
    R_imu_camera = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    # 从而得到相机坐标系在世界坐标系下的表示
    R_world_camera = np.matmul(R_world_imu, R_imu_camera)
    plot_coordinate(R_world_camera, ax, length=0.2, linewidth=2, shift=[-0.02, -0.02, 0])
    plt.show()
    # 解剖坐标系在世界坐标系下的表示为
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    R_world_body = np.matmul(R_body_camera, R_world_camera.T)
    plot_coordinate(R_world_body, ax)
    # 将点云从解剖坐标系变换到世界坐标系下，可以看到结果正确
    pcd_in_world = np.matmul(R_world_body, pcd_in_body.T).T
    ax.scatter3D(pcd_in_world[0:-1:30, 0], pcd_in_world[0:-1:30, 1], pcd_in_world[0:-1:30, 2], color='blue',
                 linewidths=0.5)
    plt.show()


if __name__ == "__main__":
    # data_collect()

    fig = plt.figure(figsize=(7, 10))

    ax = fig.add_subplot(3, 2, 1, projection='3d')
    data_save_path = "data/"
    pcd = np.load(data_save_path + "20_.npy")
    imu_data = np.load(data_save_path + "imu_data_.npy")
    eular_angle = imu_data[19, :][6:9]
    #################################################################
    # imu 在世界坐标系下的位姿
    X_world_imu = R_world_imu = Rotation.from_euler('xyz', [eular_angle[0], eular_angle[1], eular_angle[2]],
                                                    degrees=True).as_matrix()
    # camera 在世界坐标系下的位姿
    ## 考虑到安装情况，相机到IMU的坐标变换矩阵为
    R_imu_camera = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    X_world_camera = R_world_camera = np.matmul(R_world_imu, R_imu_camera)

    # body 在世界坐标系下的位姿
    R_body_imu = Rotation.from_euler('xyz', [eular_angle[0] - 90, 0, 180], degrees=True).as_matrix()
    X_world_body = R_world_body = np.matmul(R_world_imu, R_body_imu.T)
    # 世界坐标系
    X_world = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    # pcd在世界坐标系下的表示
    pcd_in_world = np.matmul(R_world_camera, pcd.T).T
    # 依次画出
    plot_coordinate(X_world_imu, ax, length=0.1, linewidth=2, shift=[-0.1, 0.1, 0.8], color=imu_color)
    plot_coordinate(X_world_camera, ax, length=0.1, linewidth=2, shift=[0.1, -0.1, 0.8], color=camera_color)
    plot_coordinate(X_world_body, ax, linewidth=3, length=1, color=body_color, shift=[0, 0, 0.8])
    plot_coordinate(X_world, ax, length=1.2, linewidth=1, color=world_color, shift=[0, 0, -0.5])
    ax.scatter3D(pcd_in_world[0:-1:20, 0], pcd_in_world[0:-1:20, 1], pcd_in_world[0:-1:20, 2],
                 c=100 * pcd_in_world[0:-1:20, 2], marker='*',
                 linewidths=1)
    ##################################################################
    ax2 = fig.add_subplot(3, 2, 2, projection='3d')
    X_body = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    R_body_camera = np.matmul(R_body_imu, R_imu_camera)
    pcd_in_body = np.matmul(R_body_camera, pcd.T).T
    plot_coordinate(X_body, ax2, length=1.2, linewidth=3, color=body_color, shift=[0, -0.8, 0])
    ax2.scatter3D(pcd_in_body[0:-1:20, 0], pcd_in_body[0:-1:20, 1], pcd_in_body[0:-1:20, 2],
                  c=-pcd_in_body[0:-1:20, 2], marker='*',
                  linewidths=1)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-0.5, 1.5)
    # 相机安装在机器左测，实际地形取相机偏右侧的部分对应机器正中间
    chosen_idx = np.logical_and(pcd[:, 0] < 0.05, pcd[:, 0] > 0.02)
    pcd_chosen = pcd[chosen_idx, :]
    pcd_chosen_in_body = np.matmul(R_body_camera, pcd_chosen.T).T
    ax2.plot3D(pcd_chosen_in_body[0:-1:2, 0], pcd_chosen_in_body[0:-1:2, 1] - 0.4, pcd_chosen_in_body[0:-1:2, 2],
               color='red',
               linewidth=2)

    #######################################################################
    ax3 = fig.add_subplot(3, 2, 3)
    # 选取0.1-2距离的点
    chosen_idx = np.logical_and(abs(pcd_chosen_in_body[:, 1]) < 2, pcd_chosen_in_body[:, 1] > 0.01)
    chosen_y = pcd_chosen_in_body[chosen_idx, 1]
    chosen_z = pcd_chosen_in_body[chosen_idx, 2]
    ax3.scatter(chosen_y, chosen_z)
    # 和z=0,y=1对齐
    y_max = np.max(chosen_y)
    z_min = np.min(chosen_z)
    chosen_y = chosen_y + (0.99 - y_max)
    chosen_z = chosen_z + (0.01 - z_min)
    # 只取出最前方1m^2内的点
    chosen_idx = np.logical_and(chosen_y > 0, chosen_z < 1)
    chosen_y = chosen_y[chosen_idx]
    chosen_z = chosen_z[chosen_idx]
    ax3.scatter(chosen_y, chosen_z, linewidth=0.1)
    ax3.plot([0.01, 0.99, 0.99], [0.01, 0.01, 0.99], color='b')
    ax3.plot([0.01, 0.01, 0.99], [0.01, 0.99, 0.99], color='b')
    # ax3.set_xlim([0, 1.5])
    # ax3.set_ylim([-0.5, 1])
    # ax3.axis('equal')
    pixel_y = np.floor(100 * chosen_y).astype('int')
    pixel_z = np.floor(100 * chosen_z).astype('int')
    img_binary = np.zeros((100, 100)).astype('uint8')
    for i in range(np.size(pixel_y)):
        img_binary[pixel_y[i], pixel_z[i]] = 255
    img_binary[:, 1] = np.flip(img_binary[:, 1], axis=0)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.imshow(img_binary, cmap='gray')
    ###################################################################
    ax5 = fig.add_subplot(3, 2, 5)
    chosen_idx = np.logical_and(abs(pcd_chosen_in_body[:, 1]) < 2, pcd_chosen_in_body[:, 1] > 0.01)
    pcd_2d = pcd_chosen_in_body[chosen_idx, 2:0:-1]
    pcd_2d[:, 1] = -pcd_2d[:, 1]
    # KNN均值滤波
    ax5.scatter(pcd_2d[:, 0], pcd_2d[:, 1], linewidth=0.2)
    ax5.set_xlim([0, 1.5])
    ax5.set_ylim([-1.5, 0])
    ax5.axis('equal')
    nb1 = NearestNeighbors(n_neighbors=20, algorithm='auto')
    nb1.fit(pcd_2d)
    _, idx = nb1.kneighbors(pcd_2d)
    pcd_thin = np.mean(pcd_2d[idx, :], axis=1)
    ymax = np.max(pcd_thin[:, 0])
    idx_chosen = pcd_thin[:, 0] > 0.1
    pcd_thin = pcd_thin[idx_chosen, :]
    idx_remove = np.where(ymax - pcd_thin[:, 0] < 0.02)[0]
    if len(idx_remove) < 10:
        print('remove')
        pcd_thin = np.delete(idx_remove, pcd_thin)
    ax5.plot(pcd_thin[:, 0], pcd_thin[:, 1], color='yellow')
    #################################################################
    classification_model = torch.load('/home/yuxuan/Project/CCH_Model/realworld_model_epoch_29.pt',
                                      map_location=torch.device('cpu'))
    print(classification_model)
    img_input = img_binary.reshape(1, 1, 100, 100).astype('uint8')
    img_input = torch.tensor(img_input, dtype=torch.float)
    with torch.no_grad():
        output = classification_model(img_input)
    pred = output.data.max(1)[1].cpu().numpy()
    print(pred)
    ##################################################################
    X = pcd_thin[:, 0]
    Y = pcd_thin[:, 1]
    if np.max(Y) - np.min(Y) < 0.1:
        print('点云高度差过小，当前地形应该是平地，检测错误')
    th = 0.05
    X0 = X[Y - np.min(Y) < 0.25].reshape((-1, 1))
    Y0 = Y[Y - np.min(Y) < 0.25].reshape((-1, 1))
    inlier_mask1, outlier_mask1, line_y_ransac1, line_X1 = RANSAC(X0, Y0, th)
    mean_y1 = np.mean(Y0[inlier_mask1])
    idx1 = np.where(abs(Y0 - mean_y1) < 0.008)[0]
    x1 = X0[idx1, :]
    y1 = Y0[idx1, :]
    ax5.plot(x1, y1, color='red')
    X1 = np.delete(X0, idx1).reshape((-1, 1))
    Y1 = np.delete(Y0, idx1).reshape((-1, 1))
    inlier_mask2, outlier_mask2, line_y_ransac2, line_X2 = RANSAC(X1, Y1, th)
    mean_y2 = np.mean(Y1[inlier_mask2])
    idx2 = np.where(abs(Y1 - mean_y2) < 0.008)[0]
    x2 = X1[idx2, :]
    y2 = Y1[idx2, :]
    ax5.plot(x2, y2, color='red')
    if mean_y1 > mean_y2:
        stair_high_x, stair_high_y = x1, y1
        stair_low_x, stair_low_y = x2, y2
    else:
        stair_high_x, stair_high_y = x2, y2
        stair_low_x, stair_low_y = x1, y1


    plt.show()
