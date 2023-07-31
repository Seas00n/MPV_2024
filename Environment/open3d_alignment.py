import open3d as o3d
import copy
import matplotlib.pyplot as plt
import os
from my_feature import *


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


class open3d_alignment(object):
    def __init__(self, pcd_s, pcd_t):
        self.pcd_s2d = pcd_s
        self.pcd_t2d = pcd_t
        print(np.shape(pcd_s))
        print(np.shape(pcd_t))
        pcd_s3d = pcd2d_to_3d(self.pcd_s2d, num_rows=5)
        self.pcd_s = o3d.t.geometry.PointCloud()
        self.pcd_s.point['positions'] = o3d.core.Tensor(pcd_s3d)
        pcd_t3d = pcd2d_to_3d(self.pcd_t2d, num_rows=5)
        self.pcd_t = o3d.t.geometry.PointCloud()
        self.pcd_t.point['positions'] = o3d.core.Tensor(pcd_t3d)

    def draw_regis_result(self, transformation=None):
        if transformation is None:
            transformation = np.eye(4)
        source_temp = copy.deepcopy(self.pcd_s)
        target_temp = copy.deepcopy(self.pcd_t)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], width=1000, height=1000)

    def alignment(self, transformation=None):
        if transformation is None:
            transformation = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

        treg = o3d.t.pipelines.registration
        self.pcd_s.estimate_normals()
        self.pcd_t.estimate_normals()

        estimation = treg.TransformationEstimationPointToPoint()
        criteria = treg.ICPConvergenceCriteria(relative_fitness=1e-2,
                                               relative_rmse=1e-3,
                                               max_iteration=50)
        criteria_list = [
            treg.ICPConvergenceCriteria(relative_fitness=1e-2,
                                        relative_rmse=1e-3,
                                        max_iteration=20),
            treg.ICPConvergenceCriteria(1e-2, 1e-3, 15),
            treg.ICPConvergenceCriteria(1e-2, 1e-3, 10)
        ]
        voxel_size = 0.3
        voxel_sizes = o3d.utility.DoubleVector([0.5, 0.3, 0.1])

        max_correspondence_distance = 0.22
        max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.22, 0.01])

        # reg = treg.icp(source=self.pcd_s, target=self.pcd_t,
        #                max_correspondence_distance=max_correspondence_distance,
        #                estimation_method=estimation,
        #                criteria=criteria,
        #                voxel_size=voxel_size
        #                )
        reg = treg.multi_scale_icp(source=self.pcd_s, target=self.pcd_t,
                                   voxel_sizes=voxel_sizes,
                                   criteria_list=criteria_list,
                                   max_correspondence_distances=max_correspondence_distances,
                                   estimation_method=estimation
                                   )
        print("Fitness:{},RMSE:{}".format(reg.fitness, reg.inlier_rmse))
        return reg.transformation.numpy()


if __name__ == '__main__':
    data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST4_OPEN3D/"
    traj_x = np.load(data_save_path + "traj_x.npy")
    traj_y = np.load(data_save_path + "traj_y.npy")
    env_type = np.load(data_save_path + "env_type_buffer.npy")
    open3d_x = np.load("camera_x_buffer.npy")
    open3d_y = np.load("camera_y_buffer.npy")
    # plt.plot(traj_x, traj_y)
    # plt.plot(open3d_x, open3d_y)
    # plt.show()

    file_list = os.listdir(data_save_path)
    num_frames = len(file_list) - 3
    pcd_s = []
    pcd_t = []

    camera_dx_buffer = []
    camera_dy_buffer = []
    camera_x_buffer = []
    camera_y_buffer = []
    flag_buffer = []
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    plt.ion()
    ax.view_init(elev=10, azim=-20)
    xmove = 0
    ymove = 0
    trans = np.eye(4)
    pcd_to_align_new = np.zeros((0, 2))
    pcd_to_align_pre = np.zeros((0, 2))
    for i in range(num_frames):
        plt.cla()
        pcd_new = np.load(data_save_path + "{}_pcd2d.npy".format(i))
        fea_A, fea_B, fea_C = get_fea_sa(pcd_new)
        if i == 0:
            pcd_pre = pcd_new
            pcd_to_align_pre = pcd_to_align_new
            camera_dx_buffer.append(0)
            camera_dy_buffer.append(0)
            camera_x_buffer.append(0)
            camera_y_buffer.append(0)
            flag_buffer.append(1)
        else:
            if int(env_type[i]) == 1:
                reg = open3d_alignment(pcd_s=pcd_pre, pcd_t=pcd_new)
                vx = camera_dx_buffer[-1]
                vy = camera_dy_buffer[-1]
                trans_init = np.eye(4)
                trans_init[1, 3] = -vx
                trans_init[2, 3] = -vy
                try:
                    trans = reg.alignment(transformation=trans_init)
                    flag_buffer.append(1)
                except:
                    print('对齐失败')
                    flag_buffer.append(0)

                if flag_buffer[-1] == 1:
                    dx = -trans[1, 3]
                    dy = -trans[2, 3]
                    print(dx)
                    print(dy)
                    if abs(dx) < 0.1 and abs(dy) < 0.1:
                        xmove = dx
                        ymove = dy
                        camera_dx_buffer.append(xmove)
                        camera_dy_buffer.append(ymove)
                        print("对齐成功")
                    else:
                        print("+++++++++++++++++++++++++移动距离过大+++++++++++++++++++++++++++")
                        xmove_prev = camera_dx_buffer[-1]
                        ymove_prev = camera_dy_buffer[-1]
                        camera_dx_buffer.append(xmove_prev)
                        camera_dy_buffer.append(ymove_prev)
                else:
                    camera_dx_buffer.append(xmove)
                    camera_dy_buffer.append(ymove)
            else:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
            camera_x_buffer.append(camera_x_buffer[-1] + camera_dx_buffer[-1])
            camera_y_buffer.append(camera_y_buffer[-1] + camera_dy_buffer[-1])
            pcd3d_new = pcd2d_to_3d(pcd_new)
            pcd3d_pre = pcd2d_to_3d(pcd_pre)

            xx = pcd3d_new[:, 0]
            yy = pcd3d_new[:, 1] + camera_x_buffer[-1]
            zz = pcd3d_new[:, 2] + camera_y_buffer[-1]
            ax.plot3D(xx[0:-1:21],
                      yy[0:-1:21],
                      zz[0:-1:21],
                      '.:c', linewidth=2)
            xx = pcd3d_pre[:, 0]
            yy = pcd3d_pre[:, 1] + camera_x_buffer[-2]
            zz = pcd3d_pre[:, 2] + camera_y_buffer[-2]
            ax.plot3D(xx[0:-1:21],
                      yy[0:-1:21],
                      zz[0:-1:21],
                      '.:b', linewidth=1)

            xx = pcd3d_new[:, 0]
            yy = pcd3d_new[:, 1] + traj_x[i]
            zz = pcd3d_new[:, 2] + traj_y[i]
            ax.plot3D(xx[0:-1:21],
                      yy[0:-1:21],
                      zz[0:-1:21],
                      '.:y')
            xx = pcd3d_pre[:, 0]
            yy = pcd3d_pre[:, 1] + traj_x[i - 1]  # - traj_x[i]
            zz = pcd3d_pre[:, 2] + traj_y[i - 1]  # - traj_y[i]
            ax.plot3D(xx[0:-1:21],
                      yy[0:-1:21],
                      zz[0:-1:21],
                      '.:r', linewidth=1)
            ax.plot3D(np.zeros((len(camera_x_buffer, ))),
                      np.array(camera_x_buffer),
                      np.array(camera_y_buffer),
                      color='m')
            ax.plot3D(np.zeros((len(camera_x_buffer, ) - 1)),
                      traj_x[0:i],
                      traj_y[0:i],
                      color='r')

            pcd_pre = pcd_new
            plt.draw()
            plt.pause(0.01)

# np.save('camera_x_buffer.npy', np.array(camera_x_buffer))
# np.save('camera_y_buffer.npy', np.array(camera_y_buffer))
# print('')
