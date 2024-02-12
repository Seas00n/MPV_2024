import datetime

import open3d as o3d
import copy
from Environment.backup.my_feature import *

def pcd2d_to_3d(pcd_2d, num_rows=5):
    num_points = np.shape(pcd_2d)[0]
    pcd_3d = np.zeros((num_points * num_rows, 3))
    pcd_3d[:, 1:] = np.repeat(pcd_2d, num_rows, axis=0)
    x = np.linspace(-0.2, 0.2, num_rows).reshape((-1, 1))
    xx = np.repeat(x, num_points, axis=1)
    # weights_diag = np.diag(np.linspace(1, 5, num_rows))
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
        pcd_s3d = pcd2d_to_3d(self.pcd_s2d, num_rows=3)
        self.pcd_s = o3d.geometry.PointCloud()
        self.pcd_s.points = o3d.utility.Vector3dVector(pcd_s3d)
        pcd_t3d = pcd2d_to_3d(self.pcd_t2d, num_rows=3)
        self.pcd_t = o3d.geometry.PointCloud()
        self.pcd_t.points = o3d.utility.Vector3dVector(pcd_t3d)

    def draw_regis_result(self, transformation=None):
        if transformation is None:
            transformation = np.eye(4)
        source_temp = copy.deepcopy(self.pcd_s)
        target_temp = copy.deepcopy(self.pcd_t)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], width=1000, height=1000)

    def alignment_new(self):
        t0 = datetime.datetime.now()
        trans_init = np.eye(4)
        reg = o3d.pipelines.registration.registration_icp(
            self.pcd_s, self.pcd_t, 1, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        print("Fitness:{},RMSE:{}".format(reg.fitness, reg.inlier_rmse))
        t1 = datetime.datetime.now()
        print("#=====FeatureAlignO3d:{}=====#".format(
            (t1 - t0).total_seconds() * 1000))
        return reg.transformation

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

        max_correspondence_distance = 0.3
        max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.3, 0.3])

        # reg = treg.icp(source=self.pcd_s, target=self.pcd_t,
        #                max_correspondence_distance=max_correspondence_distance,
        #                estimation_method=estimation,
        #                criteria=criteria,
        #                voxel_size=voxel_size)
        reg = treg.multi_scale_icp(source=self.pcd_s, target=self.pcd_t,
                                   voxel_sizes=voxel_sizes,
                                   criteria_list=criteria_list,
                                   max_correspondence_distances=max_correspondence_distances,
                                   estimation_method=estimation
                                   )
        print("Fitness:{},RMSE:{}".format(reg.fitness, reg.inlier_rmse))
        return reg.transformation.numpy()
