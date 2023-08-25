import os.path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class MyData(Dataset):
    def __init__(self, dataset_path):
        self.fea_path = os.path.join(dataset_path + "/fea/")
        self.voxel_path = os.path.join(dataset_path + "/voxel/")
        self.fea_npy_path = os.path.join(dataset_path + "/fea/fea_npy/")
        self.voxel_npy_path = os.path.join(dataset_path + "/voxel/voxel_npy/")
        self.num_fea = len(os.listdir(self.fea_npy_path))
        self.num_voxel = len(os.listdir(self.voxel_npy_path))

    def clean(self):
        for f in os.listdir(self.fea_npy_path):
            os.remove(self.fea_npy_path + f)
        for f in os.listdir(self.voxel_npy_path):
            os.remove(self.voxel_npy_path + f)
        self.num_fea = 0
        self.num_voxel = 0

    def save_new_fea_file(self, fea_file):
        fea_all = np.load(self.fea_path + fea_file)
        num_fea = np.shape(fea_all)[0]
        num_0 = self.num_fea
        for i in range(num_fea):
            np.save(self.fea_npy_path + "{}.npy".format(i + num_0), fea_all[i, :])
            self.num_fea += 1

    def save_new_voxel_file(self, voxel_file):
        voxel_all = np.load(self.voxel_path + voxel_file)
        num_voxel = np.shape(voxel_all)[0]
        num_0 = self.num_voxel
        for i in range(num_voxel):
            np.save(self.voxel_npy_path + "{}.npy".format(i + num_0), voxel_all[i, :])
            self.num_voxel += 1

    def resave_all(self):
        self.clean()
        for file in os.listdir(self.fea_path):
            if file[-4:] == ".npy":
                self.save_new_fea_file(file)
        for file in os.listdir(self.voxel_path):
            if file[-4:] == ".npy":
                self.save_new_voxel_file(file)

    def data_in_one(self, inputdata, data_min, data_max):
        outputdata = (inputdata - data_min) / (data_max - data_min)
        return outputdata

    def __getitem__(self, idx):
        fea_name = self.fea_npy_path + "{}.npy".format(idx)
        voxel_name = self.voxel_npy_path + "{}.npy".format(idx)
        fea_all = np.load(fea_name)
        fea = fea_all[1:]
        voxel = np.load(voxel_name)
        min_voxel_x = min(voxel[:, 0])
        max_voxel_x = max(voxel[:, 0])
        min_voxel_y = min(voxel[:, 1])
        max_voxel_y = max(voxel[:, 1])
        voxel[:, 0] = self.data_in_one(voxel[:, 0], min_voxel_x, max_voxel_x)
        voxel[:, 1] = self.data_in_one(voxel[:, 1], min_voxel_y, max_voxel_y)
        idx_not_zero = np.where(fea != 0)[0]
        fea[idx_not_zero[0::2]] = self.data_in_one(fea[idx_not_zero[0::2]], min_voxel_x, max_voxel_x)
        fea[idx_not_zero[1::2]] = self.data_in_one(fea[idx_not_zero[1::2]], min_voxel_y, max_voxel_y)
        fea = fea[0:6]
        return fea, voxel

    def __len__(self):
        return self.num_fea


if __name__ == "__main__":
    db = MyData(dataset_path="/media/yuxuan/SSD/ENV_Fea_Train")
    db.resave_all()