import cv2
import open3d as o3d
import numpy as np
from lcm_msg.pcd_lcm.pcd_xyz import *
from lcm_msg.lcm_init import *
import lcm
import time
from Environment import *
import PIL
import os

totaltimestep = 1000
pcd_data = np.zeros((38528, 3))


def pcd_handler(channel, data):
    global pcd_data
    msg = pcd_xyz.decode(data)
    pcd_data[:, 0] = np.array(msg.pcd_x)
    pcd_data[:, 1] = np.array(msg.pcd_y)
    pcd_data[:, 2] = np.array(msg.pcd_z)
    pcd_data = (pcd_data - 10000) / 300.0  # int16_t to float


env = Environment()

imu_buffer_path = "../Sensor/IM948/imu_buffer.npy"
data_save_path = "/media/yuxuan/Ubuntu 20.0/IMG_TEST/TEST8/"
img_list = os.listdir(data_save_path)
for f in img_list:
    os.remove(data_save_path + f)

if __name__ == "__main__":
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
            env.pcd_to_binary_image(pcd_data, eular_angle)
            cv2.imshow("binaryimage", env.elegant_img())
            # img_save = PIL.Image.fromarray(env.img_binary)
            # img_save_name = data_save_path + "{}.png".format(i)
            # img_save.save(img_save_name, bits=1, optimize=True)

            data_temp = np.zeros((13,))
            data_temp[0:12] = imu_buffer
            data_temp[12] = time.time() - t0
            imu_data = np.vstack([imu_data, data_temp])
            img_list.append(env.img_binary)
            pcd_2d_list.append(env.pcd_2d)
            key = cv2.waitKey(1)
            if key == ord('q'):
                np.save(data_save_path + "imu_data.npy", imu_data)
                print('Wait for Saving {} Pictures'.format(len(img_list)))
                for k in range(len(img_list)):
                    np.save(data_save_path + "{}.npy".format(k), img_list[k])
                    print("{} in {} img".format(k, len(img_list)))
                for k in range(len(pcd_2d_list)):
                    np.save(data_save_path + "{}_.npy".format(k), pcd_2d_list[k])
                    print("{} in {} pcd".format(k, len(pcd_2d_list)))
                break
    except KeyboardInterrupt:
        pass
