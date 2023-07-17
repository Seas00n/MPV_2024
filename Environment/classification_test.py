import cv2
import open3d as o3d
import numpy as np
from lcm_msg.pcd_lcm.pcd_xyz import *
from lcm_msg.lcm_init import *
import lcm
import time
from Environment import *
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

if __name__ == "__main__":
    imu_buffer = np.memmap("../Sensor/IM948/imu_buffer.npy", dtype='float32', mode='r',
                           shape=(12,))
    pcd_msg, pcd_lc = pcd_lcm_initialize()
    subscriber = pcd_lc.subscribe("PCD_DATA", pcd_handler)
    try:
        for i in range(totaltimestep):
            pcd_lc.handle()
            imu_angle = [imu_buffer[6], imu_buffer[7], imu_buffer[8]]
            env.pcd_to_binary_image(pcd_data, imu_angle)
            cv2.imshow("binaryimage",env.img_binary)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
