from Environment import *
from feature_extra_new import *
from Utils.IO import fifo_data_vec
from Plot_ import *

data_save_path = "/media/yuxuan/SSD/IMG_TEST/TEST7/"

env = Environment()
env_type_buffer = []
camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
pcd_os_buffer = [[], []]
alignment_flag_buffer = [[], []]
env_paras_buffer = [[], []]

# 特征提取方法设置
use_method1 = True

# 重构参数设置
use_multi_frame_rebuild = False
num_frame_rebuild = 3
pcd_multi_build_buffer = []
for i in range(num_frame_rebuild):
    pcd_multi_build_buffer.append(np.zeros((0, 2)))

# 画图设置
plot_3d = False

if __name__ == "__main__":
    imu_data = np.load(data_save_path + "imu_data.npy")
    imu_data = imu_data[1:, :]
    idx_frame = np.arange(np.shape(imu_data)[0])

    fig = plt.figure(figsize=(5, 5))
    if plot_3d:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    plt.ion()

    for i in idx_frame:
        print("------------Frame[{}]-----------------".format(i))
        print("load binary image and pcd to process")

        env.img_binary = np.load(data_save_path + "{}_img.npy".format(i))
        env.pcd_2d = np.load(data_save_path + "{}_pcd.npy".format(i))
        env.thin()
        # env.pcd_thin = env.pcd_2d
        env.classification_from_img()
        img = cv2.cvtColor(env.elegant_img(), cv2.COLORMAP_RAINBOW)
        add_type(img, env_type=Env_Type(env.type_pred_from_nn), id=i)
        # cv2.namedWindow('binary', 0)
        # cv2.moveWindow("binary", 600, 100)
        # cv2.imshow("binary", img)
        # cv2.waitKey(1)
        plt.cla()
        env_type_buffer.append(env.type_pred_from_nn)
        if i == 0:
            camera_dx_buffer.append(0)
            camera_dy_buffer.append(0)
            env_type_buffer.append(env.type_pred_from_nn)
            pcd_os = pcd_opreator_system(pcd_new=env.pcd_thin)
            pcd_os.get_fea(_print_=True, ax=None, idx=i)
            pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_os)
        else:
            pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_opreator_system
            pcd_pre = pcd_pre_os.pcd_new
            pcd_new, pcd_new_os = env.pcd_thin, pcd_opreator_system(env.pcd_thin)
            pcd_new_os.get_fea(_print_=True, ax=None, idx=i)
            pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)
            if plot_3d:
                continue
            else:
                pcd_new_os.show_(ax, pcd_color='r', id=int(i))
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
            plt.draw()
            plt.pause(0.01)