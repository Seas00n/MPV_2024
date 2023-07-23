# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:25:37 2022

@author: cxxha
"""

import time
# import cv2
import numpy as np
import torch
from scipy import stats
from Utils import Plot, IO, Motor
import open3d as o3d
import torchvision.transforms as transforms
from PIL import Image
import math
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import pandas as pd
from Environment import *
from LegKinematics import *
# import roboticstoolbox as rtb
# from spatialmath.base import *
# from scipy.spatial.distance import pdist
import argparse
from Robot import *
import scipy.io as scio

np.set_printoptions(formatter={'float':'{:0.2f}'.format})

##argument parse
parser = argparse.ArgumentParser()
parser.add_argument('--calibrate', '-d', type=int, default=0)  # data_source 0:real measurement; 1:real measurment on human without force sensor; 2:dataset with force sensor
parser.add_argument('--vision', '-v', type=int, default=1)
parser.add_argument('--control_motor', '-c', type=int, default=1)
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()
vision_flag = args.vision
data_source = args.data
control_motor = args.control_motor
verbose = args.verbose
plot_in_realtime = 0
cam_on_knee = 1

swing_stance_flag = np.memmap('log/swing_stance_flag.npy', dtype='float32', mode='w+', shape=(1,))

##initialization
environment = Environment();
leg_state = LegState();
prosthesis = Robot();
q_e_mat = np.array(
    [[[5, 25, 80, 5], [0, 0, 6, -2]], [[8, 2, 80, 60], [0, 0, 6, -2]], [[45, 60, 80, 2], [-5, -15, 6, -2]],
     [[8, 35, 80, 2], [-5, -15, 6, -2]], [[8, 35, 80, 2], [-5, -15, 6, -2]]])
k_mat = np.array([[[90, 90, 20, 20], [100, 100, 20, 20]], [[80, 80, 10, 10], [100, 100, 20, 20]],
                  [[80, 80, 10, 10], [100, 100, 20, 20]], [[40, 40, 10, 10], [50, 50, 20, 20]],
                  [[40, 40, 10, 10], [50, 50, 20, 20]]])
b_mat = np.array([[[5, 5, 0.5, 1], [6, 6, 1, 1]], [[5, 5, 0.5, 1], [6, 6, 1, 1]], [[5, 5, 0.5, 1], [6, 6, 1, 1]],
                  [[5, 5, 0.5, 1], [6, 6, 1, 1]], [[5, 5, 0.5, 1], [6, 6, 1, 1]]])

# IO.make_video('calibrate/video_frames')
def main():
    '''Initialize state vector'''

    if data_source == 0:
        imu_data_buf = np.memmap('log/imu_euler_acc.npy', dtype='float64', mode='r', shape=(9+9*cam_on_knee,))  # thigh imu
        six_force_buf = np.memmap('log/six_force_data.npy', dtype='float64', mode='r', shape=(6,))  # force_npy
        env_type_buf = np.memmap('log/environment_type.npy', dtype='int8', mode='r', shape=(1,))
        env_fea_buf = np.memmap('log/environment_feature.npy', dtype='float32', mode='r', shape=(4,))
    if data_source == 1:
        imu_data_buf = np.memmap('log/imu_euler_acc.npy', dtype='float64', mode='r', shape=(9+9*cam_on_knee,))  # thigh shank foot imu
    #     six_force_buf = np.memmap('log/six_force_data.npy', dtype='float64', mode='r', shape=(6,))  # force_npy
    if (data_source != 2) and (vision_flag == 1):
        pcd_buf = np.memmap('log/depth.npy', dtype='float32', mode='r', shape=(38528, 3))
    # if data_source == 2:
    #     imu_data_mat = np.load('../calibrate/sim_data/imu.npy')
    #     force_mat = np.load('../calibrate/sim_data/force.npy')
    #     pcd_mat = scio.loadmat('../calibrate/sim_data/pcd.mat')["pcd_data_new"].reshape(-1)
    #     imu_data_buf = imu_data_mat[0, :]
    #     knee_angle_init = imu_data_buf[9]
    #     ankle_angle_init = imu_data_buf[18]
    
    if control_motor:
        ser = Motor.init_motor()
        initialize_prosthesis(ser)
        time.sleep(1e-2)  # sleep 10 ms
        Motor.send_signals_to_motor(ser,np.array([0,0,0,0,60,1,60,1]))
        q_qv, i_t = Motor.read_signals_from_motor(ser)
           
    leg_state.thigh_angle_init, yaw_init = initialize_q_thigh(imu_data_buf, leg_state.thigh_angle_buf)
    leg_state.thigh_angle_buf -= leg_state.thigh_angle_init
        
    flag_prev = 0  # stance_swing_flag initialized,stance

    # q_init = initialize_q(imu_data_buf) # thigh, shank, ankle, read the initial joint angles: human thigh angle, knee-centered frame theta1 and theta2
    xc = 0
    yc = 0
    w = 0
    h = 0
    xmove = 0
    ymove = 0
    q_d_mat = np.zeros((2, 2))
    six_force_one = np.zeros(6)
    six_force_data = np.zeros((5, 6))
    environment_type = 0
    environment_feature = np.zeros(4)
    time_step_num = 15
    time_vec = ["time"]
    q_thigh_vec = ["q_thigh"]
    ax_vec = ["thigh a_x"]
    ay_vec = ["thigh a_y"]
    az_vec = ["thigh a_z"]
    q_knee_vec = ["q_knee"]
    q_ankle_vec = ["q_ankle"]
    q_ankle_d_vec = ["q_ankle_d"]
    q_knee_d_vec = ["q_knee_d"]
    fz_vec = ["f_z"]
    my_vec = ["M_y"]
    flexion_flag_vec = ["flexion"]
    i_vec = ["current"]
    F_stance_swing = 20
    pcd = np.zeros((3000, 3))
    imu_data = np.zeros((9+9*cam_on_knee,))
    
    My_push = -10
    stance12_time = 0.4
    
    '''Main loop'''
    
    start_time = time.time()
    
    q_qv_d = np.array([0,0,0,0,70,1,70,1])
    
    # plt.ion()
    # plt.figure(1)
    if data_source == 0:
        for force_n in range(len(six_force_data)):
            six_force_one[:] = six_force_buf[:]
            six_force_data[force_n, :] = six_force_one[:]
            time.sleep(0.0005)
    #print(six_force_data)
    t_s = time.time()
    stance0_start_time = time.time()
    q_knee = (prosthesis.current_joint_angle[0]+leg_state.thigh_angle)/np.pi*180.
    q_ankle = prosthesis.current_joint_angle[1]/np.pi*180.
    
    while time.time()-t_s < 30:
    #for i in range(1, time_step_num):
        swing_stance_flag = leg_state.swing_stance_flag
        print('environment type: {}'.format(environment.type_pred_from_nn))
        plt.clf()
        # try:
        if data_source == 0:
            imu_data[:] = imu_data_buf[:]
            six_force_one[:] = six_force_buf[:]
            six_force_data = fifo_mat(six_force_data, six_force_one)
            Fz_max = np.median(six_force_data, axis=0)[2]
            Fz_min = np.median(six_force_data, axis=0)[2]
            My_min = np.median(six_force_data, axis=0)[4]
            fz_vec.append(Fz_max)
            my_vec.append(My_min)
            #print("fz={},My={}".format(Fz_max,My_min))
            
            
        elif data_source == 1:
            imu_data[:] = imu_data_buf[:]
            
        else:
            imu_data = imu_data_mat[i, :]
            pcd = pcd_mat[i]
            pcd = pcd[pcd[:, 1] < -0.35]
            force = force_mat[i, 2]
        if vision_flag == 1:
            pcd = pcd_buf[np.all(pcd_buf != 0, axis=1)]
            pcd = pcd[0:len(pcd):1, :]
            #environment.type_pred_from_nn = env_type_buf[0]
            environment.features[:] = env_fea_buf[:]
            environment.pcd_prev = environment.pcd_2d
            if data_source == 2:
                environment.pcd_2d = pcd
            else:
                if plot_in_realtime:
                    img, environment.pcd_2d = pcd_to_binary_image(pcd, imu_data[0:9])
                    environment.pcd_2d = thin(environment.pcd_2d)
                #cv2.imshow('binary image', img)
                #cv2.waitKey(1)

        t = time.time() - start_time
        q_thigh_raw, knee_acc_x, ay = read_q_acc(imu_data, leg_state.thigh_angle_init, yaw_init)
        leg_state.thigh_angle_buf = IO.fifo_data_vec(leg_state.thigh_angle_buf, q_thigh_raw)
        # a_x, a_y, a_z = cal_IMUacc(imu_data,yaw_init)
        leg_state.thigh_angle = np.mean(leg_state.thigh_angle_buf) / 180. * np.pi
        q_thigh_vec.append(leg_state.thigh_angle * 180. / np.pi)
        
        
        
        
        # ax_vec.append(a_x)
        # ay_vec.append(a_y)
        # az_vec.append(a_z)
        time_vec.append(time.time())

        #prosthesis.model.base = transl(np.sin(leg_state.thigh_angle)*0.6,0,1.1-np.cos(leg_state.thigh_angle)*0.6)

        flag_prev = leg_state.swing_stance_flag  # 0
        prosthesis.desired_joint_angle_prev = prosthesis.desired_joint_angle
        
        
        #icp = icp_aliment(environment)
        
        ##[1] STANCE
        if leg_state.swing_stance_flag == 0:
            if verbose:
                print("Stance!")
            
            
            if data_source == 0 or data_source == 2:
                # TODO
                # stance1
                if prosthesis.stance_start == 0:
                    print('stance0')
                    stance_01_control_step(environment.type_pred_from_nn,stance0_start_time,q_knee,q_ankle)
                if prosthesis.stance_start == 1:
                    print('stance1')
                    stance_1_control_step(environment.type_pred_from_nn)
                if prosthesis.stance_start == 1 and My_min < My_push and abs(prosthesis.current_joint_angle[1] - prosthesis.desired_thetaeq[1]) < prosthesis.desired_thetabias:
                    prosthesis.stance_start = 2
                    prosthesis.stance12_start = 1
                # stance1 to 2
                if leg_state.swing_stance_flag-prosthesis.stance_start == -2 and prosthesis.stance12_start == 1:
                    stance2_start_time=time.time()
                    prosthesis.stance12_start = 12 # transition
                    print('stance1-2')
                if prosthesis.stance12_start == 12:
                    stance_12_control_step(environment.type_pred_from_nn,stance2_start_time) 
                # stance2
                if prosthesis.stance_start == 2 and prosthesis.stance12_start == 0:
                    print('stance2')
                    stance_2_control_step(environment.type_pred_from_nn)
                # stance_control_step(My_min, environment.type_pred_from_nn)
                q_qv_d = cal_stance_control_command()
                #print(Fz_max)
                if Fz_max > F_stance_swing :  # stance to swing
                    print('stance-swing')
                    leg_state.swing_stance_flag = 1
                    
            if data_source == 1:
                if prosthesis.desired_joint_angle[0] < - leg_state.thigh_angle:
                    prosthesis.desired_joint_angle[0] = - leg_state.thigh_angle   
                if prosthesis.desired_joint_angle[1] < -20./180*np.pi:
                    prosthesis.desired_joint_angle[1] = -20./180*np.pi
                ###FOR SWING TEST!!!!
                #q_qv_d = np.array([(prosthesis.desired_joint_angle[0]+leg_state.thigh_angle)*180/np.pi,0,0,0,70,5,30,1])
                q_qv_d = cal_stance_control_command()
                print(q_qv_d)
                foot_acc = np.linalg.norm(imu_data[-6:],ord=None)-9.8
                foot_gyr = np.linalg.norm(imu_data[-3:],ord=None)
                leg_state.update_swing_stance_flag_imu(foot_acc, foot_gyr)

        ##[2] STANCE 2 SWING
        if leg_state.swing_stance_flag - flag_prev == 1:  # stance to swing
            if verbose:
                print("Switch to Swing!")
            if vision_flag == 1:
               if data_source == 2:
                   img = recorded_2dPCD_to_binary(environment.pcd_2d)
                   environment.classification_from_img(img)
               # else:
               #     environment.classification_from_img(img)  ## only classify terrian before switch to swing
               #     environment.type_pred_from_nn = environment_type
            environment.type_pred_from_nn = 1
            
            leg_state.flexion_flag = 1
            max_thigh_angle = leg_state.thigh_angle
            # if environment.type_pred_from_nn == 2:
            #     try:
            #         xc, yc, w, h = environment.get_sd(environment.pcd_2d)
            #     except:
            #         print("use previous stair features")
            # if environment.type_pred_from_nn == 5:
            #     try:
            #         xc, yc, w, h = get_ob_easy(environment.pcd_2d)
            #         environment.collision_box = [[np.array([xc, yc - h / 2]), 0.02, h], [np.array([xc+w, yc - h / 2]), 0.02, h]]
            #     except:
            #         print("use previous stair features")
                        
            # if environment.type_pred_from_nn == 1:
            #     try:
            #         xc, yc, w, h = environment.get_sa(environment.pcd_2d)
            #     except:
            #         print("use previous stair features")
            # environment.features = np.array([xc,yc,w,h])
            
            
                    
        ##[3] SWING
        if leg_state.swing_stance_flag == 1:
           
            if leg_state.thigh_angle > max_thigh_angle+0.01 :
                max_thigh_angle = leg_state.thigh_angle
            
            if leg_state.thigh_angle < max_thigh_angle-0.04:
                leg_state.flexion_flag = 0
            #elif leg_state.thigh_angle > 0.166 * np.pi:
            #    leg_state.flexion_flag = 0
            swing_control_step(leg_state.flexion_flag,xmove,ymove)
            q_qv_d = cal_swing_control_command(time_vec)
            if data_source == 0 or data_source == 2:
                # TODO
                print("Fz_min={}".format(Fz_min))
                if Fz_min < F_stance_swing:  # swing to stance
                    leg_state.swing_stance_flag = 0
            if data_source == 1:
                foot_acc = np.linalg.norm(imu_data[-6:],ord=None)-9.8
                foot_gyr = np.linalg.norm(imu_data[-3:],ord=None)
                leg_state.update_swing_stance_flag_imu(foot_acc, foot_gyr)
                
        ##[4] SWING 2 STANCE
        if leg_state.swing_stance_flag - flag_prev == -1:  # swing to stance
            if verbose:
                print("Switch to Stance!")
            prosthesis.stance12 = 1
            prosthesis.stance_start = 0
            stance0_start_time=time.time()
            q_knee = (prosthesis.current_joint_angle[0]+leg_state.thigh_angle)/np.pi*180.
            q_ankle = prosthesis.current_joint_angle[1]/np.pi*180.
            
            leg_state.flexion_flag = 1
            
            
            # qqq, iii = Motor.read_signals_from_motor(ser)
            # stance2_start_time = time.time()
            # while time.time() - stance2_start_time < stance12_time:
            #     prosthesis.desired_thetaeq[0] = qqq[0] + \
            #                                     (q_e_mat[environment.type_pred_from_nn][0][0] -
            #                                      qqq[0]) * (1.0+0.5*(1.0/(1.0+math.exp(-((time.time()-stance2_start_time)*100-10)))))
            #     prosthesis.desired_thetaeq[1] = qqq[2] + \
            #                                     (q_e_mat[environment.type_pred_from_nn][1][0] -
            #                                      qqq[2]) * (1.0+0.5*(1.0/(1.0+math.exp(-((time.time()-stance2_start_time)*100-10)))))
            #     q_qv_d = cal_stance_control_command()
            #     Motor.send_signals_to_motor(ser, q_qv_d)
            #     time.sleep(0.02)

        # prosthesis.desired_knee_angle_buffer = IO.fifo_data_vec(prosthesis.desired_knee_angle_buffer,prosthesis.desired_joint_angle[0])
        # prosthesis.desired_joint_angle[0] = np.mean(prosthesis.desired_knee_angle_buffer)  

        q_ankle_d_vec.append(q_qv_d[2])
        q_knee_d_vec.append(q_qv_d[0])
        flexion_flag_vec.append(leg_state.flexion_flag)
        
        
            # print('error: {}, desired: {}; actual: {}'.format(q_qv - q_qv_d, q_qv_d, q_qv))  
        
        #[SAFETY!!!!]
        if q_qv_d[0] >= 100.:
            q_qv_d[0] = 100.
            #raise ValueError('Desired knee angle exceed 100 degree! Unexpected value!')
        if q_qv_d[0] < -5.:
            raise ValueError('Desired knee angle is below -5 degree! Unexpected value!')
        if q_qv_d[2] >= 80.:
            raise ValueError('Desired ankle angle exceed 80 degree! Unexpected value!')
        if q_qv_d[2] <= -50.:
            raise ValueError('Desired ankle angle is below -50 degree! Unexpected value!')
        
        #[Move!!!]
        if control_motor == 0:
            prosthesis.current_joint_angle_v = prosthesis.desired_joint_angle - prosthesis.current_joint_angle
            prosthesis.current_joint_angle = prosthesis.desired_joint_angle
            q_knee_vec.append(prosthesis.current_joint_angle[0]*180/np.pi)
            q_ankle_vec.append(prosthesis.current_joint_angle[1]*180/np.pi)
            # time.sleep(0.02)
        else:
            # q_qv_d = np.zeros(4)
            #q_qv_d *= 180 / np.pi
            #Motor.send_signals_to_motor(ser, np.array([0,0,0,0,70,1,70,1]))
            Motor.send_signals_to_motor(ser,q_qv_d)
            time.sleep(0.02)
            q_qv,i_t = Motor.read_signals_from_motor(ser)
            print(q_qv)
            if leg_state.swing_stance_flag == 1:
                prosthesis.current_joint_angle_v = prosthesis.desired_joint_angle - prosthesis.current_joint_angle
                prosthesis.current_joint_angle[0] = prosthesis.desired_joint_angle[0]
                prosthesis.current_joint_angle[1] = prosthesis.desired_joint_angle[1]
            if leg_state.swing_stance_flag == 0:
                prosthesis.current_joint_angle[0] = q_qv[0]/180.*np.pi-leg_state.thigh_angle
                prosthesis.current_joint_angle[1] = q_qv[2]/180.*np.pi
                prosthesis.current_joint_angle_v = prosthesis.current_joint_angle-prosthesis.prev_joint_angle
            prosthesis.current_joint_angle_a = prosthesis.current_joint_angle_v-prosthesis.prev_joint_angle_v
            prosthesis.prev_joint_angle = prosthesis.current_joint_angle
            prosthesis.prev_joint_angle_v = prosthesis.current_joint_angle_v
            prosthesis.prev_joint_angle_a = prosthesis.current_joint_angle_a
            print(prosthesis.current_joint_angle[0]-q_qv[0]/180.*np.pi+leg_state.thigh_angle)
              # *np.pi/180
            q_knee_vec.append(q_qv[0])
            q_ankle_vec.append(q_qv[2])
            i_vec.append(i_t[0])
            #print(q_qv)
        prosthesis.linkage = prosthesis.cal_joint_position(prosthesis.current_joint_angle)
        
        if plot_in_realtime:
            ax = plt.subplot(3, 2, 1)
            Plot.watch(time_vec, q_thigh_vec)
            if leg_state.swing_stance_flag == 0:
                state_text = plt.figtext(0, 0, "Stance", fontsize=10)
            elif leg_state.flexion_flag:
                state_text = plt.figtext(0, 0, "Flexion", fontsize=10)
            else:
                state_text = plt.figtext(0, 0, "Extension", fontsize=10)
        
            ax = plt.subplot(3, 2, 2)
            # Plot.watch(time_vec,ax_vec)
            # ax = plt.subplot(2,2,3)
            # Plot.watch(time_vec,ay_vec)
            # ax = plt.subplot(2,2,4)
            plt.xlim(-1, +1)
            plt.ylim(-0.8, +1.2)
        
            plt.plot([prosthesis.linkage[0, 0], prosthesis.linkage[0, 0] - np.sin(leg_state.thigh_angle) * 0.6],
                      [prosthesis.linkage[0, 2], prosthesis.linkage[0, 2] + np.cos(leg_state.thigh_angle) * 0.6])
            for n in range(3):
                plt.plot([prosthesis.linkage[n, 0], prosthesis.linkage[n + 1, 0]],
                          [prosthesis.linkage[n, 2], prosthesis.linkage[n + 1, 2]])
        #
            plt.scatter(environment.pcd_2d[0:len(environment.pcd_2d):20, 0],
                        environment.pcd_2d[0:len(environment.pcd_2d):20, 1], s=2,c='gray')
            if environment.type_pred_from_nn == 5:
                [xc,yc,w,h] = environment.features
                plt.plot([xc,xc,xc+w,xc+w],[yc-h,yc,yc,yc-h],color='red')
                Plot.plot_collision_box(environment.collision_box)
            if environment.type_pred_from_nn == 1:
                [xc,yc,w,h] = environment.features
                plt.plot([xc,xc,xc+w,xc+w,xc+2*w,xc+2*w],[yc-h,yc,yc,yc+h,yc+h,yc+2*h],color='red')
                Plot.plot_collision_box(environment.collision_box)
        #     # if environment.type_pred_from_nn == 1:
        #     #     plt.plot([xc,xc,xc+w,xc+w],[yc-h,yc,yc,yc+h],color='red')
        #     # plt.scatter(xmove,ymove)
        #     # Plot.watch(time_vec,az_vec)
        #     # prosthesis.plot(q=np.array([0,0.2,0.5,-3*np.pi/4]))
            #ax = plt.subplot(3, 2, 3)
            #Plot.watch(time_vec, i_vec)
            #ax = plt.subplot(3, 2, 3)
            #Plot.watch(time_vec, q_knee_vec)
            #ax = plt.subplot(3, 2, 4)
            #Plot.watch(time_vec, q_ankle_vec)
            ax = plt.subplot(3, 2, 5)
#            Plot.watch(time_vec,fz_vec)
            ax = plt.subplot(3, 2, 6)
 #           Plot.watch(time_vec,my_vec)
            plt.pause(0.0001)
        
            # plt.clf()
            Artist.remove(state_text)
    np.savez('data_0420_8', time_vec[1:],q_thigh_vec[1:],q_knee_vec[1:],q_ankle_vec[1:],q_ankle_d_vec[1:],q_knee_d_vec[1:],i_vec[1:])
    plt.figure(2)
    
    plt.plot(time_vec[1:],q_thigh_vec[1:],linewidth=1.5,label = 'Thigh angle')
    plt.plot(time_vec[1:],q_knee_vec[1:],linewidth=1.5,label = 'Knee angle')
    plt.plot(time_vec[1:],q_ankle_vec[1:],linewidth=1.5,label = 'Ankle angle')
    plt.plot(time_vec[1:],q_ankle_d_vec[1:],linewidth=1.5,label = 'Ankle desired angle')
    plt.plot(time_vec[1:],q_knee_d_vec[1:],linewidth=1.5,label = 'knee desired angle')
    #plt.plot(time_vec[1:],i_vec[1:],linewidth=1.5,label = 'knee_current')
    plt.legend()
    plt.show()
    #time_vec = np.array(time_vec[1:])
    
    #np.savetxt('ex_data.txt',np.colume)
    # plt.plot(time_vec,q_knee_vec)
    # plt.plot(time_vec,q_ankle_vec)
    
    #
    #     else:
    #         time.sleep(8e-3)  # sleep 8 ms
    #     if i % 100 == 0:
    #         calibrate = {'q_qv_d_mat': q_qv_d_mat, 'q_qv_mat': q_qv_mat,
    #                 'q_thigh_vec': q_thigh_vec}
    # #         np.save('results/benchtop_test_{}.npy'.format(time.time()), calibrate)
    # #     # except:
    # #     #     print('-----------Main loop error!-------------')
    # # if control_motor:
    # #     ser.close()
    # # view_trajectory(q_qv_d_mat, q_qv_mat, phase_save_vec, q_thigh_vec)


def swing_control_step(flexion_flag,xmove,ymove):
    if verbose:
        print("swing")
    if vision_flag == 1:
        # TODO: camera_position_estimation()
        [xc,yc,w,h] = environment.features
        if environment.type_pred_from_nn == 1:
            environment.collision_box = [[np.array([xc - w, yc - 3 * h / 2]), 0.02, h], [np.array([xc, yc - h / 2]), 0.02, h],
                                 [np.array([xc + w, yc + h / 2]), 0.02, h]]
        if environment.type_pred_from_nn == 5:
            environment.collision_box = [[np.array([xc, yc - h / 2]), 0.02, h], [np.array([xc+w, yc - h / 2]), 0.02, h]]
        #environment.pcd_2d += np.array([np.cos(leg_state.thigh_angle)*0.03-np.sin(leg_state.thigh_angle)*0.07+prosthesis.model.base.t[0],np.cos(leg_state.thigh_angle)*0.07+np.sin(leg_state.thigh_angle)*0.03+prosthesis.model.base.t[2]])
        #xc+=np.cos(leg_state.thigh_angle)*0.03-np.sin(leg_state.thigh_angle)*0.07+prosthesis.model.base.t[0]
        #yc+=np.cos(leg_state.thigh_angle)*0.07+np.sin(leg_state.thigh_angle)*0.03+prosthesis.model.base.t[2]
        
        # try:
        #     xmove_temp, ymove_temp = alignment(thin(environment.pcd_prev), thin(environment.pcd_2d))
        # except:
        #     xmove_temp = 0
        #     ymove_temp = 0
        # xmove+=xmove_temp
        # ymove+=ymove_temp
        # print(xmove,ymove)
        # prosthesis.model.base = transl(xmove,0,ymove)
    # else:
    #     xc = 0
    #     yc = 0
    #     w = 0
    #     h = 0
    #     if environment.type_pred_from_nn == 1:
    #         try:
    #             xc, yc, w, h = environment.get_sa(environment.pcd_2d)
    #         except:
    #             print("use previous stair features")
        
                # if vision_flag == 0:
        #environment.collision_box = [[np.array([xc - w, yc - 3 * h / 2]), 0.02, h], [np.array([xc, yc - h / 2]), 0.02, h],
         #                        [np.array([xc + w, yc + h / 2]), 0.02, h]]
    #prosthesis.desired_joint_angle = cal_desired_joint_angle_random_search(prosthesis, environment, leg_state.thigh_angle,
     #                                                        flexion_flag)
   
    prosthesis.desired_joint_angle = cal_desired_joint_angle_quasi_newton(prosthesis, environment, leg_state.thigh_angle,
                                                             flexion_flag)
   #prosthesis.desired_joint_angle[0] *= -1
    if prosthesis.desired_joint_angle[0] < - leg_state.thigh_angle:
        prosthesis.desired_joint_angle[0] = - leg_state.thigh_angle
    if prosthesis.desired_joint_angle[1] < -20./180*np.pi:
        prosthesis.desired_joint_angle[1] = -20./180*np.pi    

def stance_control_step(My_min, environment_flag):
    # if verbose:
    #     print("stance")
    # prosthesis.desired_joint_angle = prosthesis.desired_joint_angle + prosthesis.current_joint_angle_v*0.5
    if environment_flag == 5:
        environment_flag = 0
    My_push = -8
    if prosthesis.stance_start == 1:
        if My_min < My_push and abs(prosthesis.current_joint_angle[1] - prosthesis.desired_thetaeq[1]) < prosthesis.desired_thetabias:
            prosthesis.desired_k[:] = [k_mat[environment_flag][0][1], k_mat[environment_flag][1][1]]
            prosthesis.desired_b[:] = [b_mat[environment_flag][0][1], b_mat[environment_flag][1][1]]
            prosthesis.desired_thetaeq[:] = [q_e_mat[environment_flag][0][1], q_e_mat[environment_flag][1][1]]
            prosthesis.stance_start = 0
            #print(prosthesis.current_joint_angle)
            print('enter pushoff stage')
        else:
            print(abs(prosthesis.current_joint_angle[1] - prosthesis.desired_thetaeq[1]))
            prosthesis.desired_k = [k_mat[environment_flag][0][0], k_mat[environment_flag][1][0]]
            prosthesis.desired_b = [b_mat[environment_flag][0][0], b_mat[environment_flag][1][0]]
            prosthesis.desired_thetaeq = [q_e_mat[environment_flag][0][0], q_e_mat[environment_flag][1][0]]
    if prosthesis.desired_joint_angle[0] < - leg_state.thigh_angle:
        prosthesis.desired_joint_angle[0] = - leg_state.thigh_angle
    if prosthesis.desired_joint_angle[1] < -20./180*np.pi:
        prosthesis.desired_joint_angle[1] = -20./180*np.pi
    
    # My_push = -8
    # stance12_time = 0.3
    # if My_min < My_push and abs(prosthesis.current_joint_angle[1] - prosthesis.desired_thetaeq[1]) < prosthesis.desired_thetabias:
    #     prosthesis.stance_start = 0
    # else:
    #     prosthesis.stance_start = 1
    # # stance1
    # if prosthesis.stance_start == 1:
    #     stance_1_control_step(environment.type_pred_from_nn)
    # # stance1 to 2
    # if leg_state.swing_stance_flag-prosthesis.stance_start == 0 & prosthesis.stance12 == 1:
    #     stance2_start_time=time.time()
    #     while time.time()-stance2_start_time < stance12_time:
    #         prosthesis.desired_thetaeq[0] = q_e_mat[environment.type_pred_from_nn][0][0] + \
    #                                       (q_e_mat[environment.type_pred_from_nn][0][1]-q_e_mat[environment.type_pred_from_nn][0][0])*math.sin((time.time()-stance2_start_time)*np.pi/2/stance12_time)
    #         prosthesis.desired_thetaeq[1] = q_e_mat[environment.type_pred_from_nn][1][0] + \
    #                                       (q_e_mat[environment.type_pred_from_nn][1][1]-q_e_mat[environment.type_pred_from_nn][1][0])*math.sin((time.time()-stance2_start_time)*np.pi/2/stance12_time)
    #         q_qv_d = cal_stance_control_command()
    #         Motor.send_signals_to_motor(ser, q_qv_d)
    #     prosthesis.stance12 = 0
    # # stance2
    # if prosthesis.stance_start == 0:
    #     stance_2_control_step(environment.type_pred_from_nn)
        
    # if prosthesis.desired_joint_angle[0] < - leg_state.thigh_angle:
    #     prosthesis.desired_joint_angle[0] = - leg_state.thigh_angle
    # if prosthesis.desired_joint_angle[1] < -20./180*np.pi:
    #     prosthesis.desired_joint_angle[1] = -20./180*np.pi
        
def stance_2_control_step(environment_flag):
    if environment_flag == 5:
        environment_flag = 0
    prosthesis.desired_k[:] = [k_mat[environment_flag][0][1], k_mat[environment_flag][1][1]]
    prosthesis.desired_b[:] = [b_mat[environment_flag][0][1], b_mat[environment_flag][1][1]]
    prosthesis.desired_thetaeq[:] = [q_e_mat[environment_flag][0][1], q_e_mat[environment_flag][1][1]]

def stance_1_control_step(environment_flag):
    if environment_flag == 5:
        environment_flag = 0
    prosthesis.desired_k = [k_mat[environment_flag][0][0], k_mat[environment_flag][1][0]]
    prosthesis.desired_b = [b_mat[environment_flag][0][0], b_mat[environment_flag][1][0]]
    prosthesis.desired_thetaeq = [q_e_mat[environment_flag][0][0], q_e_mat[environment_flag][1][0]]

def stance_01_control_step(environment_flag,stance0_start_time,q_knee,q_ankle):
    stance01_time = 0.4
    if environment_flag == 5:
        environment_flag = 0
    t = time.time()-stance0_start_time
    print(t)
    if time.time() - stance0_start_time < stance01_time:
        
        prosthesis.desired_thetaeq[0] = q_knee + \
                                      (q_e_mat[environment_flag][0][0]-q_knee)*math.sin((time.time()-stance0_start_time)*np.pi/2/stance01_time)
        prosthesis.desired_thetaeq[1] = q_ankle + \
                                      (q_e_mat[environment_flag][1][0]-q_ankle)*math.sin((time.time()-stance0_start_time)*np.pi/2/stance01_time)
    else:
        prosthesis.stance_start = 1

def stance_12_control_step(environment_flag,stance2_start_time):
    stance12_time = 0.4
    if environment_flag == 5:
        environment_flag = 0
    t = time.time()-stance2_start_time
    print(t)
    if time.time() - stance2_start_time < stance12_time:
        prosthesis.desired_thetaeq[0] = q_e_mat[environment_flag][0][0] + \
                                      (q_e_mat[environment_flag][0][1]-q_e_mat[environment_flag][0][0])*math.sin((time.time()-stance2_start_time)*np.pi/2/stance12_time)
        prosthesis.desired_thetaeq[1] = q_e_mat[environment_flag][1][0] + \
                                      (q_e_mat[environment_flag][1][1]-q_e_mat[environment_flag][1][0])*math.sin((time.time()-stance2_start_time)*np.pi/2/stance12_time)
    else:
        prosthesis.stance12_start = 0
    # if time.time() - stance2_start_time < stance12_time:
    #     prosthesis.desired_thetaeq[0] = q_e_mat[environment_flag][0][0] + \
    #                                   (q_e_mat[environment_flag][0][1]-q_e_mat[environment_flag][0][0])*(1.0+0.5*(1.0/(1.0+math.exp(-((time.time() - stance2_start_time)*100-200)))))
    #     prosthesis.desired_thetaeq[1] = q_e_mat[environment_flag][1][0] + \
    #                                   (q_e_mat[environment_flag][1][1]-q_e_mat[environment_flag][1][0])*(1.0+0.5*(1.0/(1.0+math.exp(-((time.time() - stance2_start_time)*100-200)))))
    # else:
    #     prosthesis.stance12_start = 0
        
def cal_stance_control_command():
    # if verbose:
    #     print("cal_stance_control_command")
    q_qv_d = np.zeros(8)
    # prosthesis.desired_joint_angle = 180*np.ones(2)*1/6*np.sin((time_vec[-1]-time_vec[1])*2)

    q_qv_d[0], q_qv_d[2] = prosthesis.desired_thetaeq
    #q_qv_d[2] *= -1
    # if len(time_vec)==2:
    #     dt = 0.02
    # else:
    #     dt = time_vec[-1] - time_vec[-2]
    # q_qv_d[1::2] = (prosthesis.desired_joint_angle - prosthesis.current_joint_angle) / dt
    q_qv_d[1],q_qv_d[3] = 0,0
    q_qv_d[4], q_qv_d[6] = prosthesis.desired_k
    q_qv_d[5], q_qv_d[7] = prosthesis.desired_b
    #q_qv_d[2:4] = 0
    #q_qv_d[[0,1]] = 0
    print(q_qv_d)
    return q_qv_d


def cal_swing_control_command(time_vec):
    q_qv_d = np.zeros(8)
    # prosthesis.desired_joint_angle = 180*np.ones(2)*1/6*np.sin((time_vec[-1]-time_vec[1])*2)
    q_qv_d[[0,2]] = prosthesis.desired_joint_angle*180./np.pi
    q_qv_d[0] += leg_state.thigh_angle* 180. / np.pi
    
    if len(time_vec) == 2:
        dt = 0.02
    else:
        dt = time_vec[-1] - time_vec[-2]
    q_qv_d[[1,3]] = (prosthesis.desired_joint_angle - prosthesis.desired_joint_angle_prev) / dt
    # q_qv_d[2:4] = 0
    q_qv_d[4:] = np.array([80,3,80,3])
    print(q_qv_d, dt)
    # return q_qv_d
    # if verbose:
    #     print("cal_stance_control_command")
    # q_qv_d = np.zeros(8)
    # # prosthesis.desired_joint_angle = 180*np.ones(2)*1/6*np.sin((time_vec[-1]-time_vec[1])*2)
    # q_qv_d[0], q_qv_d[2] = prosthesis.desired_thetaeq
    # q_qv_d[2] *= -1
    # # if len(time_vec)==2:
    # #     dt = 0.02
    # # else:
    # #     dt = time_vec[-1] - time_vec[-2]
    # # q_qv_d[1::2] = (prosthesis.desired_joint_angle - prosthesis.current_joint_angle) / dt
    # q_qv_d[1],q_qv_d[3] = 0,0
    # q_qv_d[4], q_qv_d[6] = prosthesis.desired_k
    # q_qv_d[5], q_qv_d[7] = prosthesis.desired_b
    # q_qv_d[2:4] = 0
    #print(q_qv_d)
    return q_qv_d


def read_q_acc(imu_data_buf, q_thigh_init, yaw_init):
    q_thigh = -(imu_data_buf[0+9*cam_on_knee] - q_thigh_init)
    # TODO:projection
    ax, ay, az = cal_IMUacc(imu_data_buf, yaw_init)
    return q_thigh, ax, ay


def initialize_q_thigh(imu_data_buf, thigh_angle_buf):
    yaw_vec = np.zeros(10)
    for i in range(10):
        thigh_angle_buf = IO.fifo_data_vec(thigh_angle_buf, imu_data_buf[0+9*cam_on_knee])
        yaw_vec[i] = imu_data_buf[2+9*cam_on_knee]
        time.sleep(1e-2)
    q_thigh_init = np.mean(thigh_angle_buf)
    yaw_init = np.mean(yaw_vec)
    print('Initialized q_thigh={}, yaw = {}!'.format(q_thigh_init, yaw_init))
    return q_thigh_init, yaw_init


def cal_IMUacc(imu_data, yaw_init):  # 根据IMU数据计算世界坐标系下的加速度
    eular = np.zeros(3, )
    eular[0:2] = imu_data[(0+9*cam_on_knee):(2+9*cam_on_knee)]
    eular[2] = imu_data[2+9*cam_on_knee] - yaw_init
    eular = eular * np.pi / 180
    acc = imu_data[(3+9*cam_on_knee):(6+9*cam_on_knee)]
    gyr = imu_data[(6+9*cam_on_knee):(9+9*cam_on_knee)]
    acc_G = np.dot(Eular2R(eular), acc) - np.array([0, 0, 9.8])
    return acc_G[0], acc_G[1], acc_G[2]


def Eular2R(eular):  # 根据欧拉角计算旋转矩阵
    Rz = np.array([np.cos(eular[2]), -np.sin(eular[2]), 0, np.sin(eular[2]), np.cos(eular[2]), 0, 0, 0, 1]).reshape(3,
                                                                                                                    3)
    Ry = np.array([np.cos(eular[1]), 0, np.sin(eular[1]), 0, 1, 0, -np.sin(eular[1]), 0, np.cos(eular[1])]).reshape(3,
                                                                                                                    3)
    Rx = np.array([1, 0, 0, 0, np.cos(eular[0]), -np.sin(eular[0]), 0, np.sin(eular[0]), np.cos(eular[0])]).reshape(3,
                                                                                                                    3)

    R = np.dot(Rz, Ry)
    R = np.dot(R, Rx)
    return R


def initialize_prosthesis(ser):  # turn the prosthesis joint angle as [0,0]
    q_qv_d = np.array([0,0,0,0,30,0.5,30,0.5])
    Motor.send_signals_to_motor(ser, q_qv_d)
    step_num = 100
    q_qv,i_t = Motor.read_signals_from_motor(ser)
    for i in range(step_num):
        q_qv_d[:4:2] = q_qv[::2] * (99 - i) / 100
        Motor.send_signals_to_motor(ser, q_qv_d)
        q_qv,i_t = Motor.read_signals_from_motor(ser)
        time.sleep(8e-3)


def fifo_mat(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat


def environment_perception():
    if verbose:
        print("environment classification")
        print("environmental parameter estimation")
        print("environmental reconstruction")


if __name__ == '__main__':
    main()


def try_control():
    # IO.make_video('calibrate/video_frames')

    dst_RGBD = '../dataset/RGBD_out'
    dst_IMU = '../dataset/IMU_out'

    time_vec = np.load('{}/time_vec.npy'.format(dst_IMU))
    roll0 = np.load('{}/{:.3f}.npy'.format(dst_IMU, time_vec[0]))[[9, 9 * 2, 9 * 3]]
    flag_prev = 0
    plt.ion()
    # foot_acc = np.zeros(len(time_vec))
    # foot_gyr = np.zeros(len(time_vec))
    for i in range(3070, len(time_vec)):
        if time_vec[i] != 0:
            img_name = ('{}/{:.3f}'.format(dst_RGBD, time_vec[i]))
            imu_data_vec = np.load('{}/{:.3f}.npy'.format(dst_IMU, time_vec[i]))
            environment.classification_offline(imu_data_vec, img_name)
            eular = imu_data_vec[0:3]  # IMU的欧拉角
            foot_acc = np.linalg.norm(imu_data_vec[30:33], ord=None) - 9.8
            foot_gyr = np.linalg.norm(imu_data_vec[33:36], ord=None)

            flag_prev = leg_state.swing_stance_flag
            leg_state.update_swing_stance_flag(foot_acc, foot_gyr)
            theta = (-(eular[0] + 95) / 180) * np.pi  # 加个补偿，转成弧度
            roll = (imu_data_vec[[9, 9 * 2, 9 * 3]] - roll0) / 180 * np.pi
            [hip_angle, knee_angle, ankle_angle] = [roll[0], -roll[1] + roll[0], roll[1] - roll[2]]

            # if leg_state.swing_stance_flag - flag_prev == -1:#swing2stance
            environment.classification_offline(imu_data_vec, img_name)

            environment.update_2D_cloud_in_cam(img_name)
            cloud_in_cam = environment.cloud_in_cam
            cloud_in_1 = rotx(theta) @ cloud_in_cam

            cloud_in_2 = rotz(np.pi) @ roty(-np.pi / 2) @ cloud_in_1
            if environment.type_pred_from_nn == 1:
                xc, yc, w, h = environment.get_sa(cloud_in_2[0, :], cloud_in_2[1, :])

            hip_in_2 = np.array([-0.15, -0.07, 0]).reshape(-1, 1)
            knee_in_hip = np.array([0, -0.5, 0]).reshape(-1, 1)

            ankle_in_knee = np.array([0, -0.35, 0]).reshape(-1, 1)
            heel_in_ankle = np.array([-0.05, -0.05, 0]).reshape(-1, 1)
            toe_in_ankle = np.array([0.15, -0.05, 0]).reshape(-1, 1)
            knee_in_2 = rotz(hip_angle) @ knee_in_hip + hip_in_2;
            ankle_in_hip = rotz(-knee_angle) @ ankle_in_knee + knee_in_hip;
            ankle_in_2 = rotz(hip_angle) @ ankle_in_hip + hip_in_2;
            toe_in_knee = rotz(ankle_angle) @ toe_in_ankle + ankle_in_knee;
            toe_in_hip = rotz(-knee_angle) @ toe_in_knee + knee_in_hip;
            toe_in_2 = rotz(hip_angle) @ toe_in_hip + hip_in_2;
            heel_in_knee = rotz(ankle_angle) @ heel_in_ankle + ankle_in_knee;
            heel_in_hip = rotz(-knee_angle) @ heel_in_knee + knee_in_hip;
            heel_in_2 = rotz(hip_angle) @ heel_in_hip + hip_in_2;
            plt.clf()

            Plot.plot_skelenton(cloud_in_2, hip_in_2, knee_in_2, ankle_in_2, heel_in_2, toe_in_2, leg_state)
            # for environmentline in environment_structure:
            # plt.plot([environmentline[0,0],environmentline[1,0]],[environmentline[0,1],environmentline[1,1]],color='red')
            if environment.type_pred_from_nn == 1:
                plt.plot([xc, xc, xc + w, xc + w], [yc - h, yc, yc, yc + h], color='red')
            plt.pause(0.001)
            plt.ioff()





