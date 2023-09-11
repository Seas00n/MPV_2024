import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

from mujoco_base import MuJoCoBase

save_path = "./data/"
ctrl_list = []
q_list = []
qd_list = []
qdd_list = []
time_list = []


class Wakler2d(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.simend = 100.0

    def reset(self):
        self.cam.azimuth = 89.608063
        self.cam.elevation = -11.588379
        self.cam.distance = 5.0
        self.cam.lookat = np.array([0.0, 0.0, 1.5])

        self.model.opt.gravity[2] = -9.81

        mj.set_mjcb_control(self.controller)


    def controller(self, model: mj.MjModel, data: mj.MjData):
        data.ctrl[0] = 0
        data.ctrl[1] = 0
        data.ctrl[2] = 0
        data.ctrl[3] = 0
        self.set_hip_pos_vel(pos=np.sin(data.time), vel=np.cos(data.time), data=data)
        self.set_knee_pos_vel(pos=np.sin(data.time), vel=np.cos(data.time), data=data)
        self.set_ankle_pos_vel(pos=0.5 * np.cos(data.time) + np.pi / 2, vel=-0.5 * np.sin(data.time) + np.pi / 2,
                               data=data)
        noise1 = mj.mju_standardNormal(0.0)
        noise2 = mj.mju_standardNormal(0.0)
        noise3 = mj.mju_standardNormal(0.0)
        data.qfrc_applied[2] = noise1
        data.qfrc_applied[3] = noise2
        data.qfrc_applied[4] = noise3

    def set_hip_pos_vel(self, pos, vel, data, kp=1000, kd=100):
        data.ctrl[4] = kp*(pos-data.qpos[2])+kd*(vel-data.qvel[2])

    def set_knee_pos_vel(self, pos, vel, data, kp=1000, kd=100):
        data.ctrl[5] = kp * (pos - data.qpos[3]) + kd * (vel - data.qvel[3])

    def set_ankle_pos_vel(self, pos, vel, data, kp=1000, kd=100):
        data.ctrl[6] = kp * (pos - data.qpos[4]) + kd * (vel - data.qvel[4])

    def simulate(self):
        global ctrl_list, q_list, qd_list, qdd_list, time_list
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            while self.data.time - simstart < 1.0 / 60.0:
                # Step simulation environment
                mj.mj_step(self.model, self.data)

                # Apply control
                self.controller(self.model, self.data)
                ctrl_list.append(np.copy(self.data.sensordata))
                qdd_list.append(np.copy(self.data.qacc))
                qd_list.append(np.copy(self.data.qvel))
                q_list.append(np.copy(self.data.qpos))
                time_list.append(np.copy(self.data.time))
                print(self.data.sensordata)
            if self.data.time >= self.simend:
                np.save(save_path + "ctrl.npy", np.array(ctrl_list))
                np.save(save_path + "qdd.npy", np.array(qdd_list))
                np.save(save_path + "qd.npy", np.array(qd_list))
                np.save(save_path + "q.npy", np.array(q_list))
                np.save(save_path + "t.npy", np.array(time_list))
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 0

            # Update scene and render
            # self.cam.lookat[0] = self.data.qpos[0]
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


def main():
    xml_path = "walker2d.xml"
    sim = Wakler2d(xml_path)
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
