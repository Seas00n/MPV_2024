import matplotlib.pyplot as plt
import matplotlib
from matplotlib import *
import numpy as np
import time


class FastPlotCanvas(object):
    ax: matplotlib.axes._axes.Axes
    fig: matplotlib.figure.Figure

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line_pcd, = self.ax.plot(np.zeros(1, ))
        self.pcd = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='r', alpha=0.1)
        self.pcd2 = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='b', alpha=0.1)
        self.fea_A = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='m', linewidths=2)
        self.fea_B = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='g', linewidths=2)
        self.fea_C = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='y', linewidths=2)
        self.fea_D = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='tab:purple', linewidths=2)
        self.fea_E = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='tab:green', linewidths=2)
        self.fea_F = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='tab:orange', linewidths=2)
        self.text_info = self.ax.text(0, 0, "")
        plt.show(block=False)


    def update_canvas(self):
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        self.ax.draw_artist(self.ax.patch)

    def set_pcd(self, pcd_data, idx):
        if idx == 'new':
            self.pcd.set_offsets(pcd_data)
            self.ax.draw_artist(self.pcd)
        else:
            self.pcd2.set_offsets(pcd_data)
            self.ax.draw_artist(self.pcd2)

    def set_fea_A(self, fea_data):
        self.fea_A.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_A)

    def set_fea_B(self, fea_data):
        self.fea_B.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_B)

    def set_fea_C(self, fea_data):
        self.fea_C.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_C)

    def set_fea_D(self, fea_data):
        self.fea_D.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_D)

    def set_fea_E(self, fea_data):
        self.fea_E.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_E)

    def set_fea_F(self, fea_data):
        self.fea_F.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_F)

    def set_info(self, px, py, id, corner_situation, env_rotate):
        self.text_info.set_text("id:{},corner_situation:{},env_rotate:{}".format(id,
                                                                                 round(corner_situation, 2),
                                                                                 round(env_rotate, 2)))
        self.text_info.set_x(px)
        self.text_info.set_y(py)
        self.ax.draw_artist(self.text_info)

if __name__ == '__main__':
    fpc = FastPlotCanvas()

    tstart = time.time()
    num_frames = 0
    while time.time() - tstart < 5:
        pcd_data = np.random.random((100, 2))
        pcd_data[:, 0] = np.linspace(-1, 1, np.shape(pcd_data)[0])
        fpc.set_pcd(pcd_data, 'new')
        fpc.set_pcd(pcd_data, 'pre')
        fpc.set_info(0, -0.2, 1, 1, 1)
        fpc.update_canvas()
        num_frames += 1
    print(num_frames / 5)
