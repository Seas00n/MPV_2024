import numpy as np
import lcm
from lcm_msg.mvp_r import msg_r
from lcm_msg.mvp_t import msg_t
from lcm_msg.pcd_lcm.pcd_xyz import pcd_xyz


def lcm_initialize():
    high2mid_msg = msg_t()  # msg to be send to node
    mid2high_msg = msg_r()  # msg to be receive from node
    lc_high2mid = lcm.LCM()
    lc_mid2high = lcm.LCM()
    return high2mid_msg, mid2high_msg, lc_high2mid, lc_mid2high


def pcd_lcm_initialize():
    lc = lcm.LCM()
    pcd_msg = pcd_xyz()
    return pcd_msg, lc
