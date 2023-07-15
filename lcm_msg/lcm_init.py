import numpy as np
import lcm
from lcm_msg.mvp_r import msg_r
from lcm_msg.mvp_t import msg_t


def lcm_initialize():
    msgs = msg_t()# msg to be send to node
    msgr = msg_r()# msg to be receive from node
    lc_t = lcm.LCM()
    lc_r = lcm.LCM()
    return msgs, msgr, lc_t, lc_r