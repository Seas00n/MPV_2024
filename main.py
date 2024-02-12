import numpy as np
from Motor.motor import *
from lcm_msg.lcm_init import *
from lcm_msg.mvp_r import *
from lcm_msg.mvp_t import *

Knee = Motor()
Ankle = Motor()
high2mid_msg, mid2high_msg, lc_high2mid, lc_mid2high = lcm_initialize()


def bridge_handler(channel, data):
    msg = mid2high_msg.decode(data)
    Knee.pos_actual = msg.knee_position_actual
    Ankle.pos_actual = msg.ankle_position_actual
    Knee.vel_actual = msg.knee_velocity_actual
    Ankle.vel_actual = msg.ankle_velocity_actual
    Knee.cur_actual = msg.knee_torque_actual
    Ankle.cur_actual = msg.ankle_torque_actual
    # print("Received message on channel \"%s\"" % channel)


def main():
    lcm_subscriber = lc_mid2high.subscribe("MIDDLE_TO_HIGH", bridge_handler)
    lc_high2mid.publish("HIGH_TO_MIDDLE", high2mid_msg.encode())
    lc_mid2high.handle()
