import numpy as np
from Motor.motor import *
from lcm_msg.lcm_init import *
from lcm_msg.mvp_r import *
from lcm_msg.mvp_t import *



Knee = Motor()
Ankle = Motor()
lcm_msg_send, lcm_msg_receive, lcm_send_handler, lcm_receive_handler = lcm_initialize()
def bridge_handler(channel, data):
    msg = msg_r.decode(data)
    Knee.pos_actual = msg.knee_position_actual
    Ankle.pos_actual = msg.ankle_position_actual
    Knee.vel_actual = msg.knee_velocity_actual
    Ankle.vel_actual = msg.ankle_velocity_actual
    Knee.cur_actual = msg.knee_torque_actual
    Ankle.cur_actual = msg.ankle_torque_actual
    #print("Received message on channel \"%s\"" % channel)

def main():
    lcm_subscriber = lcm_receive_handler.subscribe("MIDDLE_TO_HIGH", bridge_handler)






    
    lcm_send_handler.publish("HIGH_TO_MIDDLE",lcm_msg_send.encode())
    lcm_receive_handler.handle()


