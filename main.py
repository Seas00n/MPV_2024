import numpy as np

# mode0: realtime
# mode1: simulation
# mode2:
running_mode = 0

def main():
    if running_mode == 0:
        imu_data_buf = []
        six_force_buf = []
        env_type_buf = []
        env_fea_buf = []
    if running_mode == 1:
        imu_data_buf = []

