class Motor():
    def __init__(self):
        self.pos_desired = 0
        self.pos_actual = 0
        self.vel_desired = 0
        self.vel_actual = 0
        self.cur_desired = 0
        self.cur_actual = 0
        self.Kp = 0
        self.Kd = 0
        self.Angle_eq = 0
        self.Mode = 0
        self.State = 1
