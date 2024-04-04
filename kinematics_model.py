import numpy as np
from vehicle_config import *

class KinematicBicycleModel:
    def __init__(self, lf, lr, Ts):
        self.lf = lf  # distance from the center of mass to the front axle (m)
        self.lr = lr  # distance from the center of mass to the rear axle (m)
        self.Ts = Ts  # sampling time (ms)
        
        self.x_dot = None
        self.y_dot = None
        self.yaw_dot = None
        self.v_x_dot = None
        
        self.model = np.array([self.x_dot, self.y_dot, self.yaw_dot, self.v_x_dot]).T
        
        self.Fx = None

    def update_model_params(self, xk, uk):
        x = xk[0]
        y = xk[1]
        yaw = xk[2]
        vx = xk[3]
        vy = xk[4]
        r = xk[5]
        d = uk[0]
        sigma = uk[1]
        
        self.x_dot = vx*np.cos(yaw) - vy*np.sin(yaw)
        self.y_dot = vx*np.sin(yaw) - vy*np.cos(yaw)
        self.yaw_dot = (vx/(self.lr + self.lf))*np.tan(sigma)
        self.Fx = self.calculate_Fx(self, uk, vx)
        
    def predict_next_state(self, xk, uk): 
        pass        
        
    def calculate_Fx(self, uk, velocity):
        # Unpack the control input vector u, where d is the throttle and sigma is the steering angle
        d, sigma = uk

        F_propulsion = 2 * eta_motor * T_max * d / r_wheel
        F_aero = -0.5 * rho * Cd * Af * velocity ** 2
        F_rolling = -Crr * m * g
        Fx = F_propulsion + F_aero + F_rolling

        return Fx
        
        
        
