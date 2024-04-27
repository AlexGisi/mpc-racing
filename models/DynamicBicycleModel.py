"""
Not implemented correctly or something, speed blows up. todo: fix and test.
"""

from typing import Type
import numpy as np
from models.Model import Model
from models.VehicleParameters import VehicleParameters
from models.State import State


class DynamicBicycleModel(Model):
    def __init__(self, initial_state: Type[State]):
        super().__init__(initial_state)

    def step(self, throttle_cmd: float, steer_cmd: float) -> Type[State]:
        # Unpack parameters for easier access
        params = self.params
        m = params.m
        Iz = params.Iz
        lf = params.lf
        lr = params.lr
        g = params.g
        Cf = params.Cf
        Cr = params.Cr
        Ts = params.Ts

        # Current state unpacking
        x, y, yaw, v_x, v_y, yaw_dot = self.state.x, self.state.y, self.state.yaw, self.state.v_x, self.state.v_y, self.state.yaw_dot
        vel = np.hypot(v_x, v_y)

        # Convert commands to physical values.
        Fx = self.Fx(throttle_cmd)
        delta = self.steer_cmd_to_angle(steer_cmd, vel)

        # Convert body frame velocities to global frame
        v = np.sqrt(v_x**2 + v_y**2)
        beta = np.arctan2(lr * yaw_dot, v)  # Slip angle at center of mass

        # See Vehicle Dynamics And Control (2005) by Rajamani, page 31.
        # Calculate tire velocity angle at front and rear.
        theta_Vf = np.arctan2((v_y + lf * yaw_dot), v_x)
        theta_Vr = np.arctan2((v_y - lr * yaw_dot), v_x)

        # Calculate lateral forces at front and rear using linear tire model.
        Fyf = Cf * (delta - theta_Vf)
        Fyr = Cr * (-theta_Vr)

        # Dynamics equations
        # See "Online Learning of MPC for Autonomous Racing" by Costa et al
        v_x_dot = ( (Fx - Fyf*np.sin(delta)) / m ) + v_y * yaw_dot
        v_y_dot = ((Fyf*np.cos(delta) + Fyr) / m) - v_x * yaw_dot
        yaw_dot_dot = (Fyf*np.cos(delta)*lf - Fyr*lr) / Iz

        # Integrate to find new state
        x_new = x + (v_x * np.cos(yaw) - v_y * np.sin(yaw)) * Ts
        y_new = y + (v_x * np.sin(yaw) + v_y * np.cos(yaw)) * Ts
        yaw_new = yaw + yaw_dot * Ts
        v_x_new = v_x + v_x_dot * Ts
        v_y_new = v_y + v_y_dot * Ts
        yaw_dot_new = yaw_dot + yaw_dot_dot * Ts

        # Update the state
        self.state = State(x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new)

        return self.state
