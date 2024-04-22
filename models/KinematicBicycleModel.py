from typing import Type
import numpy as np
from models.Model import Model
from models.VehicleParameters import VehicleParameters
from models.State import State

class KinematicBicycleModel(Model):
    def __init__(self, initial_state: Type[State]):
        super().__init__(initial_state)

    def step(self, throttle_cmd: float, steer_cmd: float) -> Type[State]:
        """
        Calculate the next state using the kinematic bicycle model.
        """
        # Unpack parameters for easier access
        params = self.params
        lf = params.lf
        lr = params.lr
        Ts = params.Ts
        m = params.m

        # Current state unpacking
        x, y, yaw, v_x, v_y, yaw_dot = self.state.x, self.state.y, self.state.yaw, self.state.v_x, self.state.v_y, self.state.yaw_dot
        
        # Convert commands to physical values.
        delta = self.steer_cmd_to_angle(steer_cmd)
        Fx = self.Fx(throttle_cmd, steer_cmd)

        # Kinematic model equations
        v = np.sqrt(v_x**2 + v_y**2)
        
        # Update the state equations
        # x_new = x + v * np.cos(yaw + beta) * Ts
        # y_new = y + v * np.sin(yaw + beta) * Ts
        # yaw_new = yaw + (v / lr) * np.sin(beta) * Ts
        # v_x_new = v_x + a * np.cos(yaw) * Ts
        # v_y_new = v_y + a * np.sin(yaw) * Ts
        # yaw_dot_new = (v / lr) * np.sin(beta)

        x_new = x + (v_x * np.cos(yaw) - v_y * np.sin(yaw)) * Ts
        y_new = y + (v_x * np.sin(yaw) + v_y * np.cos(yaw)) * Ts
        yaw_new = yaw + ( (v_x / (lr + lf)) * np.tan(delta) ) * Ts
        v_x_new = v_x + (Fx / m) * Ts

        yaw_dot_new = (v_x_new * (lr + lf)) * np.tan(delta)
        v_y_new = yaw_dot_new * lr

        # Update the state
        self.state = State(x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new)

        return self.state
