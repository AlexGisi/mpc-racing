from typing import Type
import numpy as np
from models.State import State
from models.VehicleParameters import VehicleParameters

class Model:
    """
    Vehicle model abstract class.
    """
    def __init__(self, initial_state: Type[State]):
        self.params = VehicleParameters()
        self.state = initial_state
    
    def Fx(self, throttle: float) -> float:
        """
        TODO: big improvement would be adjusting motor efficiency based off
        torque curve

        the 2 in front of wheel force is sus

        steer: command in [-1, 1]
        """
        wheel_force = 2*throttle*self.params.eta_motor*self.params.T_max*self.params.R / self.params.r_wheel
        drag_force = 0.5*self.params.rho*self.params.C_d*self.params.A_f*(self.state.v_x**2)
        rolling_resistance = self.params.C_roll*self.params.m*self.params.g

        return wheel_force - drag_force - rolling_resistance

    def steer_cmd_to_angle(self, steer_cmd: float) -> float:
        """
        Maps a steer command in [-1, 1] to the angle of the wheels
        over the next timestep.
        """
        vel = np.hypot(self.state.v_x, self.state.v_y)
        if vel < 20.0:
            gain = 1.0
        elif vel < 60.0:
            gain = 0.9
        elif vel < 120.0:
            gain = 0.8
        else:
            gain = 0.7

        return np.deg2rad(steer_cmd*self.params.max_steer*gain)

    def step(self, throttle_cmd: float, steer_cmd: float) -> Type[State]:
        """
        :throttle: [-1, 1]
        :steer_cmd: [-1, 1]
        """
        raise NotImplementedError
