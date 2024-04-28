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
        steer: command in [-1, 1]
        """
        info = {}
        # Model regenerative braking.
        if throttle == 0.0:
            regen_brake_force = self.params.regen_brake_accel * 9.81 * self.params.m
        else:
            regen_brake_force = 0

        # The motor efficiency depends on the rpm, as described by the throttle curve
        # in model3_data.py. 
        eta = self.params.eta_motor
        rpm = self.params.max_rpm * np.abs(throttle) * self.params.R
        if rpm < 9000:
            eta = 1.0
        elif rpm < 9500:
            eta = 0.88
        elif rpm < 10_400:
            eta = 0.81
        elif rpm < 12_500:
            eta = 0.71
        elif rpm < 15_000:
            eta = 0.675
        else:
            eta = 0.6
        
        wheel_force = throttle*eta*self.params.T_max*self.params.R / self.params.r_wheel
        drag_force = 0.5*self.params.rho*self.params.C_d*self.params.A_f*(self.state.v_x**2)
        rolling_resistance = self.params.C_roll*self.params.m*self.params.g

        info['wheel'] = wheel_force
        info['drag'] = drag_force
        info['rolling_resistance'] = rolling_resistance
        info['regen_brake'] = regen_brake_force
        info['eta'] = eta
        info['rpm'] = rpm

        return wheel_force - drag_force - rolling_resistance - regen_brake_force, info

    def steer_cmd_to_angle(self, steer_cmd: float) -> float:
        """
        Maps a steer command in [-1, 1] to the angle of the wheels
        over the next timestep.
        """
        vel = np.hypot(self.state.v_x, self.state.v_y) * 3.6  # Convert to km/h
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
        :throttle: [-1, 1], in practice the addition of the carla brake and throttle commands
        :steer_cmd: [-1, 1]
        """
        raise NotImplementedError
