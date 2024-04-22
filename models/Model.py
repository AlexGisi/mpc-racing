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
    
    # TODO - ID these
    def Fx(self, throttle: float, steer: float) -> float:
        """
        steer: command in [-1, 1]
        throttle: command in [-1, 1]
        Longitudinal acceleration under a given throttle input. 
        
        Identification script in scripts/id_Fx.py, need to check
        that analysis.
        """
        return throttle*4000

    # TODO - ID these
    def steer_cmd_to_angle(self, steer_cmd: float) -> float:
        """
        Maps a steer command in [-1, 1] to the angle of the wheels
        over the next timestep.
        """
        return np.deg2rad(steer_cmd*45.0)

    # TODO - ID these
    def throttle_cmd_to_acceleration(self, throttle_cmd: float) -> float:
        """
        Maps a throttle command in [-1, 1] to the acceleration over
        the next timestep.
        """
        return throttle_cmd

    def step(self, throttle_cmd: float, steer_cmd: float) -> Type[State]:
        """
        :throttle: [-1, 1]
        :steer_cmd: [-1, 1]
        """
        raise NotImplementedError
