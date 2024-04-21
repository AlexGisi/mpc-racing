from typing import Type
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
    def steer_cmd_to_angle(steer_cmd: float) -> float:
        """
        Maps a steer command in [-1, 1] to the angle of the wheels
        over the next timestep.
        """
        pass

    # TODO - ID these
    def throttle_cmd_to_acceleration(throttle_cmd: float) -> float:
        """
        Maps a throttle command in [-1, 1] to the acceleration over
        the next timestep.
        """
        pass

    def step(self, throttle_cmd: float, steer_cmd: float) -> Type[State]:
        """
        :throttle: [-1, 1]
        :steer_cmd: [-1, 1]
        """
        raise NotImplementedError
