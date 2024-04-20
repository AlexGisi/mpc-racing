from typing import Type
import numpy as np
from KinematicBicycleModel import KinematicBicycleModel
from DynamicBicycleModel import DynamicBicycleModel
from State import State
from VehicleParameters import VehicleParameters


class BlendedBicycleModel:
    def __init__(self, initial_state: Type[State]):
        super().__init__(initial_state)

    def __init__(self, vblendmin, vblendmax, params):
        self.vblendmin = vblendmin
        self.vblendmax = vblendmax
        self.kinematic_model = KinematicBicycleModel(params['lf'], params['lr'], params['Ts'])
        self.dynamic_model = DynamicBicycleModel(params)
    
    def step(self, xk, uk, vk):
        # Calculate the blend factor based on the vehicle's velocity vk
        λk = np.clip((vk - self.vblendmin) / (self.vblendmax - self.vblendmin), 0, 1)
        
        # Predict the next state using both kinematic and dynamic models
        xk_next_kin = self.kinematic_model.predict_next_state(xk, uk)
        xk_next_dyn = self.dynamic_model.predict_next_state(xk, uk)
        
        # Blend the next state predictions
        xk_next = λk * xk_next_dyn + (1 - λk) * xk_next_kin
        return xk_next


# Example usage:
# Initialize the blended bicycle model
blended_bicycle_model = BlendedBicycleModel(vblendmin=2, vblendmax=5, params=vehicle_params)

