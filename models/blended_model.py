import numpy as np
from kinematics_model import KinematicBicycleModel
from dynamics_model import DynamicBicycleModel


class BlendedBicycleModel:
    def __init__(self, vblendmin, vblendmax, params):
        self.vblendmin = vblendmin
        self.vblendmax = vblendmax
        self.kinematic_model = KinematicBicycleModel(params['lf'], params['lr'], params['Ts'])
        self.dynamic_model = DynamicBicycleModel(params)
    
    def predict_next_state(self, xk, uk, vk):
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

