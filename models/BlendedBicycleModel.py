"""
Haven't looked at this too closely yet.
"""
from typing import Type
import numpy as np
from models.Model import Model
from models.KinematicBicycleModel import KinematicBicycleModel
from models.DynamicBicycleModel import DynamicBicycleModel
from models.State import State


class BlendedBicycleModel(Model):
    def __init__(self, initial_state: Type[State]):
        super().__init__(initial_state)
        self.kinematic_model = None
        self.dynamic_model = None
        
    def step(self, throttle_cmd: float, steer_cmd: float, dt=None):
        self.kinematic_model = KinematicBicycleModel(initial_state=self.state)
        self.dynamic_model = DynamicBicycleModel(initial_state=self.state)

        Ts = self.params.Ts if dt is None else dt
        vel = np.hypot(self.state.v_x, self.state.v_y)

        # Calculate the blend factor based on the vehicle's velocity vk
        lam = np.clip((vel - self.params.Vblendmin) / (self.params.Vblendmax - self.params.Vblendmin), 0, 1)
        
        # Forward prediction.
        kin_pred, _ = self.kinematic_model.step(throttle_cmd, steer_cmd, dt=Ts)
        dyn_pred, info = self.dynamic_model.step(throttle_cmd, steer_cmd, dt=Ts)
        
        # Blend
        kin = np.array([kin_pred.x,
                        kin_pred.y,
                        kin_pred.yaw,
                        kin_pred.v_x,
                        kin_pred.v_y,
                        kin_pred.yaw_dot])
        dyn = np.array([dyn_pred.x,
                        dyn_pred.y,
                        dyn_pred.yaw,
                        dyn_pred.v_x,
                        dyn_pred.v_y,
                        dyn_pred.yaw_dot])
        
        blend = lam * dyn + (1 - lam) * kin

        self.state = State(
            x=blend[0],
            y=blend[1],
            yaw=blend[2],
            v_x=blend[3],
            v_y=blend[4],
            yaw_dot=blend[5]
        )

        info['lambda'] = lam
        return self.state, info
    
    def __repr__(self) -> str:
        return "BlendedBicycleModel"
