from dataclasses import dataclass

@dataclass
class State:
    x: float
    y: float
    yaw: float
    v_x: float
    v_y: float
    yaw_dot: float
