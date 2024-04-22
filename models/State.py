from dataclasses import dataclass

@dataclass
class State:
    x: float        # Global X position
    y: float        # Global Y position
    yaw: float      # Angle with the X axis
    v_x: float      # Velocity in the x direction in the local frame
    v_y: float      # Velocity in the y direciton in the local frame
    yaw_dot: float
