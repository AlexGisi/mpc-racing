from dataclasses import dataclass

@dataclass
# x = [X Y PHI Vx Vy r].T
def State():
    x_pos: float
    y_pos: float
    orientation: float 
    velocity_x: float 
    velocity_y: float 
    orientation_dot: float 
    
@dataclass
# u = [d sigma].T
def Input():
    normalized_throttle: float 
    steering_angle: float 