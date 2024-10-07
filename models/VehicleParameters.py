from dataclasses import dataclass

@dataclass
class VehicleParameters:
    # Model parameters for the kinematic, dynamic, and blended models.
    
    # General car parameters
    m: float = 1845.0  # kg
    max_steer = 70.0  # deg
    min_steer = -70.0  # deg

    # For Fx
    eta_motor = 0.9 # (default, changed in Fx() function)
    T_max = 743.0  # Max value on torque curve (N*m)
    r_wheel = 0.37 # wheel radius (m)
    C_wheel = 2*3.14*r_wheel  # wheel circumference
    R = 9.0  # transmission gear ratio
    rho = 1.225  # air density (at sea level)
    C_d = 0.23  # drag coefficient (via teslaoracle.com)
    A_f = 2.2  # Frontal area (via internet)
    C_roll = 0.012  # Roll coefficient (confident estimate)
    max_rpm = 15000
    regen_brake_accel = 0.2  # applied when throttle==brake==0 (g)

    # For dynamic model
    Iz: float = 3960.0  # Iz around z-axis in (kg m^2) (from modeling it as a prism)
    lf: float = 0.8  # Center of mass to the front axle (m) (trial and error)
    lr: float = 2  # Center of mass to the rear axle (m) (trial and error)

    # lf: float = 0.9088  # Center of mass to the front axle (m) (trained)
    # lr: float = 1.7092  # Center of mass to the rear axle (m) (trained)
    
    Cf: float = 65_000  # Cornering stiffness coefficient, front tires (trained)
    Cr: float = 65_000  # Cornering stiffness coefficient, rear tires (trained)

    g: float = 9.81  # Acceleration due to gravity in meters per second squared
    Vblendmin: float = 2  # Minimum blending speed in meters per second
    Vblendmax: float = 15  # Maximum blending speed in meters per second
    Ts: int = 0.05  # Sampling time in seconds

    car_width = 1.85  # car width is 1.85, this adds safety expansion (m)
