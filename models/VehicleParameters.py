from dataclasses import dataclass

@dataclass
class VehicleParameters:
    # FST10d model parameters for the kinematic, dynamic, and blended models.
    
    # General car parameters
    m: float = 1845.0  # kg
    max_steer = 70.0  # deg
    min_steer = -70.0  # deg
    max_vel = 33.0  # m/s

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
    
    Cf: float = 65_000  # Cornering stiffness coefficient, front tires (trained)
    Cr: float = 65_000  # Cornering stiffness coefficient, rear tires (trained)

    # Pacejka tire model
    pac_Bf: float = 10  # Pacejka tire model coefficient Bf, dimensionless
    pac_Br: float = 10  # Pacejka tire model coefficient Br, dimensionless
    pac_Cf: float = 138  # Pacejka tire model coefficient Cf, dimensionless 
    pac_Cr: float = 138
    pac_Df: float = 1500   # Pacejka tire model coefficient Df in Newtons  
    pac_Dr: float = 1500   # Pacejka tire model coefficient Dr in Newtons 

    g: float = 9.81  # Acceleration due to gravity in meters per second squared
    Vblendmin: float = 2  # Minimum blending speed in meters per second
    Vblendmax: float = 15  # Maximum blending speed in meters per second
    Ts: int = 0.05  # Sampling time in seconds

    car_width = 1.85  # car width is 1.85, this adds safety expansion (m)
