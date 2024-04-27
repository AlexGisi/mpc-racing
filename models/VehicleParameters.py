from dataclasses import dataclass

@dataclass
class VehicleParameters:
    # FST10d model parameters for the kinematic, dynamic, and blended models.
    
    # General car parameters
    m: float = 1845.0  # Vehicle mass in kilograms
    max_steer = 70  # Deg
    min_steer = -70  # Deg

    # For Fx
    eta_motor = 0.9 # (technically it changes significantly with torque curve)
    T_max = 743.0  # Max value on torque curve (N*m)
    r_wheel = 0.37 # (m)
    R = 9.0  # transmission gear ratio
    rho = 1.225  # air density (at sea level)
    C_d = 0.23  # drag coefficient (via teslaoracle.com)
    A_f = 2.2  # Frontal area (estimate)
    C_roll = 0.012  # Roll coefficient (confident estimate)

    # For dynamic model
    Iz: float = 80  # Iz around z-axis in (kg m^2)
    lf: float = 1.4  # Center of mass to the front axle (m) (informed guess)
    lr: float = 1.4  # Center of mass to the rear axle (m) (informed guess)
    
    # Need to ID
    Cf: float = 2  #3 Cornering stiffness coefficient, front tires
    Cr: float = 2  # Cornering stiffness coefficient,  rear tires

    # Don't need, would be nice but using linear tire model for now
    Bf: int = 15  # Pacejka tire model coefficient Bf, dimensionless
    Br: int = 15  # Pacejka tire model coefficient Br, dimensionless
    Cf: int = 1.4  # Pacejka tire model coefficient Cf, dimensionless 
    Df: int = 4524.86   # Pacejka tire model coefficient Df in Newtons  
    Dr: int = 4524.86   # Pacejka tire model coefficient Dr in Newtons 

    g: float = 9.81  # Acceleration due to gravity in meters per second squared
    Vblendmin: float = 10  # Minimum blending speed in meters per second
    Vblendmax: float = 40  # Maximum blending speed in meters per second
    Ts: int = 0.1  # Sampling time in seconds
