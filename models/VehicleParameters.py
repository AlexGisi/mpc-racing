from dataclasses import dataclass

@dataclass
class VehicleParameters:
    # FST10d model parameters for the kinematic, dynamic, and blended models.
    
    m: float = 1847  # Vehicle mass in kilograms - I VERIFIED IN CARLA

    # Need these (probably can get from carla/api)
    Iz: float = 80  # Moment of inertia around the z-axis in kilogram meter squared
    lf: float = 0.832  # Distance from the center of mass to the front axle in meters
    lr: float = 0.708  # Distance from the center of mass to the rear axle in meters

    # Don't need, could be nice
    Tmax: float = 21  # Maximum torque produced by the motors in Newton meters
    
    # Need to ID
    Cf: float = 0.092  # Cornering stiffness coefficient of the front tires, dimensionless
    Cr: float = 0.092  # Cornering stiffness coefficient of the rear tires, dimensionless

    # Don't need, would be nice but using linear tire model for now
    Bf: int = 10  # Pacejka tire model coefficient Bf, dimensionless
    Br: int = 10  # Pacejka tire model coefficient Br, dimensionless
    Cf: int = 138  # Pacejka tire model coefficient Cf, dimensionless 
    Df: int = 1500  # Pacejka tire model coefficient Df in Newtons  
    Dr: int = 1500  # Pacejka tire model coefficient Dr in Newtons 

    g: float = 9.81  # Acceleration due to gravity in meters per second squared
    Vblendmin: float = 2  # Minimum blending speed in meters per second
    Vblendmax: float = 5  # Maximum blending speed in meters per second
    Ts: int = 0.1  # Sampling time in seconds
