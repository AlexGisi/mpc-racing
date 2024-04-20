from dataclasses import dataclass

@dataclass
class VehicleParameters:
    # FST10d model parameters for the kinematic, dynamic, and blended models.
    
    m: float = 250  # Vehicle mass in kilograms
    Iz: float = 80  # Moment of inertia around the z-axis in kilogram meter squared
    lf: float = 0.832  # Distance from the center of mass to the front axle in meters
    lr: float = 0.708  # Distance from the center of mass to the rear axle in meters
    Cd: float = 1.2  # Aerodynamic drag coefficient, dimensionless
    AF: float = 1.18  # Frontal area of the vehicle in square meters
    GR: float = 15.74  # Gear ratio, dimensionless
    rwheel: float = 0.23  # Radius of the wheels in meters
    Tmax: float = 21  # Maximum torque produced by the motors in Newton meters
    etamotor: float = 0.9  # Efficiency of the motor, dimensionless
    Cr: float = 0.092  # Cornering stiffness coefficient of the rear tires, dimensionless
    Bf: int = 10  # Pacejka tire model coefficient Bf, dimensionless
    Br: int = 10  # Pacejka tire model coefficient Br, dimensionless
    Cf: int = 138  # Pacejka tire model coefficient Cf, dimensionless
    Df: int = 1500  # Pacejka tire model coefficient Df in Newtons
    Dr: int = 1500  # Pacejka tire model coefficient Dr in Newtons
    rho: float = 1.18  # Air density in kilograms per cubic meter
    g: float = 9.81  # Acceleration due to gravity in meters per second squared
    Vblendmin: float = 2  # Minimum blending speed in meters per second
    Vblendmax: float = 5  # Maximum blending speed in meters per second
    Ts: int = 0.01  # Sampling time in seconds
