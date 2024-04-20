import carla
import time
import numpy as np
from numpy.linalg import norm
from splines.ParameterizedCenterline import ParameterizedCenterline
from Logger import Logger

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.centerline = ParameterizedCenterline()
        self.centerline.from_file("/home/alex/Projects/graic/autobots-race/waypoints/shanghai_intl_circuit")
        self.progress = None  # Car is spawned in the middle of the track.
        
        self.X = None
        self.Y = None
        self.yaw = None
        self.vx = None
        self.vy = None
        
        self.steps = 0
        self.error = None
        self.last_error = 0
        self.cum_error = 0
        self.cmd_steer = 0
        self.logger = Logger("data-mydrive.csv")

    def progress_bound(self):
        """
        Return a bound on the possible progress of the vehicle in order. Such a bound
        is required in order to use the ParameterizedCenterline projection method, but
        it need not be tight.
        """
        if self.progress is None:
            return None
        
        lower = (self.progress-2)  % self.centerline.length
        upper = (self.progress+2) % self.centerline.length

        return (lower, upper) if lower < upper else (upper, lower)

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation. Times out in 10s.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s, in global coordinates
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        self.X = transform.location.x
        self.Y = transform.location.y
        self.yaw = np.deg2rad(transform.rotation.yaw)

        # Velocities in vehicle coordinates (y is lateral).
        # Positive lateral velocity is to the left.
        self.vx = vel.x * np.cos(-self.yaw) - vel.y * np.sin(-self.yaw)
        self.vy = vel.x * np.sin(-self.yaw) + vel.y * np.cos(-self.yaw)

        self.progress, dist = self.centerline.projection(self.X, self.Y, bounds=self.progress_bound())
        self.error = dist * self.centerline.error_sign(self.X, self.Y, self.progress)
        
        print(self.steps)
        print("progress: ", self.progress)
        print("error: ", self.error)

        print("X: ", self.X)
        print("Y: ", self.Y)
        print("Yaw: ", self.yaw)

        print("vx: ", self.vx)
        print('vy: ', self.vy)

        control = carla.VehicleControl()

        ### PD control ###
        D = self.error - self.last_error
        P = self.error
        kD = 5
        kP = 0.2

        control.throttle = 0.5
        control.steer = kD * D + kP * P
        ### end PID control ###

        control.steer = control.steer
        
        print("D: ", D)
        print("P: ", P)
        print("Dt: ", D*kD)
        print("Pt: ", P*kP)
        print("control: ", control.throttle, control.steer)

        self.steps += 1
        self.last_error = self.error
        self.cum_error += self.error
        self.cmd_steer = control.steer

        self.logger.log(self)

        return control

