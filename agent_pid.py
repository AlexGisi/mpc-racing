import carla
import time
import numpy as np
from numpy.linalg import norm
from splines.ParameterizedCenterline import ParameterizedCenterline
from Logger import Logger

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.centerline = ParameterizedCenterline("shanghai_intl_circuit")
        self.progress = None  # Car is spawned in the middle of the track.
        
        # Vehicle state
        self.X = None
        self.Y = None
        self.yaw = None
        self.vx = None
        self.vy = None
        self.yawdot = None
        self.vel = None

        # For data collection
        self.left_lane_points = None
        self.right_lane_points = None

        self.next_left_lane_point_x = None
        self.next_left_lane_point_y = None
        self.next_right_lane_point_x = None
        self.next_right_lane_point_y = None

        self.steps = 0
        self.error = None
        self.last_error = 0
        self.sim_time = 0
        self.last_ts = 0
        self.mpc_time = -1


        self.last_throttle_error = 0
        
        self.steps = 0
        self.error = None
        self.last_error = 0

        # Tunable parameters
        self.k = 1.3  # Target vel aggressiveness, k=1.2 is reasonable aggressive.
        self.zeta = 15 # Distance we look ahead to determine the curvature of the road.

        self.kP_steer = 0.1
        # self.kD_steer = 1
        self.kD_steer = 0.1

        self.kP_throttle = 0.82954
        self.kD_throttle = 1.21341

        self.lookahead = 3.0  # Pure pursuit lookahead.
        self.vel_lookahead_factor = 40  # Inverse to the vel lookahead.
        ### 

        # Pure pursuit parameters
        self.delta = None
        self.kappa = None
        self.alpha = None
        self.target_vel = None

        self.cmd_steer = 0
        self.cmd_throttle = 0
        self.cmd_break = 0
        self.logger = Logger("runs/")

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
    
    def get_alpha(self, lookahead_dist):
        """
        Angle between the car and the point at lookahead_dist
        """
        # Fetch global coordinates of the lookahead point
        lookahead_x_global = self.centerline.Gx(self.progress + lookahead_dist)
        lookahead_y_global = self.centerline.Gy(self.progress + lookahead_dist)

        # Vehicle's current global position
        vehicle_x_global = self.centerline.Gx(self.progress)
        vehicle_y_global = self.centerline.Gy(self.progress)

        # Translate global coordinates by subtracting the vehicle's current position
        translated_x = lookahead_x_global - vehicle_x_global
        translated_y = lookahead_y_global - vehicle_y_global

        # Rotate translated coordinates into vehicle coordinates
        lookahead_x = translated_x * np.cos(-self.yaw) - translated_y * np.sin(-self.yaw)
        lookahead_y = translated_x * np.sin(-self.yaw) + translated_y * np.cos(-self.yaw)

        # Compute the angle of the lookahead point vector in vehicle coordinates
        angle = np.arctan2(lookahead_y, lookahead_x)  # In [-1, 1]

        return angle
    
    def pp_delta(self, lookahead):
        WHEELBASE = 2.87
        self.alpha = self.get_alpha(lookahead)
        self.delta = np.arctan2(2*WHEELBASE*np.sin(self.alpha), lookahead)
        return self.delta
    
    def get_target_vel(self, lookahead):
        lower = np.clip(self.progress+self.zeta, lookahead, 10000)
        self.kappa = self.centerline.mean_curvature(lower, lookahead)
        self.target_vel = np.clip(self.k*np.sqrt(9.81 / self.kappa), 10, 100)

        return self.target_vel, self.kappa

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary, simulation_time):
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
        self.last_ts = simulation_time - self.sim_time
        self.sim_time = simulation_time
        time.sleep(0.001)

        self.X = transform.location.x
        self.Y = transform.location.y
        old_yaw = self.yaw
        self.yaw = np.deg2rad(transform.rotation.yaw)
        if old_yaw is None:
            self.yawdot = None
        elif self.yaw - old_yaw > 3:
            self.yawdot = (self.yaw - (old_yaw + np.pi * 2) ) / self.last_ts
        elif self.yaw - old_yaw < -3:
            self.yawdot = (self.yaw - (old_yaw - np.pi * 2) ) / self.last_ts
        else:
            self.yawdot = (self.yaw - old_yaw) / self.last_ts

        self.left_lane_points = [(waypoint.transform.location.x, waypoint.transform.location.y) for waypoint in boundary[0]]
        self.right_lane_points = [(waypoint.transform.location.x, waypoint.transform.location.y) for waypoint in boundary[1]]
        
        self.next_left_lane_point_x = self.left_lane_points[0][0]
        self.next_left_lane_point_y = self.left_lane_points[0][1]
        
        self.next_right_lane_point_x = self.right_lane_points[0][0]
        self.next_right_lane_point_y = self.right_lane_points[0][1]

        # Velocities in vehicle coordinates (y is lateral).
        # Positive lateral velocity is to the left.
        self.vx = vel.x * np.cos(-self.yaw) - vel.y * np.sin(-self.yaw)
        self.vy = vel.x * np.sin(-self.yaw) + vel.y * np.cos(-self.yaw)
        self.vel = np.sqrt(self.vx**2 + self.vy**2)

        self.progress, dist = self.centerline.projection(self.X, self.Y, bounds=self.progress_bound())
        self.error = dist * self.centerline.error_sign(self.X, self.Y, self.progress)
        
        print(self.steps)
        print("progress: ", self.progress)

        control = carla.VehicleControl()
        control.steer = self.pp_delta(self.lookahead) + self.error*self.kP_steer + (self.last_error - self.error)*self.kD_steer

        vel_lookahead = (self.vel**2 / self.vel_lookahead_factor)
        target_vel, kappa = self.get_target_vel(vel_lookahead)
        throttle_error = (target_vel - self.vel)
        throttle = np.clip(throttle_error*self.kP_throttle + (throttle_error - self.last_throttle_error) * self.kD_throttle, -1, 1)
        if throttle < 0:
            control.brake = np.clip(-throttle, 0.5, 1)
            control.throttle = 0
        else:
            control.brake = 0
            control.throttle = throttle

        print("kappa: ", kappa)
        print("vel_lookahead: ", vel_lookahead)
        print("control: ", control.throttle, control.steer, control.brake)

        self.steps += 1
        self.last_error = self.error
        self.cmd_steer = control.steer
        self.cmd_throttle = control.throttle
        self.cmd_brake = control.brake
        self.last_throttle_error = throttle_error

        self.logger.log(self)

        return control

