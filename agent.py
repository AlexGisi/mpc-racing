from time import time
import carla
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from Logger import Logger
from GA.pdGA import GeneticAlgorithm
from control.ControllerParameters import (
    RuntimeControllerParameters,
    FixedControllerParameters,
)
from models.State import State
from control.MPC import MPC
from models.VehicleParameters import VehicleParameters


class Agent:
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.cl = ParameterizedCenterline("shanghai_intl_circuit")
        self.progress = None  # Assume we don't know initial vehicle location.

        # Genetic algorithm
        # self.genetic_algorithm = GeneticAlgorithm()
        self.segment_count = 0
        self.population_index = 0
        self.total_segments = 20  ## update from GA
        self.curr_segtime_start = None

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

        self.cmd_steer = 0
        self.cmd_throttle = 0
        self.cmd_break = 0
        self.logger = Logger(runs_fp='runs/')

        # For pp control
        self.delta = None
        self.kappa = None
        self.alpha = None
        self.lookahead = None
        self.target_vel = None

        # For mpc control
        self.start_control_at = 50
        self.runtime_params = RuntimeControllerParameters()
        self.predicted_states = None  # Includes initial state
        self.controls = None
        self.mean_ts = 0.3
        self.mpc_time = 0

    def progress_bound(self):
        """
        Return a bound on the possible progress of the vehicle in order. Such a bound
        is required in order to use the ParameterizedCenterline projection method, but
        it need not be tight.
        """
        if self.progress is None:
            return None

        lower = (self.progress - 2) % self.cl.length
        upper = (self.progress + 2) % self.cl.length

        return (lower, upper) if lower < upper else (upper, lower)

    def get_alpha(self, lookahead_dist):
        """
        Angle between the car's heading and the point at lookahead_dist
        """
        # Fetch global coordinates of the lookahead point
        lookahead_x_global = self.cl.Gx(self.progress + lookahead_dist)
        lookahead_y_global = self.cl.Gy(self.progress + lookahead_dist)

        # Vehicle's current global position
        vehicle_x_global = self.cl.Gx(self.progress)
        vehicle_y_global = self.cl.Gy(self.progress)

        # Translate global coordinates by subtracting the vehicle's current position
        translated_x = lookahead_x_global - vehicle_x_global
        translated_y = lookahead_y_global - vehicle_y_global

        # Rotate translated coordinates into vehicle coordinates
        lookahead_x = translated_x * np.cos(-self.yaw) - translated_y * np.sin(
            -self.yaw
        )
        lookahead_y = translated_x * np.sin(-self.yaw) + translated_y * np.cos(
            -self.yaw
        )

        # Compute the angle of the lookahead point vector in vehicle coordinates
        angle = np.arctan2(lookahead_y, lookahead_x)  # In [-1, 1]

        return angle

    # TODO: could use for initializing mpc
    def pp_delta(self, lookahead):
        WHEELBASE = 2.87
        self.alpha = self.get_alpha(lookahead)
        self.delta = np.arctan2(2 * WHEELBASE * np.sin(self.alpha), lookahead)
        return self.delta

    def get_target_vel(self, lookahead, zeta=15):
        lower = np.clip(self.progress + zeta, lookahead, 10000)
        k = 1.2
        self.kappa = self.cl.mean_curvature(lower, lookahead)
        self.target_vel = np.clip(k * np.sqrt(9.81 / self.kappa), 10, 100)

        return self.target_vel, self.kappa

    def run_mpc(self):
        POLY_LOOKBACK = 5
        POLY_DEG = 4
        LOOKAHEAD = 20
        state0 = State(
            x=self.X,
            y=self.Y,
            yaw=self.yaw,
            v_x=self.vx,
            v_y=self.vy,
            yaw_dot=self.yawdot,
            throttle=self.cmd_throttle,
            steer=self.cmd_steer,
        )
        # N = int(np.ceil(LOOKAHEAD / (self.mean_ts * (state0.v_x))))
        # N = np.clip(N, 5, 9)
        N = 8
        print(N)
        cl_x_coeffs = self.cl.x_as_coeffs(
            s=self.progress - POLY_LOOKBACK, 
            lookahead=LOOKAHEAD + 25, 
            deg=POLY_DEG,
        )
        cl_y_coeffs = self.cl.y_as_coeffs(
            s=self.progress - POLY_LOOKBACK, 
            lookahead=LOOKAHEAD + 25, 
            deg=POLY_DEG,
        )
        max_err = self.cl.lookup_error(self.progress, LOOKAHEAD + 25) - (
            VehicleParameters.car_width / 2
        )

        start_time = time()
        mpc = MPC(
            state0=state0,
            sol0=None,
            duals=None,
            s0=self.progress,
            centerline_x_poly_coeffs=cl_x_coeffs,
            centerline_y_poly_coeffs=cl_y_coeffs,
            max_error=max_err,
            runtime_params=self.runtime_params,
            Ts=self.mean_ts,
            N=N,
        )
        sol, ret, duals = mpc.solution()
        self.mpc_time = time() - start_time

        States, U, S_hat, e_hat_c, e_hat_l = ret[0], ret[1], ret[2], ret[3], ret[4]
        self.predicted_states = [
            State(x=t[0], y=t[1], yaw=t[2], v_x=t[3], v_y=t[4], yaw_dot=t[5])
            for t in zip(
                States[0, :],
                States[1, :],
                States[2, :],
                States[3, :],
                States[4, :],
                States[5, :],
            )
        ]
        self.controls = [
            (throttle, steer) for throttle, steer in zip(U[0, :], U[1, :])
        ][
            1:
        ]  # Ignore the fixed initial command

    def run_step(
        self, filtered_obstacles, waypoints, vel, transform, boundary, simulation_time
    ):
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
        ### Data collection and transformation
        self.last_ts = simulation_time - self.sim_time
        self.sim_time = simulation_time
        
        self.X = transform.location.x
        self.Y = transform.location.y
        if self.yaw is None:
            self.yawdot = 0
        else:
            self.yawdot = np.clip((np.deg2rad(transform.rotation.yaw) - self.yaw) / self.last_ts, -1, 1)
        self.yaw = np.deg2rad(transform.rotation.yaw)

        # Velocities in vehicle coordinates (y is lateral). Positive lateral velocity is to the left.
        self.vx = vel.x * np.cos(-self.yaw) - vel.y * np.sin(-self.yaw)
        self.vy = vel.x * np.sin(-self.yaw) + vel.y * np.cos(-self.yaw)
        self.vel = np.sqrt(self.vx**2 + self.vy**2)
        # self.yawdot = (
        #     self.vx / (VehicleParameters.lr + VehicleParameters.lf)
        # ) * np.tan(
        #     self.cmd_steer
        # )  # Estimate yawdot geometrically.

        self.left_lane_points = [
            (waypoint.transform.location.x, waypoint.transform.location.y)
            for waypoint in boundary[0]
        ]
        self.right_lane_points = [
            (waypoint.transform.location.x, waypoint.transform.location.y)
            for waypoint in boundary[1]
        ]

        self.next_left_lane_point_x = self.left_lane_points[0][0]
        self.next_left_lane_point_y = self.left_lane_points[0][1]

        self.next_right_lane_point_x = self.right_lane_points[0][0]
        self.next_right_lane_point_y = self.right_lane_points[0][1]

        self.progress, dist = self.cl.projection(
            self.X, self.Y, bounds=self.progress_bound()
        )
        self.error = dist * self.cl.error_sign(self.X, self.Y, self.progress)
        ###

        print("step ", self.steps)

        # 
        print(f"progress {self.progress}")
        if self.steps > 50:
            self.logger.log(self)
            if self.steps % 1 == 0 or self.controls is None:
                self.run_mpc()  # Sets self.predicted_states, self.controls
                self.logger.pickle_mpc_res(self)
            throttle, steer = self.controls.pop(0)
        else:
            throttle, steer = 0.5, 0.0
        
        self.mean_ts = self.mean_ts + ((self.last_ts - self.mean_ts) / (self.steps + 1))
        ###

        control = carla.VehicleControl()
        control.steer = steer
        if throttle < 0:
            control.brake = -throttle
            control.throttle = 0
        else:
            control.brake = 0
            control.throttle = throttle

        print("control: ", control.throttle, control.steer, control.brake)

        self.steps += 1
        self.last_error = self.error
        self.cmd_steer = control.steer
        self.cmd_throttle = control.throttle
        self.cmd_brake = control.brake

        if self.steps <= 50:
            self.logger.log(self)

        return control
