import carla
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from Logger import Logger
from GA.pdGA import GeneticAlgorithm
from control.ControllerParameters import RuntimeControllerParameters, FixedControllerParameters
from models.State import State
from control.MPC import MPC
from models.VehicleParameters import VehicleParameters

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.cl = ParameterizedCenterline()
        self.cl.from_file("/home/alex/Projects/graic/autobots-race/waypoints/shanghai_intl_circuit")
        self.progress = None  # Car is spawned somwhere in the middle of the track.

        # Genetic algorithm
        # self.genetic_algorithm = GeneticAlgorithm()
        self.segment_count = 0
        self.population_index = 0
        self.total_segments = 20 ## update from GA
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
        self.logger = Logger("run/data-mydrive.csv", mpc_fp="run/mpc")

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

    def progress_bound(self):
        """
        Return a bound on the possible progress of the vehicle in order. Such a bound
        is required in order to use the ParameterizedCenterline projection method, but
        it need not be tight.
        """
        if self.progress is None:
            return None
        
        lower = (self.progress-2)  % self.cl.length
        upper = (self.progress+2) % self.cl.length

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
        lookahead_x = translated_x * np.cos(-self.yaw) - translated_y * np.sin(-self.yaw)
        lookahead_y = translated_x * np.sin(-self.yaw) + translated_y * np.cos(-self.yaw)

        # Compute the angle of the lookahead point vector in vehicle coordinates
        angle = np.arctan2(lookahead_y, lookahead_x)  # In [-1, 1]

        return angle
    
    # TODO: could use for initializing mpc
    def pp_delta(self, lookahead):
        WHEELBASE = 2.87
        self.alpha = self.get_alpha(lookahead)
        self.delta = np.arctan2(2*WHEELBASE*np.sin(self.alpha), lookahead)
        return self.delta
    
    def get_target_vel(self, lookahead, zeta=15):
        lower = np.clip(self.progress+zeta, lookahead, 10000)
        k = 1.2
        self.kappa = self.cl.mean_curvature(lower, lookahead)
        self.target_vel = np.clip(k*np.sqrt(9.81 / self.kappa), 10, 100)

        return self.target_vel, self.kappa
    
    def run_mpc(self):
        POLY_LOOKBACK = 5
        POLY_DEG = 4
        LOOKAHEAD = 30
        state0 = State(x=self.X, y=self.Y, yaw=self.yaw, v_x=self.vx, v_y=self.vy, yaw_dot=self.yawdot, 
                       throttle=self.cmd_throttle, steer=self.cmd_steer)
        N = int(np.ceil(LOOKAHEAD / (self.mean_ts * (state0.v_x))))
        N = np.clip(N, 3, 6)
        cl_x_coeffs = self.cl.x_as_coeffs(self.progress-POLY_LOOKBACK, LOOKAHEAD+25, deg=POLY_DEG)
        cl_y_coeffs = self.cl.y_as_coeffs(self.progress-POLY_LOOKBACK, LOOKAHEAD+25, deg=POLY_DEG)
        max_err = self.cl.lookup_error(self.progress, LOOKAHEAD+25) - (VehicleParameters.car_width / 2)

        mpc = MPC(state0=state0,
                  sol0=None,
                  duals=None,
                  s0=self.progress,
                  centerline_x_poly_coeffs=cl_x_coeffs,
                  centerline_y_poly_coeffs=cl_y_coeffs,
                  max_error=max_err,
                  runtime_params=self.runtime_params,
                  Ts=self.mean_ts,
                  N=N)

        sol, ret, duals = mpc.solution()
        States, U, S_hat, e_hat_c, e_hat_l = ret[0], ret[1], ret[2], ret[3], ret[4]
        self.predicted_states = [State(x=t[0], y=t[1], yaw=t[2], v_x=t[3], v_y=t[4], yaw_dot=t[5]) for t in zip(States[0, :], States[1, :], States[2, :], States[3, :], States[4, :], States[5, :])]
        self.controls = [(throttle, steer) for throttle, steer in zip(U[0, :], U[1, :])][1:] # Ignore the fixed initial command 

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
        ### Data collection and transformation
        self.X = transform.location.x
        self.Y = transform.location.y
        self.yaw = np.deg2rad(transform.rotation.yaw)

        # Velocities in vehicle coordinates (y is lateral). Positive lateral velocity is to the left.
        self.vx = vel.x * np.cos(-self.yaw) - vel.y * np.sin(-self.yaw)
        self.vy = vel.x * np.sin(-self.yaw) + vel.y * np.cos(-self.yaw)
        self.vel = np.sqrt(self.vx**2 + self.vy**2)
        self.yawdot = (self.vx / (VehicleParameters.lr + VehicleParameters.lf)) * np.tan(self.cmd_steer)  # Estimate yawdot geometrically.


        self.left_lane_points = [(waypoint.transform.location.x, waypoint.transform.location.y) for waypoint in boundary[0]]
        self.right_lane_points = [(waypoint.transform.location.x, waypoint.transform.location.y) for waypoint in boundary[1]]
        
        self.next_left_lane_point_x = self.left_lane_points[0][0]
        self.next_left_lane_point_y = self.left_lane_points[0][1]
        
        self.next_right_lane_point_x = self.right_lane_points[0][0]
        self.next_right_lane_point_y = self.right_lane_points[0][1]

        self.progress, dist = self.cl.projection(self.X, self.Y, bounds=self.progress_bound())
        self.error = dist * self.cl.error_sign(self.X, self.Y, self.progress)
        
        self.last_ts = simulation_time - self.sim_time
        self.sim_time = simulation_time
        ###
        
        print(self.steps)

        ### GA 
        # if (self.curr_segtime_start == None):
        #     self.curr_segtime_start = simulation_time

        # if self.genetic_algorithm.getSegmentNumber(self.X, self.Y) != self.segment_count:
        #     past_segment_time = simulation_time - self.curr_segtime_start
        #     self.genetic_algorithm.updateSegmentTimings(self.population_index, past_segment_time)
        #     
        #     if self.population_index >= self.genetic_algorithm.getPopSize():
        #         print("Genetic Algorithm Updating Genes:")
        #         self.genetic_algorithm.runStep()
        #         self.population_index = 0
        #         self.curr_segtime_start = simulation_time
        #     else:
        #         self.population_index += 1

        # TODO: get RuntimeParams from GA
        # current_controller = self.genetic_algorithm.getPop(self.population_index)

        ###
                
        ### Control computation
        # if self.steps % 10 == 0:  # Run open-loop for half a second

        # TODO: just a fix for weird interpolation at beginning, need to fix that
        if self.progress > 20 and self.progress < 1000:
            if self.steps % 1 == 0:
                self.run_mpc()  # Sets self.predicted_states, self.controls
                self.logger.pickle_mpc_res(self)
                
            throttle, steer = self.controls.pop(0)
            self.mean_ts = self.mean_ts + ( (self.last_ts - self.mean_ts) / self.steps)
        else:
            steer = 0
            throttle = 0.5
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

        self.logger.log(self)

        return control
