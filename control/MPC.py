from typing import Type, List
import casadi as ca
from models.State import State
from splines.ParameterizedCenterline import ParameterizedCenterline
from control.ControllerParameters import FixedControllerParameters, RuntimeControllerParameters
from control.util import make_poly,



class MPC:
    def __init__(self, 
                 state: Type[State],
                 s: float, 
                 centerline_x_poly_coeffs: List[float],
                 centerline_y_poly_coeffs: List[float],
                 error_poly_coeffs: List[float], 
                 runtime_params: Type[RuntimeControllerParameters]) -> None:
        """
        state: initial vehicle state
        s: initial vehicle progress
        centerline_{x,y}_poly_coeffs: coefficients of centerline polynomial (over ControllerParameters.lookahead_distance)
        error_poly_coeffs: coefficients of min allowable error (over ControllerParameters.lookahead_distance)
        params: runtime parameters for this instantiation of the problem
        """
        opti = ca.Opti()

        self.fixed_params = FixedControllerParameters()
        self.runtime_params = runtime_params
        self.state0 = state

        # Unpack variables for clarity.
        N = self.fixed_params.N
        q_v_y = self.runtime_params.q_v_y
        alpha_c = self.runtime_params.alpha_c
        alpha_L = self.fixed_params.alpha_L
        beta_delta = self.runtime_params.beta_delta
        q_v_max = self.fixed_params.q_v_max
        v_max = self.fixed_params.v_max
        lambda_s = self.fixed_params.lambda_s

        min_steer_angle = None  # These need to be identified!!!
        max_steer_angle = None
        min_acceleration = None
        max_acceleration = None
        min_steer_angle_Delta = None  # These need to be chosen!!!
        max_steer_angle_Delta = None
        min_acceleration_Delta = None
        max_acceleration_Delta = None

        # Decision variables. Column i is the <u/s/x> vector at time i. 
        U = opti.variable(2, N)
        S = opti.variable(1, N+1)
        States = opti.variable(6, N+1) # X, Y, yaw, vx, vy, r

        # Symbols.
        s = ca.SX.sym('s') # Progress
        X = ca.SX.sym('X', 6, 1)  # State

        centerline_x_poly = make_poly(s, centerline_x_poly_coeffs)
        centerline_y_poly = make_poly(s, centerline_y_poly_coeffs)

        Gx = ca.Function('Gx', [s], [centerline_x_poly])
        Gy = ca.Function('Gy', [s], [centerline_y_poly])
        dGx = ca.Function('dGx', [s], [ca.gradient(centerline_x_poly, s)])
        dGy = ca.Function('dGy', [s], [ca.gradient(centerline_y_poly, s)])

        e_hat_C = ca.Function('e_hat_C', [s, X], [dGy(s)*(X(0,0) - Gx(s)) - dGx(s)*(X(1,0) - Gy(s))])
        e_hat_L = ca.Function('e_hat_L', [s, X], [-dGx(s)*(X(0, 0) - Gx(s)) - dGy(s)*(X(1, 0) - Gy(s))])
        
        # Initial conditions.
        opti.subject_to(S[0, 0] == 0)

        opti.subject_to(X[0, 0] == state.x)
        opti.subject_to(X[1, 0] == state.y)
        opti.subject_to(X[2, 0] == state.yaw)
        opti.subject_to(X[3, 0] == state.v_x)
        opti.subject_to(X[4, 0] == state.v_y)
        opti.subject_to(X[5, 0] == state.yaw_dot)

        # Cost function (terminal costs).

        # Cost function (stage costs). 

        # Constraints.
        
