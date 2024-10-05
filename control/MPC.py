from typing import Type, List
import casadi as ca
from models.State import State
from models.VehicleParameters import VehicleParameters
from control.ControllerParameters import FixedControllerParameters, RuntimeControllerParameters
from control.util import make_poly, deg2rad


class MPC:
    def __init__(self,
                 state0: Type[State],
                 s0: float, 
                 centerline_x_poly_coeffs: List[float],
                 centerline_y_poly_coeffs: List[float],
                 max_error: float,
                 runtime_params: Type[RuntimeControllerParameters],
                 sol0=None,  # return of previous iteration
                 duals=None,  # dual vars of previous
                 last_controls=None,
                 Ts=None,
                 N=None
                 ) -> None:
        """
        state: initial vehicle state
        s: initial vehicle progress
        centerline_{x,y}_poly_coeffs: coefficients of centerline polynomial (over ControllerParameters.lookahead_distance)
        params: runtime parameters for this instantiation of the problem
        sol0: last solution returned by opti.solve(), optional
        """
        opti = ca.Opti()

        self.fixed_params = FixedControllerParameters()
        self.runtime_params = runtime_params
        self.state0 = state0

        # Unpack variables for clarity.
        N = self.fixed_params.N if N is None else N
        n = self.runtime_params.n
        q_v_y = self.runtime_params.q_v_y
        alpha_c = self.runtime_params.alpha_c
        alpha_L = self.fixed_params.alpha_L
        beta_delta = self.runtime_params.beta_delta
        q_v_max = self.fixed_params.q_v_max
        v_max = self.fixed_params.v_max
        lambda_s = self.fixed_params.lambda_s

        min_steer = self.fixed_params.min_steer
        max_steer = self.fixed_params.max_steer
        min_throttle = self.fixed_params.min_throttle
        max_throttle = RuntimeControllerParameters.d_max
        max_steer_delta = self.fixed_params.max_steer_delta
        min_steer_delta = self.fixed_params.min_steer_delta
        max_throttle_delta = self.fixed_params.max_throttle_delta
        min_throttle_delta = self.fixed_params.min_throttle_delta
        

        Ts = FixedControllerParameters.Ts if Ts is None else Ts
        max_s_delta = Ts * FixedControllerParameters.v_max  # max progress per timestep
        # max_s_delta = Ts * VehicleParameters.max_vel  # max progress per timestep
        min_s_delta = 0.1

        # Decision variables. Column i is the <u/s/x> vector at time i. 
        U = opti.variable(2, N)  # throttle, steer in [-1, 1]
        S_hat = opti.variable(1, N+1)  # estimated progress (free)
        States = opti.variable(6, N+1) # X, Y, yaw, vx, vy, r

        # Symbols.
        s = ca.SX.sym('s') # Progress
        X = ca.SX.sym('X', 6, 1)  # State
        u = ca.SX.sym('u', 2, 1)  # Command

        centerline_x_poly = make_poly(s, centerline_x_poly_coeffs)
        centerline_y_poly = make_poly(s, centerline_y_poly_coeffs)

        Gx = ca.Function('Gx', [s], [centerline_x_poly])
        Gy = ca.Function('Gy', [s], [centerline_y_poly])
        dGx = ca.Function('dGx', [s], [ca.gradient(centerline_x_poly, s)])
        dGy = ca.Function('dGy', [s], [ca.gradient(centerline_y_poly, s)])

        e_hat_C = ca.Function('e_hat_C', [s, X], [dGy(s)*(X[0] - Gx(s)) - dGx(s)*(X[1] - Gy(s))])
        e_hat_L = ca.Function('e_hat_L', [s, X], [-dGx(s)*(X[0] - Gx(s)) - dGy(s)*(X[1] - Gy(s))])
        e_tot = ca.Function('t_tot', [s, X], [e_hat_C(s, X)**2 + e_hat_L(s, X)**2])

        f_vehicle = ca.Function('f_vehicle', [X, u], [self.f_vehicle(X, u, Ts=Ts)])
        
        # Cost function (terminal costs).
        J = -lambda_s*S_hat[N]
        J += q_v_y*States[4, N]**2 
        J += alpha_c*e_hat_C(S_hat[N], States[:, N])**n 
        J += alpha_L*e_hat_L(S_hat[N], States[:, N])**2
        J += ca.exp(q_v_max * (States[3, N] - v_max))

        # Cost function (stage costs). 
        for i in range(1, N):
            J += q_v_y*States[4, i]**2
            J += alpha_c*e_hat_C(S_hat[i], States[:, i])**n
            J += alpha_L*e_hat_L(S_hat[i], States[:, i])**2
            J += beta_delta*(U[1, i]-U[1, i-1])**2
            J += ca.exp(q_v_max*(States[3, i] - v_max))

        # Initial conditions.
        opti.subject_to(S_hat[0] == s0)
        opti.subject_to(States[0, 0] == state0.x)
        opti.subject_to(States[1, 0] == state0.y)
        opti.subject_to(States[2, 0] == state0.yaw)
        opti.subject_to(States[3, 0] == state0.v_x)
        opti.subject_to(States[4, 0] == state0.v_y)
        opti.subject_to(States[5, 0] == state0.yaw_dot)

        opti.set_initial(S_hat[0], s0)
        opti.set_initial(States[0, 0], state0.x)
        opti.set_initial(States[1, 0], state0.y)
        opti.set_initial(States[2, 0], state0.yaw)
        opti.set_initial(States[3, 0], state0.v_x)
        opti.set_initial(States[4, 0], state0.v_y)
        opti.set_initial(States[5, 0], state0.yaw_dot)

        # Constraints, convienent to do some basic initialization here for now.
        init_s0 = ca.vertcat(state0.x, state0.y, state0.yaw, state0.v_x, state0.v_y, state0.yaw_dot)

        if last_controls is not None:
            u_pred = ca.horzcat(*last_controls[1:], last_controls[-1])
        else:
            u_pred = ca.horzcat(*[(state0.throttle, state0.steer) for _ in range(N)])

        opti.set_initial(U, u_pred)
        for i in range(1, N+1):
            # opti.set_initial(S_hat[i], s0 + i*Ts*init_s0[3])
            opti.set_initial(S_hat[i], s0 + i*Ts*VehicleParameters.max_vel)  # todo

            pred = self.f_vehicle(init_s0, u_pred[:, i-1], Ts=Ts)
            opti.set_initial(States[:, i], pred)
            init_s0 = pred

            opti.subject_to(States[:, i] == f_vehicle(States[:, i-1], U[:, i-1]))
            opti.subject_to( opti.bounded(min_s_delta, S_hat[i] - S_hat[i-1], max_s_delta) )  # Bound progress estimation differences
            opti.subject_to( opti.bounded(-max_error, e_hat_C(S_hat[i], States[:, i]), max_error))  # Stay within the lane
            
        for i in range(0, N):
            opti.subject_to(U[0, i] < max_throttle)
            opti.subject_to(U[0, i] > min_throttle)
            opti.subject_to(U[1, i] < max_steer)
            opti.subject_to(U[1, i] > min_steer)
            opti.subject_to( opti.bounded(min_throttle_delta, U[0, i] - U[0, i-1], max_throttle_delta) )
            opti.subject_to( opti.bounded(min_steer_delta, U[1, i] - U[1, i-1], max_steer_delta) )

        if state0.throttle is not None:
            opti.subject_to( opti.bounded(min_throttle_delta, U[0, 0] - state0.throttle, max_throttle_delta) )

        if state0.steer is not None:
            opti.subject_to( opti.bounded(min_steer_delta, U[1, 0] - state0.steer, max_steer_delta) )

        opti.minimize(J)
        opti.solver('ipopt', {
            'ipopt': {
                'max_iter': FixedControllerParameters.max_iter,  # Maximum number of iterations
                'print_level': 4,  # Adjust to control the verbosity of IPOPT output
                'tol': 1e-4,  # Solver tolerance
                'acceptable_tol': 1e-2,
                'warm_start_init_point': 'no',
                'check_derivatives_for_naninf': 'yes',
            }
        })
        
        # Use same return structure so failures can be handled in the same way.
        try:
            self.sol = opti.solve()
            self.ret = (self.sol.value(States), 
                        self.sol.value(U), 
                        self.sol.value(S_hat), 
                        [self.sol.value(e_hat_C(S_hat[i], States[:, i])) for i in range(0, N)], 
                        [self.sol.value(e_hat_L(S_hat[i], States[:, i])) for i in range(N)])
            self.dual = self.sol.value(opti.lam_g)
        except RuntimeError as e:  # found infeasible solution or exceeded iterations or ...
            print(e)
            # breakpoint()
            self.ret = (opti.debug.value(States), 
                        opti.debug.value(U), 
                        opti.debug.value(S_hat), 
                        [opti.debug.value(e_hat_C(S_hat[i], States[:, i])) for i in range(N)], 
                        [opti.debug.value(e_hat_L(S_hat[i], States[:, i])) for i in range(N)])
            self.sol = None
            self.dual = None
    
    def solution(self):
        return self.sol, self.ret, self.dual

    def f_vehicle(self, x_k, u_k, Ts):
        """
        x_k: column of vehicle states
        u_k: column of commands (throttle, steer) in [-1, 1]
        Vehicle dynamics for state constraint.
        """
        # Unpack parameters.
        m = VehicleParameters.m
        Iz = VehicleParameters.Iz
        lf = VehicleParameters.lf
        lr = VehicleParameters.lr
        Cf = VehicleParameters.Cf
        Cr = VehicleParameters.Cr

        x, y, yaw, v_x, v_y, yaw_dot = x_k[0], x_k[1], x_k[2], x_k[3], x_k[4], x_k[5]

        # Convert commands to physical values.
        Fx = self.Fx(u_k[0], v_x)
        delta = self.steer_cmd_to_angle(u_k[1], v_x, v_y)

        # Calculate slip angles.
        theta_Vf = ca.atan2((v_y + lf * yaw_dot), v_x+0.1)
        theta_Vr = ca.atan2((v_y - lr * yaw_dot), v_x+0.1)

        # Calculate lateral forces at front and rear using linear tire model.
        Fyf = Cf * (delta - theta_Vf)
        Fyr = Cr * (-theta_Vr)

        # Dynamics equations
        # See "Online Learning of MPC for Autonomous Racing" by Costa et al
        v_x_dot = ( (Fx - Fyf*ca.sin(delta)) / m ) + (v_y * yaw_dot)
        v_y_dot = ((Fyf*ca.cos(delta) + Fyr) / m) - (v_x * yaw_dot)
        yaw_dot_dot = ( (Fyf*ca.cos(delta)*lf) - (Fyr*lr)) / Iz

        # Integrate to find new state
        x_new = x + (v_x * ca.cos(yaw) - v_y * ca.sin(yaw)) * Ts
        y_new = y + (v_x * ca.sin(yaw) + v_y * ca.cos(yaw)) * Ts
        yaw_new = yaw + yaw_dot * Ts
        v_x_new = v_x + v_x_dot * Ts
        v_y_new = v_y + v_y_dot * Ts
        yaw_dot_new = yaw_dot + yaw_dot_dot * Ts

        state_new = ca.vertcat(x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new)
        return state_new
    
    def f_vehicle_kinematic(self, x_k, u_k, Ts):
        """
        x_k: column of vehicle states
        u_k: column of commands (throttle, steer) in [-1, 1]
        Vehicle dynamics for state constraint.
        """
        # Unpack parameters.
        m = VehicleParameters.m
        Iz = VehicleParameters.Iz
        lf = VehicleParameters.lf
        lr = VehicleParameters.lr
        Cf = VehicleParameters.Cf
        Cr = VehicleParameters.Cr

        x, y, yaw, v_x, v_y, yaw_dot = x_k[0], x_k[1], x_k[2], x_k[3], x_k[4], x_k[5]

        # Convert commands to physical values.
        Fx = self.Fx(u_k[0], v_x)
        delta = self.steer_cmd_to_angle(u_k[1], v_x, v_y)

        # Integrate to find new state
        x_new = x + (v_x * ca.cos(yaw) - v_y * ca.sin(yaw)) * Ts
        y_new = y + (v_x * ca.sin(yaw) + v_y * ca.cos(yaw)) * Ts
        yaw_new = yaw + ( yaw_dot ) * Ts
        v_x_new = v_x + (Fx / m) * Ts
        v_y_new = yaw_dot * lr
        yaw_dot_new = (v_x / (lr + lf)) * ca.tan(delta)

        state_new = ca.vertcat(x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new)
        return state_new

    def steer_cmd_to_angle(self, steer_cmd, v_x, v_y):
        """
        Maps a steer command in [-1, 1] to the angle of the wheels over the next timestep.
        """
        vel = ca.sqrt(v_x**2 + v_y**2) * 3.6  # km/h

        # Define gain based on velocity via sim-defined steering curve.
        gain = -0.001971664699 * vel + 0.986547

        return deg2rad(steer_cmd * gain * VehicleParameters.max_steer)

    def Fx(self, throttle, v_x):
        wheel_rpm = (v_x / VehicleParameters.C_wheel) * 60
        rpm = wheel_rpm * VehicleParameters.R * 4.5  # 4.5 is an estimated adjust based on observation

        eta = -0.00004428225806 * rpm + 1.282413306
        
        wheel_force = throttle*eta*VehicleParameters.T_max*VehicleParameters.R / VehicleParameters.r_wheel
        drag_force = 0.5*VehicleParameters.rho*VehicleParameters.C_d*VehicleParameters.A_f*(v_x**2)
        rolling_resistance = VehicleParameters.C_roll*VehicleParameters.m*VehicleParameters.g

        return wheel_force - drag_force - rolling_resistance
