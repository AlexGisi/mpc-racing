from dataclasses import dataclass

@dataclass
class FixedControllerParameters:
    # Fixed MPC parameters (Costa p8).
    lambda_s: float = 300  # Reward on track progress at prediction horizon
    alpha_L: float = 500 # Penalty on lag approximation

    min_steer: float = -0.9
    max_steer: float = 0.9
    min_throttle: float = -1.0
    max_steer_delta: float = 0.2
    min_steer_delta: float = -0.2
    max_throttle_delta: float = 2.0  # todo: can increase these
    min_throttle_delta: float = -0.4

    q_v_max: float = 2  # Soft constraint on velocity
    v_max: float = 50
    
    Ts: float = 0.05  # (s), currently we actually use an adaptive timestep (see agent.py).
    N: int = 30  # Prediction horizon, currently ignored be set adaptively (see agent.py).
    lookahead_distance: float = N*Ts*v_max # (m), length that the centerline/error polynomials are computed.
    max_iter: int = 500

@dataclass
class RuntimeControllerParameters:
    # Runtime MPC parameters (Costa p8).
    alpha_c: float = 1000  # Penalty on centerline error
    d_max: float = 0.85  # Max throttle
    q_v_y: float = 50  # Penalty on lateral velocity
    n: int = 2  # Exponent on e_hat_C; \in {2, 4, 6, 9, ...}; See top of p8
    beta_delta: float = 5000  # Penalty on difference in steering
