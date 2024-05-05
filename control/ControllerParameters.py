from dataclasses import dataclass

@dataclass
class FixedControllerParameters:
    # Fixed MPC parameters (Costa p8).
    lambda_s: float = 500  # Reward on track progress at prediction horizon
    alpha_L: float = 2000 # Penalty on lag approximation
    
    q_v_max: float = 5  # Soft constraint on velocity
    v_max: float = 30
    Delta_d_max: float = 0.2
    Delta_delta_max: float = 0.075
    delta_max: float = 0.47

    # Constraints on change in progress, should be
    # recalculated based on model 3 top speed (para 3 p8) (TODO).
    Delta_s_min: float = 0.1
    Delta_s_max: float = 1.5
    
    Ts: float = 0.05  # (s)
    N: int = 30  # Prediction horizon
    lookahead_distance: float = N*Ts*v_max # (m)  how far the centerline/error polynomials are computed for
    max_iter: int = 500

@dataclass
class RuntimeControllerParameters:
    # Runtime MPC parameters (Costa p8).
    alpha_c: float = 650  # Penalty on centerline error
    d_max: float = 0.8  # Max throttle
    q_v_y: float = 100  # Penalty on lateral acceleration
    n: int = 4  # Exponent on e_hat_C; \in {2, 4, 6, 9, ...}; See top of p8
    beta_delta: float = 50000  # Penalty on difference in steering
