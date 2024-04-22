from dataclasses import dataclass

@dataclass
class FixedControllerParameters:
    lookahead_distance: float = 100 # (m)  how far the centerline/error polynomials are computed for

    # Fixed MPC parameters (Costa p8).
    lambda_s: float = 200  # Weight of track progress at prediction horizon
    alpha_L: float = 1000 # Weight on lag approximation
    q_v_max: float = 5
    v_max: float = 30
    Delta_d_max: float = 0.2
    Delta_delta_max: float = 0.075
    delta_max: float = 0.47

    # Constraints on change in progress, should be
    # recalculated based on model 3 top speed (para 3 p8) (TODO).
    Delta_s_min: float = 0.1
    Delta_s_max: float = 1.5

    e_hat_CL_max: float = 1
    
    T_s: float = 0.1  # (s) todo: adjust this
    N: int = 40  # Prediction horizon

@dataclass
class RuntimeControllerParameters:
    # Runtime MPC parameters (Costa p8).
    alpha_c: float
    d_max: float
    q_v_y: float
    n: int  # \in {2, 4, 6, 9, ...}; See top of p8
    beta_delta: float