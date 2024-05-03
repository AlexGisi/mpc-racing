from dataclasses import dataclass

@dataclass
class FixedControllerParameters:
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
    
    Ts: float = 0.05  # (s) todo: adjust this
    N: int = 40  # Prediction horizon
    lookahead_distance: float = N*Ts*v_max # (m)  how far the centerline/error polynomials are computed for


@dataclass
class RuntimeControllerParameters:
    # Runtime MPC parameters (Costa p8).
    alpha_c: float = 200
    d_max: float = 0.5
    q_v_y: float = 20
    n: int = 4  # \in {2, 4, 6, 9, ...}; See top of p8
    beta_delta: float = 400