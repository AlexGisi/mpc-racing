import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from learning.vehicle import Vehicle
from learning.util import get_abs_fp, load_dataset
from models.VehicleParameters import VehicleParameters


###
FP = "data/uniform-vy/validate.csv"

FEATURES = [
    "X_0",
    "Y_0",
    "yaw_0",
    "vx_0",
    "vy_0",
    "yawdot_0",
    "throttle_0",
    "steer_0",
    "last_ts",
]
TARGETS = [
    "X_1",
    "Y_1",
    "yaw_1",
    "vx_1",
    "vy_1",
    "yawdot_1",
]
###


def state_to_slip(x):
    lf = VehicleParameters.lf
    lr = VehicleParameters.lr
    
    X, Y, yaw, v_x, v_y, yaw_dot, dt = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 8]
    theta_Vf = torch.atan2((v_y + lf * yaw_dot), v_x)
    theta_Vr = torch.atan2((v_y - lr * yaw_dot), v_x)

    alphas = torch.stack([theta_Vf, theta_Vr], dim=1)
    return alphas  # (len(x), 2)


with torch.no_grad():
    X, y = load_dataset(get_abs_fp(__file__, FP), FEATURES, TARGETS, 'cpu', torch.float32)
    ay = (y[:, 4] - X[:, 4]) / X[:, 6]
    # Fy = VehicleParameters.m * ay
    Fy = ay

    alpha = state_to_slip(X)

    Fy = Fy.cpu().numpy()
    alpha = alpha.cpu().numpy()
    alpha = np.rad2deg(alpha)

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].scatter(alpha[:, 0], Fy)
axs[0].set_title('front tire slip curve')
axs[0].set_xlabel('side slip angle')
axs[0].set_ylabel('fy')
axs[0].grid(True)

axs[1].scatter(alpha[:, 1], Fy)
axs[1].set_title('back tire slip curve')
axs[1].set_xlabel('side slip angle')
axs[1].set_ylabel('fy')
axs[1].grid(True)

plt.show()
