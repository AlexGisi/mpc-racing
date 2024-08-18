import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.State import State

RUN_DIR = sys.argv[1]
run_fp = os.path.join(RUN_DIR, "steps.csv")
run_df = pd.read_csv(run_fp)

row_to_state = lambda r: State(
    x=r["X"],
    y=r["Y"],
    yaw=r["yaw"],
    v_x=r["vx"],
    v_y=r["vy"],
    yaw_dot=r["yawdot"],
    steer=r["cmd_steer"],
    throttle=r["cmd_throttle"] - r["cmd_brake"],
)

mpcs = []
file_numbers = sorted(
    [int(f) for f in list(os.walk(os.path.join(RUN_DIR, "mpc")))[0][2]]
)
for step_num in file_numbers:
    with open(os.path.join(RUN_DIR, "mpc", str(step_num)), "rb") as f:
        step_dict = pickle.load(f)
        mpcs.append((step_num, step_dict))

errors = {
    "step": [],
    "x": [],
    "y": [],
    "yaw": [],
    "v_x": [],
    "v_y": [],
    "yaw_dot": [],
}
for step, mpc in mpcs[:-1]:
    if not mpc["controlled"]:
        continue

    state0_mpc = mpc["predicted_states"][0].set_controls(*mpc["controls"][0])
    state1_mpc = mpc["predicted_states"][1].set_controls(*mpc["controls"][1])
    state0_data = row_to_state(run_df.iloc[step, :])
    state1_data = row_to_state(run_df.iloc[step + 1, :])

    errors["step"].append(step)
    errors["x"].append(state1_data.x - state1_mpc.x)
    errors["y"].append(state1_data.y - state1_mpc.y)
    errors["yaw"].append(state1_data.yaw - state1_mpc.yaw)
    errors["v_x"].append(state1_data.v_x - state1_mpc.v_x)
    errors["v_y"].append(state1_data.v_y - state1_mpc.v_y)
    errors["yaw_dot"].append(state1_data.yaw_dot - state1_mpc.yaw_dot)

errors_df = pd.DataFrame(errors)

statistics = errors_df.drop(labels="step", axis=1).describe()
print(statistics)

features = ["x", "y", "yaw", "v_x", "v_y", "yaw_dot"]
num_features = len(features)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3

    max_abs_error = abs(errors_df[feature]).max()
    x_range = (-max_abs_error, max_abs_error)

    axs[row, col].hist(
        errors_df[feature],
        # bins=np.arange(x_range[0], x_range[1] + 0.1, 0.1),
        bins=51,
        color="blue",
        alpha=0.7,
        range=x_range,
        edgecolor="black",
        linewidth=0.2,
    )
    axs[row, col].set_title(f"{feature} errors")
    axs[row, col].set_xlabel("error")
    axs[row, col].set_ylabel("frequency")
    axs[row, col].set_xlim(x_range)
    # axs[row, col].grid(True)

plt.tight_layout()
plt.show()
