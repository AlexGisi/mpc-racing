import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.State import State
from splines.ParameterizedCenterline import ParameterizedCenterline

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

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
    "X": [],
    "Y": [],
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
    errors["X"].append(state1_data.x - state1_mpc.x)
    errors["Y"].append(state1_data.y - state1_mpc.y)
    errors["yaw"].append(state1_data.yaw - state1_mpc.yaw)
    errors["v_x"].append(state1_data.v_x - state1_mpc.v_x)
    errors["v_y"].append(state1_data.v_y - state1_mpc.v_y)
    errors["yaw_dot"].append(state1_data.yaw_dot - state1_mpc.yaw_dot)

errors_df = pd.DataFrame(errors)

statistics = errors_df.drop(labels="step", axis=1).describe()
print(statistics)

features = ["X", "Y", "yaw", "v_x", "v_y", "yaw_dot"]
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

worst_steps = errors_df.idxmax()
for f in features:
    idx = worst_steps[f]
    step = errors_df.at[idx, 'step']
    step_n, mpc = mpcs[step+mpcs[0][0]]
    assert step == step_n

    pred_states = mpc['predicted_states']
    pred_n = min(len(run_df)-step-1, len(pred_states))
    real_states = run_df.iloc[step:step+pred_n+1, :]

    plt.scatter([s.x for s in pred_states], [s.y for s in pred_states], marker='x', label='pred')
    for s in pred_states:
        plt.arrow(s.x, s.y, 0.5*np.cos(s.yaw), 0.5*np.sin(s.yaw))

    plt.scatter(real_states['X'], real_states['Y'], label='real')
    for x,y,yaw in zip(real_states['X'], real_states['Y'], real_states['yaw']):
        plt.arrow(x, y, 0.5*np.cos(yaw), 0.5*np.sin(yaw))

    # Plot centerline and lanes.
    ss = np.linspace(real_states.at[step, 'progress']-200, real_states.at[step+pred_n, 'progress']+200, 1000)
    plt.plot([cl.left_lane.Gx(s) for s in ss], [cl.left_lane.Gy(s) for s in ss], 'b')
    plt.plot([cl.right_lane.Gx(s) for s in ss], [cl.right_lane.Gy(s) for s in ss], 'b')
    plt.plot([cl.Gx(s) for s in ss], [cl.Gy(s) for s in ss], 'b--')

    plt.title(f"worst for {f} at step {step} ({round(errors_df.at[idx, f], 4)})")
    plt.legend()
    plt.show()
