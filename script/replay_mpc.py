"""
Visualize the results of a run where the mpc results were
recorded.
"""
import pickle
import os
from math import floor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from control.ControllerParameters import RuntimeControllerParameters, FixedControllerParameters
from control.util import make_poly
from models.State import State
from models.VehicleParameters import VehicleParameters

START_NUM = 1
STEP_MODULO = 2
DATA_DIR = 'runs/mpc-run-simple'
POLY_DEG = 4
POLY_LOOKBACK = 5

cl = ParameterizedCenterline()
df = pd.read_csv(os.path.join(DATA_DIR, 'data-mydrive.csv'))
mpcs = []
file_numbers = sorted([ int(f) for f in list(os.walk(os.path.join(DATA_DIR, 'mpc')))[0][2] ])
for step_num in file_numbers:
    with open(os.path.join(DATA_DIR, 'mpc', str(step_num)), 'rb') as f:
        step_dict = pickle.load(f)
        mpcs.append(step_dict)

# start_idx = floor((START_NUM-file_numbers[0]) / STEP_MODULO)
for i, mpc in enumerate(mpcs[START_NUM:]):
    print(f"controlled: {mpc['controlled']}")

    step_df = df[(df['steps'] >= mpc['step'])]
    state0 = mpc['predicted_states'][0]
    states = mpc['predicted_states']
    controls = mpc['controls']
    steps = range(int(step_df.iloc[0]['steps']), np.max(step_df['steps'])+1)
    s0 = step_df.iloc[0]['progress']

    dynamic_lookahead = FixedControllerParameters.Ts * (state0.v_x + (15/40)*FixedControllerParameters.N) * FixedControllerParameters.N
    cl_x_coeffs = cl.x_as_coeffs(s0-POLY_LOOKBACK, dynamic_lookahead, deg=POLY_DEG)
    cl_y_coeffs = cl.y_as_coeffs(s0-POLY_LOOKBACK, dynamic_lookahead, deg=POLY_DEG)
    max_err = cl.lookup_error(s0, dynamic_lookahead) - (VehicleParameters.car_width / 2)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3)

    # Assign subplots to the grid
    ax1 = fig.add_subplot(gs[0, :])  # Makes ax1 span all columns in the first row
    ax2 = fig.add_subplot(gs[1, 0])  # Remaining subplots in specified positions
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    cl_x = lambda s: make_poly(s, cl_x_coeffs)
    cl_y = lambda s: make_poly(s, cl_y_coeffs)
    ss = np.linspace(s0-5, s0+dynamic_lookahead, 25)
    uprs = [np.array(cl.unit_principal_normal(s)) for s in ss]
    ax1.plot([cl_x(s) for s in ss], [cl_y(s) for s in ss], linewidth=2, label=f"{POLY_DEG}th degree local interpolation")

    ax1.plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r')
    ax1.plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r')
    ax1.plot([cl.Gx(s)+upr[0]*max_err for s, upr in zip(ss, uprs)], [cl.Gy(s)+upr[1]*max_err for s, upr in zip(ss, uprs)], linestyle=':', color='r', label="Error bound")
    ax1.plot([cl.Gx(s)+upr[0]*-max_err for s, upr in zip(ss, uprs)], [cl.Gy(s)+upr[1]*-max_err for s, upr in zip(ss, uprs)], linestyle=':', color='r')
    ax1.plot(state0.x, state0.y, marker='o', zorder=100, color='r', ms=6, label="Initial position")
    ax1.scatter([s.x for s in states], [s.y for s in states], zorder=50, s=2, linewidths=4, marker='o', linestyle='-', color='r', label="Predicted path")
    ax1.scatter(step_df['X'], step_df['Y'], zorder=49, s=2, linewidths=4, marker='o', linestyle='-', color='g', label="Actual path")

    ax1.set_title(r"States ($s_0$=" + str(s0) + ")")
    ax1.set_xlim(state0.x-80, state0.x+80)
    ax1.set_ylim(np.min([s.y for s in states])-20, np.max([s.y for s in states])+20)
    ax1.set_aspect('equal')

    ax1.legend(loc='upper center', fancybox=True, shadow=True, ncol=5, fontsize=10)

    ax2.plot(range(len(controls)), [s for t, s in controls], label="Steer")
    try:
        ax2.set_title(f"Generated commands (in {round(mpc['time'], 4)})")
    except KeyError:
        ax2.set_title(f"Generated commands")
    ax2.set_xlabel("Step")
    ax2.grid(True)
    ax2.plot(range(len(controls)), [t for t, s in controls], label="Throttle")
    ax2.set_ylim(-1, 1)
    ax2.legend()

    ax3.plot(range(len(states)), [state.v_x for state in states], label="v_x")
    ax3.plot(range(len(states)), [state.v_y for state in states], label="v_y")
    ax3.set_title("Predicted velocities")
    ax3.set_xlabel("Step")
    ax3.legend()
    ax3.grid(True)

    ax4.plot(steps, step_df['last_ts'])
    ax4.set_title(f"Simulation time steps (mean {round(mpc['mean_ts'], 3)})")
    ax4.grid(True)

    print(state0)
    print(s0)
    print(controls[0])
    print("step ", file_numbers[i+START_NUM])

    # ax4.plot(range(len(S_hat)), S_hat, label="$\hat{s}$")
    # ax4.plot(range(len(states)), [cl.projection_local(s.x, s.y, bounds=(s0-10, s0+100), warn=False)[0] for s in states], label="s")
    # ax4.set_xlabel("Step")
    # ax4.grid(True)
    # ax4.legend()
    # ax4.set_title(r"Estimated and actual centerline progress")

    # ax6.plot(range(len(e_hat_l)), e_hat_l, label=r"$\hat{e}_l$")
    # ax6.plot(range(len(e_hat_c)), e_hat_c, label=r"$\hat{e}_c$")
    # ax6.plot(range(len(e_hat_c)), [-max_err for _ in range(len(e_hat_c))], label="Min error", color='r')
    # ax6.plot(range(len(e_hat_c)), [max_err for _ in range(len(e_hat_c))], label="Max error", color='r')
    # ax6.set_title("Estimated centerline and lag errors")
    # ax6.set_xlabel("Step")
    # ax6.legend()
    # ax6.grid(True)

    plt.tight_layout()
    plt.show()
