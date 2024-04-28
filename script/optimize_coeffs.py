"""
For all the parameters I tried to tune in this fashion,
it didn't actually work...
"""

import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from models.KinematicBicycleModel import KinematicBicycleModel
from models.DynamicBicycleModel import DynamicBicycleModel
from models.State import State


START_IDX = 50
PREDICTION_HORIZON = 20

drive = pd.read_csv('data/easy-drive-deltas.csv').iloc[START_IDX:START_IDX+PREDICTION_HORIZON]
steers = drive['cmd_steer'].to_numpy()
throttles = drive['cmd_throttle'].to_numpy()
steps = np.arange(START_IDX, START_IDX+PREDICTION_HORIZON)  # Steps for x-axis in subplots

# Set up simulation.
initial_state = State(
    x=drive.loc[START_IDX, 'X'],
    y=drive.loc[START_IDX, 'Y'],
    yaw=drive.loc[START_IDX, 'yaw'],
    v_x=drive.loc[START_IDX, 'vx'],
    v_y=drive.loc[START_IDX, 'vy'],
    yaw_dot=(drive.loc[START_IDX+1, 'yaw'] - drive.loc[START_IDX, 'yaw'])*0.0
)
model = DynamicBicycleModel(initial_state)

def _cost(sim_df, model_states):
    sim_points = np.array(list(zip(sim_df['X'], sim_df['Y'])))
    model_points = np.array(list(zip([s.x for s in model_states], 
                                     [s.y for s in model_states])))
    return np.linalg.norm(sim_points-model_points)

def cost(dt):
    dt = dt[0]
    model = DynamicBicycleModel(initial_state)
    # Cf, Cr = coeffs[0], coeffs[1]
    sim_states = [initial_state]
    for steer, throttle in zip(steers[:-1], throttles[:-1]):
        model.step(throttle, steer, dt=dt)
        sim_states.append(model.state)
    return _cost(drive, sim_states)

ret = optimize.dual_annealing(
    cost,
    # [(10**4, 10**6), (10**4, 10**6)],
    [(0.001, 0.2)],
    maxiter=10_000,
    # options={'maxtime': 90000},
    # workers=10
)
print(ret.x)
print(ret.fun)
print(ret.message)

sim_states = []
for steer, throttle in zip(steers, throttles):
    model.step(throttle, steer, dt=ret.x)
    sim_states.append(model.state)


# Plot trajectory
plt.plot([state.x for state in sim_states], 
               [state.y for state in sim_states], 
               'r:', label="Model")
plt.plot(drive['X'], drive['Y'], 'g:', label="Simulation")
plt.legend()
plt.title("Kinematic model vs simulation result")
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()
