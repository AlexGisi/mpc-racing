import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.KinematicBicycleModel import KinematicBicycleModel
from models.State import State
from splines.ParameterizedCenterline import ParameterizedCenterline

drive = pd.read_csv('data/easy-drive.csv')
steers = drive['cmd_steer'].to_numpy()
throttles = drive['cmd_throttle'].to_numpy()
# dts = drive['prev_step_size'].to_numpy()

# Set up simulation.
initial_state = State(
    x=drive.loc[0, 'X'],
    y=drive.loc[0, 'X'],
    yaw=drive.loc[0, 'yaw'],
    v_x=drive.loc[0, 'vx'],
    v_y=drive.loc[0, 'vy'],
    yaw_dot=0.0
)
model = KinematicBicycleModel(initial_state)

# Perform simulation.
sim_states = []
for steer, throttle in zip(steers, throttles):
    model.step(throttle, steer)
    sim_states.append(model.state)

# Plot centerline and lanes.
left_df = pd.read_csv('lanes/left.csv')
right_df = pd.read_csv('lanes/right.csv')
cl = ParameterizedCenterline()

plt.plot(left_df['x'], left_df['y'], 'b')
plt.plot(right_df['x'], right_df['y'], 'b')
# plt.plot([x for x, y in cl.waypoints], [y for x, y in cl.waypoints], 'r:')

# Plot actual and model paths.
plt.plot([state.x for state in sim_states], 
         [state.y for state in sim_states], 
         'r:', label="Model")
plt.plot(drive['X'], drive['Y'], 'g:', label="Simulation")
plt.legend()
plt.title("Kinematic model vs simulation result")
plt.show()
