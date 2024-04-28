import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.KinematicBicycleModel import KinematicBicycleModel
from models.DynamicBicycleModel import DynamicBicycleModel
from models.State import State

START_IDX = 1400
PREDICTION_HORIZON = 100
PRED_EVERY = 1

drive = pd.read_csv('data/easy-drive.csv').iloc[START_IDX:START_IDX+PREDICTION_HORIZON+1].reset_index(drop=True)
DELTA_T = np.mean(drive['dt'])
print(f"DELTA_T: {DELTA_T}")
print(f"Lookahead time: {DELTA_T*PREDICTION_HORIZON}")

steers = drive['cmd_steer'].to_numpy()
throttles = drive['cmd_throttle'].to_numpy() - drive['cmd_brake'].to_numpy()

steps = np.arange(START_IDX, START_IDX+PREDICTION_HORIZON+1)  # Steps for x-axis in subplots
sim_steps = steps[::PRED_EVERY]

# Set up simulation.
initial_state = State(
    x=drive.loc[0, 'X'],
    y=drive.loc[0, 'Y'],
    yaw=drive.loc[0, 'yaw'],
    v_x=drive.loc[0, 'vx'],
    v_y=drive.loc[0, 'vy'],
    yaw_dot=(drive.loc[1, 'yaw'] - drive.loc[0, 'yaw'])*DELTA_T
)
model = DynamicBicycleModel(initial_state)

# Perform simulation.
sim_states = [initial_state]
infos = [{'Fx': float(drive.loc[0, 'ax']) / model.params.m, 
          'Fyf': 0, 'Fyr': 0, 'delta': 0,
          'Fx_info': {'wheel': 0, 'drag': 0, 'rolling_resistance': 0}}]
for steer, throttle in zip(steers[:-1:PRED_EVERY], throttles[:-1:PRED_EVERY]):
    state, info = model.step(throttle, steer, dt=DELTA_T*PRED_EVERY, Cf=200_000, Cr=200_000)
    sim_states.append(state)
    infos.append(info)

def cost(sim_df, model_states):
    sim_points = np.array(list(zip(sim_df.iloc[::PRED_EVERY, :]['X'], sim_df.iloc[::PRED_EVERY, :]['Y'])))
    model_points = np.array(list(zip([s.x for s in model_states], [s.y for s in model_states])))
    return np.linalg.norm(sim_points-model_points)

costs = []
# for i in range(1, PREDICTION_HORIZON+1):
#     drive_df = drive.iloc[:i, :]
#     states = sim_states[:i]
#     costs.append(cost(drive_df, states))

left_df = pd.read_csv('lanes/left.csv')
right_df = pd.read_csv('lanes/right.csv')

fig, axs = plt.subplots(3, 5, figsize=(14, 10))

# Plot trajectory
axs[0, 0].plot([state.x for state in sim_states], 
               [state.y for state in sim_states], 
               'r:', label="Model")
axs[0, 0].plot(drive['X'], drive['Y'], 'g:', label="Simulation")
axs[0, 0].legend(prop={'size': 6})
axs[0, 0].set_title(f"{model.__repr__()} vs simulation result")
axs[0, 0].set_xlabel('X Position')
axs[0, 0].set_ylabel('Y Position')
# axs[0, 0].set_aspect('equal')

# Empty plot on the right of the trajectory for layout balance
axs[0, 1].plot(sim_steps, [state.yaw for state in sim_states], 'r-', label='Model v_x')
axs[0, 1].plot(steps, drive['yaw'], 'g--', label='Simulation v_x')
axs[0, 1].set_title('Comparison of Yaws')
axs[0, 1].set_xlabel('Step')
axs[0, 1].set_ylabel('Yaw (rad)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Throttle commands
axs[1, 0].plot(steps, throttles, 'm-', label='Throttle')
axs[1, 0].set_title('Throttle Commands Over Time')
axs[1, 0].set_xlabel('Step')
axs[1, 0].set_ylabel('Throttle')
axs[1, 0].grid(True)

# Steering commands
axs[1, 1].plot(steps, steers, 'c-', label='Steering')
axs[1, 1].set_title('Steering Commands Over Time')
axs[1, 1].set_xlabel('Step')
axs[1, 1].set_ylabel('Steering')
axs[1, 1].grid(True)

# X velocity comparison
axs[2, 0].plot(sim_steps, [state.v_x for state in sim_states], 'r-', label='Model v_x')
axs[2, 0].plot(steps, drive['vx'], 'g--', label='Simulation v_x')
axs[2, 0].set_title('Comparison of X Velocities')
axs[2, 0].set_xlabel('Step')
axs[2, 0].set_ylabel('X Velocity (m/s)')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Y velocity comparison
axs[2, 1].plot(sim_steps, [state.v_y for state in sim_states], 'r-', label='Model v_y')
axs[2, 1].plot(steps, drive['vy'], 'g--', label='Simulation v_y')
axs[2, 1].set_title('Comparison of Y Velocities')
axs[2, 1].set_xlabel('Step')
axs[2, 1].set_ylabel('Y Velocity (m/s)')
axs[2, 1].legend()
axs[2, 1].grid(True)

# axs[0, 2].plot(sim_steps, costs)
# axs[0, 2].set_title("Cost over time")
# axs[0, 2].set_xlabel('Step')
# axs[0, 2].set_xlabel('Cost')
# axs[0, 2].grid(True)

axs[0, 2].plot(sim_steps[1:], [i['Fx'] / model.params.m for i in infos[1:]], 'r-', label="Model")
axs[0, 2].plot(steps, drive['ax'], 'g--', label="Simulation")
axs[0, 2].set_title("Longitudinal acceleration over time")
axs[0, 2].set_xlabel("Step")
axs[0, 2].legend()
axs[0, 2].grid(True)

axs[1, 2].plot(sim_steps[1:], [i['Fyf'] for i in infos[1:]], 'r-', label="Model")
axs[1, 2].set_title("Fyf over time")
axs[1, 2].set_xlabel("Step")
axs[1, 2].legend()
axs[1, 2].grid(True)

axs[2, 2].plot(sim_steps[1:], [i['Fyr'] for i in infos[1:]], 'r-', label="Model")
axs[2, 2].set_title("Fyr over time")
axs[2, 2].set_xlabel("Step")
axs[2, 2].legend()
axs[2, 2].grid(True)

axs[0, 3].plot(sim_steps[1:], [i['delta'] for i in infos[1:]], 'r-', label='Model')
axs[0, 3].plot(steps, drive['front_left_delta'], 'g-', label="Simulation FL")
axs[0, 3].plot(steps, drive['front_right_delta'], 'b:', label="Simulation FR")
axs[0, 3].set_title("Steering angles")
axs[0, 3].set_xlabel("Step")
axs[0, 3].set_ylabel("Angle (deg)")
axs[0, 3].grid(True)
axs[0, 3].legend(prop={'size': 6})

axs[1, 3].plot(sim_steps[1:], [i['Fx_info']['eta'] for i in infos[1:]])
axs[1, 3].set_title("Motor efficiency over time")
axs[1, 3].set_xlabel("Step")
axs[1, 3].set_ylabel("Efficiency")
axs[1, 3].grid(True)

axs[2, 3].plot(sim_steps[1:], [i['Fx_info']['rpm'] for i in infos[1:]])
axs[2, 3].set_title("Estimated motor rpm over time")
axs[2, 3].set_xlabel("Step")
axs[2, 3].set_ylabel("RPM")
axs[2, 3].grid(True)

axs[0, 4].plot(sim_steps[1:], [i['Fx_info']['wheel'] / model.params.m for i in infos[1:]])
axs[0, 4].set_title("Longitudinal acceleration due to wheel force")
axs[0, 4].grid(True)

axs[1, 4].plot(sim_steps[1:], [-i['Fx_info']['drag'] / model.params.m for i in infos[1:]])
axs[1, 4].set_title("Longitudinal acceleration due to drag force")
axs[1, 4].grid(True)

axs[2, 4].plot(sim_steps[1:], [-i['Fx_info']['rolling_resistance'] / model.params.m for i in infos[1:]])
axs[2, 4].set_title("Longitudinal acceleration due to rolling resistance")
axs[2, 4].grid(True)

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.4)
plt.tight_layout()
plt.show()
