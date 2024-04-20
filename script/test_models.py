from models.KinematicBicycleModel import KinematicBicycleModel
from models.DynamicBicycleModel import DynamicBicycleModel
from models.State import State
import numpy as np
import matplotlib.pyplot as plt


def simulate(model, steps, dt, acceleration, steering_angle):
    assert len(acceleration) == steps
    states = []
    for i in range(steps):
        state = model.step(acceleration[i], steering_angle[i])
        states.append(state)
    return states

def plot_results(states_kinematic, states_dynamic, accel, steers, times):
    # Extracting positions for kinematic model
    x_kinematic = [state.x for state in states_kinematic]
    y_kinematic = [state.y for state in states_kinematic]

    # Extracting positions for dynamic model
    x_dynamic = [state.x for state in states_dynamic]
    y_dynamic = [state.y for state in states_dynamic]

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(x_kinematic, y_kinematic, 'r-', label='Kinematic Model')
    plt.plot(x_dynamic, y_dynamic, 'b--', label='Dynamic Model')
    plt.title('Vehicle Trajectories')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    velocities_kinematic = [np.sqrt(state.v_x**2 + state.v_y**2) for state in states_kinematic]
    velocities_dynamic = [np.sqrt(state.v_x**2 + state.v_y**2) for state in states_dynamic]
    plt.plot(range(len(velocities_kinematic)), velocities_kinematic, 'r-', label='Kinematic Speed')
    plt.plot(range(len(velocities_dynamic)), velocities_dynamic, 'b--', label='Dynamic Speed')
    plt.title('Vehicle Speed Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(times, accel)
    plt.title("Acceleration commands")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(times, steers)
    plt.title("Steer commands")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Parameters
initial_state = State(x=0, y=0, yaw=0, v_x=0, v_y=0, yaw_dot=0)
kinematic_model = KinematicBicycleModel(initial_state)
dynamic_model = DynamicBicycleModel(initial_state)

# Simulation settings
steps = 100
dt = 0.1  # time step duration
times = np.arange(0, steps*dt, step=dt)
acceleration = np.exp(-times)
steering_angle = np.sin(times)

# Running simulations
states_kinematic = simulate(kinematic_model, steps, dt, acceleration, steering_angle)
states_dynamic = simulate(dynamic_model, steps, dt, acceleration, steering_angle)

# Plotting results
plot_results(states_kinematic, 
             states_dynamic,
             acceleration,
             steering_angle,
             times)

