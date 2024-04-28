"""
Current script verifies kinematic model, need to verify dynamic.

Honestly we are using friction and things so this is not even a good
comparison, instead see the scripts where I compare with actual carla data
"""

from models.KinematicBicycleModel import KinematicBicycleModel        # See Vehicle Dynamics And Control (2005) by Rajamani, page 31.
        # Calculate tire velocity angle at front and rear.
        theta_Vf = np.arctan2((v_y + lf * yaw_dot), v_x)
        theta_Vr = np.arctan2((v_y - lr * yaw_dot), v_x)
from models.DynamicBicycleModel import DynamicBicycleModel
from models.State import State
import numpy as np
import matplotlib.pyplot as plt


def tester(x_0, y_0, xv_0, yv_0, yaw0, r0, cmd_throttle_arr, cmd_steering_arr, dt):
    vel = np.sqrt((xv_0 ** 2) + (yv_0**2))
    delta_time = dt
    states = []

    def update(x: float, y: float, yaw: float, velocity: float, acceleration: float, steering_angle: float):
        acceleration = 2.165674*acceleration
        
        # Compute the local velocity in the x-axis
        new_velocity = velocity + delta_time * acceleration

        steering_angle = np.deg2rad(steering_angle*45.0)

        # Compute the angular velocity
        angular_velocity = new_velocity*np.tan(steering_angle) / (0.832 + 0.708)

        # Compute the final state using the discrete time model
        new_x   = x + velocity*np.cos(yaw)*delta_time
        new_y   = y + velocity*np.sin(yaw)*delta_time
        new_yaw = yaw + angular_velocity*delta_time
        
        return new_x, new_y, new_yaw, new_velocity, steering_angle, angular_velocity
    
    states.append(State(x_0, y_0, yaw0, vel, 0, r0))
    for t, s in zip(cmd_throttle_arr, cmd_steering_arr):
        x_0, y_0, yaw0, vel, _, _ = update(x_0, y_0, yaw0, vel, t, s)
        new_state = State(x_0, y_0, yaw0, vel, 0, 0)
        states.append(new_state)
        #breakpoint()
    
    return states


def simulate(model, steps, dt, acceleration, steering_angle):
    assert len(acceleration) == steps
    states = []
    for i in range(steps):
        state = model.step(acceleration[i], steering_angle[i])
        states.append(state)
    return states

def plot_results(states_kinematic, states_tester, accel, steers, times):
    # Extracting positions for kinematic model
    x_kinematic = [state.x for state in states_kinematic]
    y_kinematic = [state.y for state in states_kinematic]
    yaw_kinematic = [state.yaw for state in states_kinematic]

    # Extracting positions for dynamic model
    x_tester = [state.x for state in states_tester]
    y_tester = [state.y for state in states_tester]

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(x_kinematic, y_kinematic, 'r-', label='Kinematic Model')

    plt.plot(x_tester, y_tester, 'b--', label='Test Model')
    plt.title('Vehicle Trajectories')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')

    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    velocities_kinematic = [np.sqrt(state.v_x**2 + state.v_y**2) for state in states_kinematic]
    velocities_tester = [np.sqrt(state.v_x**2 + state.v_y**2) for state in states_tester]
    breakpoint()

    plt.plot(range(len(velocities_kinematic)), velocities_kinematic, 'r-', label='Kinematic Speed')
    plt.plot(range(len(velocities_tester)), velocities_tester, 'b', label='Tester Speed')
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
initial_state = State(x=0, y=0, yaw=0.0, v_x=10, v_y=0, yaw_dot=0.0)
kinematic_model = KinematicBicycleModel(initial_state)
# dynamic_model = DynamicBicycleModel(initial_state)

# Simulation settings
steps = 1000
dt = 0.1  # time step duration
times = np.arange(0, steps*dt, step=dt)
acceleration_cmd = 0 * times
steering_cmd = (np.random.rand(*times.shape) - 0.5) / 10

# Running simulations
states_kinematic = simulate(kinematic_model, steps, dt, acceleration_cmd, steering_cmd)
# states_dynamic = simulate(dynamic_model, steps, dt, acceleration_cmd, steering_cmd)
states_test = tester(initial_state.x, initial_state.y+5, initial_state.v_x,
                     initial_state.v_y, initial_state.yaw, initial_state.yaw_dot, acceleration_cmd,
                     steering_cmd, dt)

# Plotting results
plot_results(states_kinematic, 
             states_test,
             acceleration_cmd,
             steering_cmd,
             times)

