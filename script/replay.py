import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline

START_STEP = 0

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")
df = pd.read_csv("data-mydrive.csv")

def init_plot(ax):
    ax.set_aspect('equal', adjustable='box')

plt.ion()
fig, ax = plt.subplots()
init_plot(ax)

# Initialize line objects
track_plot_points = np.linspace(0, cl.length, 10000)
track_x = [cl.Gx(s) for s in track_plot_points]
track_y = [cl.Gy(s) for s in track_plot_points]
car_point, = ax.plot([], [], 'ro', label='car')
proj_point, = ax.plot([], [], 'bo', label='projection') 
plt.plot(track_x, track_y)
direction_arrow = None
steer_arrow = None

first_row = df.iloc[START_STEP, :]
ax.set_xlim([first_row['X']-10, first_row['X']+10])
ax.set_ylim([first_row['Y']-10, first_row['Y']+10])

def update_plot(row, ax, track_line, car_point):
    global direction_arrow, steer_arrow
    
    proj_x, proj_y = cl.Gx(row['progress']), cl.Gy(row['progress'])

    car_point.set_data([row['X']], [row['Y']])
    proj_point.set_data([proj_x, proj_y]) 

    arrow_length = 2 
    dx = arrow_length * np.cos(row['yaw'])
    dy = arrow_length * np.sin(row['yaw'])

    if direction_arrow is not None:
        direction_arrow.remove()
        steer_arrow.remove()

    steer_vec = np.array([dy, -dx]) * np.sign(row['cmd_steer'])*-1
    direction_arrow = ax.arrow(row['X'], row['Y'], dx, dy, head_width=0.5, 
                               head_length=0.7, fc='green', ec='green')
    steer_arrow = ax.arrow(row['X'], row['Y'], steer_vec[0], steer_vec[1], head_width=0.5,
                           head_length=5*row['cmd_steer'])

    ax.set_title(f"step = {row['steps']}\nerror = -, rec_error={row['error']}\ncmd_steer: {row['cmd_steer']}")
    # ax.set_xlim([row['X']-20, row['X']+20])
    # ax.set_ylim([row['Y']-20, row['Y']+20])
    ax.autoscale_view(True,True,True)  # Autoscale
    fig.canvas.draw()
    fig.canvas.flush_events()

df = df.assign(yaw=lambda r: np.deg2rad(r['yaw']))
for i, r in df.iloc[START_STEP:].iterrows():
    update_plot(r, ax, None, car_point)
    plt.pause(0.0000001)

plt.ioff()
plt.show(block=False)
