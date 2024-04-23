"""
assume we are going around clockwise
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from splines.ParameterizedCenterline import ParameterizedCenterline

cl = ParameterizedCenterline()
cl.from_file("/home/alex/Projects/graic/autobots-race/waypoints/shanghai_intl_circuit")

df = pd.read_csv("/home/alex/Projects/graic/autobots-race/data-lanebounds.csv")
df = df.drop_duplicates()

lefts_x = df['next_left_lane_point_x']
lefts_y = df['next_left_lane_point_y']
rights_x = df['next_right_lane_point_x']
rights_y = df['next_right_lane_point_y']

left_df = df[['next_left_lane_point_x', 'next_left_lane_point_y']].copy()
right_df = df[['next_right_lane_point_x', 'next_right_lane_point_y']].copy()

ss = np.linspace(0, cl.length, 600)
xs = [cl.Gx(s) for s in ss]
ys = [cl.Gy(s) for s in ss]

temp = right_df.copy()

right_df = left_df.rename(columns={
    'next_left_lane_point_x': 'x',
    'next_left_lane_point_y': 'y'
})
left_df = temp.rename(columns={
    'next_right_lane_point_x': 'x',
    'next_right_lane_point_y': 'y'
})

plt.ion()
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

car_point, = ax.plot([], [], 'ro', label='car')

def update_plot(ax, x, y):
    car_point.set_data([x, y])
    ax.set_xlim([x-100, x+100])
    ax.set_ylim([y-100, y+100])
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.plot(left_df['x'], left_df['y'], 'r')
plt.plot(right_df['x'], right_df['y'], 'r')
plt.plot([x for (x, y) in cl.waypoints], [y for (x, y) in cl.waypoints])
plt.legend()

for x,y in zip(xs, ys):
    update_plot(ax, x, y)
    # plt.pause(0.01)
plt.ioff()
plt.show()

# right_df.to_csv("lanes/right.csv")
# left_df.to_csv("lanes/left.csv")
