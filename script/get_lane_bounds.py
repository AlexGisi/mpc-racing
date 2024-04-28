"""
assume we are going around clockwise
"""

import pandas as pd
import matplotlib.pyplot as plt 
from splines.ParameterizedCenterline import ParameterizedCenterline

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

df = pd.read_csv("data/data-lanebounds.csv")
df = df

lefts_x = df['next_left_lane_point_x']
lefts_y = df['next_left_lane_point_y']
rights_x = df['next_right_lane_point_x']
rights_y = df['next_right_lane_point_y']

left_df_temp = df[['next_left_lane_point_x', 'next_left_lane_point_y']].copy()
right_df_temp = df[['next_right_lane_point_x', 'next_right_lane_point_y']].copy()

left_df = left_df_temp.rename(columns={
    'next_left_lane_point_x': 'x',
    'next_left_lane_point_y': 'y'
}).drop_duplicates()
right_df = right_df_temp.rename(columns={
    'next_right_lane_point_x': 'x',
    'next_right_lane_point_y': 'y'
}).drop_duplicates()

plt.plot(left_df['x'], left_df['y'], label='left lane')
plt.plot(right_df['x'], right_df['y'], label='right lane')
plt.plot([x for (x, y) in cl.waypoints], [y for (x, y) in cl.waypoints], 'g--')
plt.legend()
plt.show()

right_df.to_csv("lanes/right.csv", index=False)
left_df.to_csv("lanes/left.csv", index=False)
