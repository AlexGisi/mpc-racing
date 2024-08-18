import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline


DF_FP = os.path.join(sys.argv[1], 'steps.csv')
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0

df = pd.read_csv(DF_FP).iloc[START:, :]
cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

fig, axs = plt.subplots(3, 3)

axs[0, 0].plot(df['X'], df['Y'], 'r-')
axs[0, 0].plot([cl.Gx(s) for s in np.linspace(0, cl.length, 10000)], [cl.Gy(s) for s in np.linspace(0, cl.length, 10000)], 'b--')
axs[0, 0].plot([x for x,y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], color='b')
axs[0, 0].plot([x for x,y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], color='b')
for i, (x, y, step) in enumerate(zip(df['X'], df['Y'], df['steps'])):
    axs[0, 0].text(x, y, str(step), fontsize=5, ha='center')
axs[0, 0].set_title("path")
axs[0, 0].set_aspect('equal')
axs[0, 0].grid(True)

axs[0, 1].plot(df['steps'], df['mpc_time'])
axs[0, 1].set_title(f"mpc time (mean {df['mpc_time'].mean():.4})")
axs[0, 1].grid(True)

axs[0, 2].plot(df['steps'], df['progress'])
axs[0, 2].set_title("s")
axs[0, 2].grid(True)

axs[1, 0].plot(df['steps'], df['yaw'])
axs[1, 0].set_title('yaw')
axs[1, 0].grid(True)

axs[1, 1].plot(df['steps'], df['vx'])
axs[1, 1].set_title('vx')
axs[1, 1].grid(True)

axs[1, 2].plot(df['steps'], df['vy'])
axs[1, 2].set_title('vy')
axs[1, 2].grid(True)

axs[2, 0].plot(df['steps'], df['yawdot'])
axs[2, 0].set_title('yawdot')
axs[2, 0].grid(True)

axs[2, 1].plot(df['steps'], df['cmd_throttle']-df['cmd_brake'], 'r', label="throttle")
axs[2, 1].plot(df['steps'], df['cmd_steer'], 'b', label="steer")
axs[2, 1].set_title("commands")
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
