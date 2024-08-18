import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


DF_FP = os.path.join(sys.argv[1], 'steps.csv')
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0

df = pd.read_csv(DF_FP).iloc[START:, :]

fig, axs = plt.subplots(3, 3)

axs[0, 0].plot(df['steps'], df['X'])
axs[0, 0].set_title("X")
axs[0, 0].grid(True)

axs[0, 1].plot(df['steps'], df['Y'])
axs[0, 1].set_title("Y")
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

axs[2, 1].plot(df['steps'], df['cmd_throttle'], label="throttle")
axs[2, 1].plot(df['steps'], df['cmd_steer'], label="steer")
axs[2, 1].set_title("commands")
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
