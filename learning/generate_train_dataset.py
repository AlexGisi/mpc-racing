"""
Generate dataset in appropriate format for train.py, possible combining multiple runs. 
Even if just using one logging run, must run this script to generate correct columns.
"""

import os
import pandas as pd

###
PARENT_DIR = '../runs/'
FILEPATHS = [
    '../runs/pid-79/steps.csv',
]

THROW_OUT_FIRST = 20  # Use data starting after...
FINAL_NAME = "../runs/combined.csv"  # If empty use first from FILEPATHS
###

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
fps = [os.path.join(parent_dir, fp) for fp in FILEPATHS]
dfs = [pd.read_csv(fp) for fp in fps]

if FINAL_NAME == "":
    final_name = os.path.basename(FILEPATHS[0])
else:
    final_name = FINAL_NAME

all_data = []
for df in dfs:
    for i, row in list(df.iterrows())[THROW_OUT_FIRST:-1]:
        row1 = df.loc[i+1, :]
        all_data.append({
            'X_0': row['X'],
            'Y_0': row['Y'],
            'yaw_0': row['yaw'],
            'vx_0': row['vx'],
            'vy_0': row['vy'],
            'yawdot_0': row['yawdot'],
            'throttle_0': row['cmd_throttle'] - row['cmd_brake'],
            'steer_0': row['cmd_steer'],
            'last_ts': row1['last_ts'],
            'X_1': row1['X'],
            'Y_1': row1['Y'],
            'yaw_1': row1['yaw'],
            'vx_1': row1['vx'],
            'vy_1': row1['vy'],
            'yawdot_1': row1['yawdot'],
        })

final_df = pd.DataFrame(all_data)
final_df.to_csv(os.path.join(parent_dir, final_name), index=False)
