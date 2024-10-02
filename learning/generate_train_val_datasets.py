"""
Generate dataset in appropriate format for train.py, possible combining multiple runs. 
Even if just using one logging run, must run this script to generate correct columns.

Puts a training
"""
import os
import pandas as pd
import numpy as np

import learning.dataset_filters as filters


###
PARENT_DIR = '../runs/'
FILEPATHS = [
    'sin-1/steps.csv',
    'no-damp/steps.csv',
    'pid-79/steps.csv',
    'sin-long/steps.csv',
    'sin-longer/steps.csv'
]

THROW_OUT_FIRST = 20  # Use data starting after...
TRAIN_SPLIT = 0.85

# Apply some filter on the combined-dataset dataframe.
# POST_FILTER = lambda df: filters.uniform('vy_1', 3, df)
POST_FILTER = None
OUT_DIR = "data/"
###

out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_DIR))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), PARENT_DIR))
fps = [os.path.join(parent_dir, fp) for fp in FILEPATHS]
dfs = [pd.read_csv(fp) for fp in fps]


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

all_df = pd.DataFrame(all_data)
if POST_FILTER:
    all_df = POST_FILTER(all_df)

n_train_idx = np.ceil(TRAIN_SPLIT * len(all_df)).astype('int')

train_idx = np.random.choice(all_df.index, size=(n_train_idx,), replace=False)
val_idx = [i for i in range(len(all_df)) if i not in train_idx]

train_df = all_df.loc[train_idx, :].reindex()
val_df = all_df.loc[val_idx, :].reindex()

print(f"created train df with {len(train_df)} rows")
print(f"created val df with {len(val_df)} rows")
assert(len(train_df) + len(val_df) == len(all_df))

train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(out_dir, 'validate.csv'), index=False)
