import pandas as pd
import os
from learning.util import get_abs_fp, Writer

###
datasets = [
    {'dir': 'data/nodamp'},
    {'dir': 'data/nodamp-pid-79'},
    {'dir': 'data/sin-1'},
    {'dir': 'data/uniform-vy'}
]

FEATURES = [
    'X_0', 'Y_0', 'yaw_0', 'vx_0', 'vy_0', 'yawdot_0', 'throttle_0', 'steer_0', 'last_ts',
]
TARGETS = [
    'X_1', 'Y_1', 'yaw_1', 'vx_1', 'vy_1', 'yawdot_1',
]
###
def errors(df):
    step0 = df.loc[:, FEATURES[:6]].to_numpy()
    step1 = df.loc[:, TARGETS].to_numpy()
    err = step1 - step0
    err_df = pd.DataFrame(err, columns=['X_err', 'Y_err', 'yaw_err', 'vx_err', 'vy_err', 'yawdot_err'])
    return err_df
    

for d in datasets:
    d['abs_dir'] = get_abs_fp(__file__, d['dir'])
    d['train_df'] = pd.read_csv(os.path.join(d['abs_dir'], 'train.csv'))
    d['val_df'] = pd.read_csv(os.path.join(d['abs_dir'], 'validate.csv'))
    d['train_err_df'] = errors(d['train_df'])
    d['val_err_df'] = errors(d['val_df'])

    w = Writer(os.path.join(d['abs_dir'], 'description.txt'), delete_if_exists=True)

    w('train')
    w(d['train_df'].describe())
    w('\nval')
    w(d['val_df'].describe())

    w('train_err')
    w(d['train_err_df'].describe())
    w('\nval_err')
    w(d['val_err_df'].describe())
    