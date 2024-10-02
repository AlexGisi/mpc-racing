import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from learning.vehicle import Vehicle
from learning.util import load_dataset, get_abs_fp

###
models = [
    {"model": Vehicle("pacejka"), "name": "pacejka-1", "fp": "models/pacejka-1/model"},
    {"model": Vehicle("linear"), "name": "linear-1", "fp": "models/linear-1/model"},
]

DATA_VAL_FP = "data/uniform-vy/validate.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
SEED = 1337

FEATURES = [
    "X_0",
    "Y_0",
    "yaw_0",
    "vx_0",
    "vy_0",
    "yawdot_0",
    "throttle_0",
    "steer_0",
    "last_ts",
]
TARGETS = [
    "X_1",
    "Y_1",
    "yaw_1",
    "vx_1",
    "vy_1",
    "yawdot_1",
]

LABELS = ['vx', 'vy', 'yawdot']
###

torch.manual_seed(SEED)

X, y = load_dataset(get_abs_fp(__file__, DATA_VAL_FP), FEATURES, TARGETS, DEVICE, DTYPE)
criterion = nn.MSELoss()

for m in models:
    m['model'] = m['model'].to(DEVICE, dtype=DTYPE)
    m['model'].eval()

    with torch.no_grad():
        m['out'] = m['model'](X)  # [(X, Y, yaw, vx, vy, yawdot) x len(X)]
        m['errors'] = y - m['out']
        
        m['loss_vx'] = criterion(m['out'][:, 3], y[:, 3]).cpu().item()
        m['loss_vy'] = criterion(m['out'][:, 4], y[:, 4]).cpu().item()
        m['loss_yawdot'] = criterion(m['out'][:, 5], y[:, 5]).cpu().item()

        m['df'] = pd.DataFrame(m['errors'][:, 3:].detach().cpu().numpy(), columns=LABELS).describe()
        m['df'].loc['loss'] = [m['loss_vx'], m['loss_vy'], m['loss_yawdot']]

fig, axs = plt.subplots(1, 3)
for i in range(len(LABELS)):
    ax = axs[i]
    errs = [m['errors'][:, i+3].detach().cpu().numpy() for m in models]
    ax.boxplot(errs, labels=[m['name'] for m in models])
    ax.set_xlabel(LABELS[i])

for m in models:
    print(m['name'])
    print(m['df'])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.suptitle("Errors for vx, vy, yawdot across models")
plt.show()
