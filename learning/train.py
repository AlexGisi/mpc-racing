from datetime import datetime
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.vehicle import Vehicle
from learning.util import load_dataset, get_abs_fp, Writer, make_parameter_heatmap


###----------

LR = 1e-2
BATCH_SIZE = 64
EPOCHS = 100
FACTOR = 1/2
PATIENCE = 2

OPTIMIZER = 'AdamW'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
DTYPE = torch.float64

LOG_DIR = ""  # If empty use logs/(datetime)
DATA_TRAIN_FP = "data/big/train.csv"
DATA_VAL_FP = "data/big/validate.csv"

FEATURES = [
    'X_0', 'Y_0', 'yaw_0', 'vx_0', 'vy_0', 'yawdot_0', 'throttle_0', 'steer_0', 'last_ts',
]
TARGETS = [
    'X_1', 'Y_1', 'yaw_1', 'vx_1', 'vy_1', 'yawdot_1',
]

TIRES = 'mlp2'

###----------

torch.manual_seed(SEED)

train_fp = get_abs_fp(__file__, DATA_TRAIN_FP)
val_fp = get_abs_fp(__file__, DATA_VAL_FP)

if LOG_DIR == "":
    LOG_DIR = "logs/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

log_fp = get_abs_fp(__file__, LOG_DIR)
os.makedirs(log_fp)

writer = Writer(os.path.join(log_fp, 'log.txt'))

writer("---hyperparameters---")
writer(f"LR={LR}, BATCH_SIZE={BATCH_SIZE}, FACTOR={FACTOR}, PATIENCE={PATIENCE}, OPTIMIZER={OPTIMIZER}")
writer(f"DATA_TRAIN_FP={DATA_TRAIN_FP}")
writer(f"DATA_VAL_FP={DATA_VAL_FP}")
writer(f"TIRES={TIRES}")

writer(f"Using device {DEVICE}.")
writer("Loading data...")
X_train, y_train = load_dataset(train_fp, FEATURES, TARGETS, DEVICE, DTYPE)
X_val, y_val = load_dataset(val_fp, FEATURES, TARGETS, DEVICE, DTYPE)
writer(f"Loaded {len(X_train)} rows of training data and {len(X_val)} rows of validation data.")

if torch.isnan(X_train).any() or torch.isnan(y_train).any() or torch.isnan(X_val).any() or torch.isnan(y_val).any():
    writer("data has a nan")
    exit()

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Vehicle(TIRES).to(DEVICE, dtype=DTYPE)

if OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif OPTIMIZER == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
elif OPTIMIZER == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=LR)
elif OPTIMIZER == "LBFGS":
    optimizer = torch.optim.LBFGS(model.parameters(), lr=LR, max_iter=20)
else:
    raise ValueError(f"Optimizer {OPTIMIZER} is not available.")

criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR, patience=PATIENCE)

initial_str = ""
initial_str += "Initial report\n---"
initial_str += 'car'
for n, p in model.named_parameters():
    initial_str += f"\t{n}: {p}"

# Initial loss.
model.eval()
val_loss = 0.0
with torch.no_grad():
    for x,y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out.squeeze(), y)

        val_loss += loss.item()
avg_val_loss = val_loss / len(val_loader)
writer(f"Initial loss: {avg_val_loss}")
# 

tb_writer = SummaryWriter(log_fp)
start = time.monotonic()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    i = 0
    x0 = None
    y0 = None
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()

        if torch.isnan(out).any() or torch.isnan(x).any() or torch.isnan(y).any():
            writer("encountered nan in training")
            breakpoint()

        running_loss += loss.item()
        i+= 1
        x0 = x
        y0 = y
    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x,y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out.squeeze(), y)

            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)
    writer(f"\tlr={round(scheduler.get_last_lr()[0], 10)}")

    tb_writer.add_scalar("Loss/train", avg_train_loss, epoch)
    tb_writer.add_scalar("Loss/validation", avg_val_loss, epoch)
    for name, param in model.named_parameters():
        tb_writer.add_histogram("param/" + name, param.cpu(), epoch)
        tb_writer.add_histogram("grad/" + name, param.grad.cpu(), epoch)

        if len(param.shape) == 2:
            fig = make_parameter_heatmap(param)
            tb_writer.add_figure(f"weights/{name}", fig, epoch)

    writer(
        f"Epoch {epoch+1}/{EPOCHS}\t Train Loss: {avg_train_loss:.7f}\t Validation Loss: {avg_val_loss:.7f}"
    )
end = time.monotonic()

if TIRES != 'mlp' and TIRES != 'mlp2':
    writer(initial_str)
    writer("Final report\n---")
    writer("car")
    for n, p in model.named_parameters():
        writer(f"\t{n}: {p}")

writer(f"Training finished in {end-start} seconds")

model_fp = os.path.join(log_fp, "model")
torch.save(model.state_dict(), model_fp)
