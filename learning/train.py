from datetime import datetime
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.vehicle import Vehicle
from learning.util import load_dataset


###----------

LR = 1e2
BATCH_SIZE = 64
TEST_SPLIT = 0.15
EPOCHS = 25
FACTOR = 0.1
PATIENCE = 3

OPTIMIZER = 'Adam'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
DTYPE = torch.float64

LOG_DIR = ""  # If empty use logs/(datetime)
DATA_TRAIN_FP = "data/nodamp/train.csv"  # Relative to this file
DATA_VAL_FP = "data/nodamp/validate.csv"

FEATURES = [
    'X_0', 'Y_0', 'yaw_0', 'vx_0', 'vy_0', 'yawdot_0', 'throttle_0', 'steer_0', 'last_ts',
]
TARGETS = [
    'X_1', 'Y_1', 'yaw_1', 'vx_1', 'vy_1', 'yawdot_1',
]

###----------


train_fp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_TRAIN_FP))
val_fp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_VAL_FP))

if LOG_DIR == "":
    LOG_DIR = "logs/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

log_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_DIR)
)

torch.manual_seed(SEED)

print(f"Using device {DEVICE}.")
print("Loading data...")
X_train, y_train = load_dataset(train_fp, FEATURES, TARGETS, DEVICE, DTYPE)
X_val, y_val = load_dataset(val_fp, FEATURES, TARGETS, DEVICE, DTYPE)
print(f"Loaded {len(X_train)} rows of training data and {len(X_val)} rows of validation data.")
breakpoint()

if torch.isnan(X_train).any() or torch.isnan(y_train).any() or torch.isnan(X_val).any() or torch.isnan(y_val).any():
    print("data has a nan")
    breakpoint()

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Vehicle('pacejka', dtype=DTYPE).to(DEVICE)

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

initial_str = f"ReduceLROnPlateau, factor=f{FACTOR}, patience={PATIENCE}\n"
initial_str += "Initial report\n---"
initial_str += "Front tire"
for n, p in model.front_tire.named_parameters():
    initial_str += f"\t{n}: {p}"
initial_str += "Back tire"
for n, p in model.back_tire.named_parameters():
    initial_str += f"\t{n}: {p}"

writer = SummaryWriter(log_fp)
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

        if torch.isnan(out).any() or torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(model.front_tire.a).any() or torch.isnan(model.back_tire.a).any():
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

    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch)
    for name, param in model.front_tire.named_parameters():
        writer.add_histogram("front_tire/param/" + name, param.cpu(), epoch)
        writer.add_histogram("front_tire/grad/" + name, param.grad.cpu(), epoch)
    for name, param in model.back_tire.named_parameters():
        writer.add_histogram("back_tire/param/" + name, param.cpu(), epoch)
        writer.add_histogram("back_tire/grad/" + name, param.grad.cpu(), epoch)

    print(
        f"Epoch {epoch+1}/{EPOCHS}\t Train Loss: {avg_train_loss:.6f}\t Validation Loss: {avg_val_loss:.6f}"
    )
end = time.monotonic()

print(initial_str)
print(f"Training finished in {end-start} seconds")
print("Final report\n---")
print("Front tire")
for n, p in model.front_tire.named_parameters():
    print(f"\t{n}: {p}")
print("Back tire")
for n, p in model.back_tire.named_parameters():
    print(f"\t{n}: {p}")

model_fp = os.path.join(log_fp, "model")
torch.save(model.state_dict(), model_fp)
