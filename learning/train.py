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
from learning.util import get_most_recent_subdirectory


###----------

LR = 1e-3
BATCH_SIZE = 64
TEST_SPLIT = 0.15
EPOCHS = 100
OPTIMIZER = 'Adam'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337

LOG_DIR = ""  # If empty use datetime
DATA_CSV_FP = "../runs/combined.csv"  # Relative to this file

FEATURES = [
    'X_0', 'Y_0', 'yaw_0', 'vx_0', 'vy_0', 'yawdot_0', 'throttle_0', 'steer_0', 'last_ts',
]
TARGETS = [
    'X_1', 'Y_1', 'yaw_1', 'vx_1', 'vy_1', 'yawdot_1',
]

###----------


data_fp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_CSV_FP))

if LOG_DIR == "":
    LOG_DIR = "logs/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

log_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_DIR)
)

torch.manual_seed(SEED)

print(f"Using device {DEVICE}.")
print("Loading data...")
steps_df = pd.read_csv(data_fp)
data_X = steps_df.loc[:, FEATURES].to_numpy()
data_Y = steps_df.loc[:, TARGETS].to_numpy()

X = torch.tensor(data_X, dtype=torch.float32, device=DEVICE)
Y = torch.tensor(data_Y, dtype=torch.float32, device=DEVICE)
print(f"Loaded {len(steps_df)} rows.")

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=TEST_SPLIT, random_state=1337,
)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Vehicle()

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
else:
    raise ValueError(f"Optimizer {OPTIMIZER} is not available.")
criterion = nn.MSELoss()

print("Initial report\n---")
print("Front tire")
for n, p in model.front_tire.named_parameters():
    print(f"\t{n}: {p}")
print("Back tire")
for n, p in model.back_tire.named_parameters():
    print(f"\t{n}: {p}")

writer = SummaryWriter(log_fp)
start = time.monotonic()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

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
