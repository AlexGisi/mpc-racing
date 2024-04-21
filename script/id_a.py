"""
Identify the vehicle acceleration *over the next timestep* as a function of the throttle.
"""
from models.VehicleParameters import VehicleParameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('throttle_sin.csv')
df['time'] = df['dt'].cumsum()

# Smoothing the velocity data using a simple moving average
window_size = 3  # The window size can be adjusted depending on the data
df['vx'] = df['vx'].rolling(window=window_size, center=True, min_periods=1).mean()
df['vy'] = df['vy'].rolling(window=window_size, center=True, min_periods=1).mean()

def calculate_acceleration(df):
    n = len(df)
    acceleration = np.zeros(n)
    
    # Calculate variable time steps for central difference
    df['dt_next'] = df['dt'].shift(-1).fillna(method='ffill')  # Shift dt upwards to use for forward time difference
    
    # Central difference for interior points
    for i in range(1, n-1):
        ax = (df.loc[i + 1, 'vx'] - df.loc[i - 1, 'vx']) / (df.loc[i + 1, 'time'] - df.loc[i - 1, 'time'])
        ay = (df.loc[i + 1, 'vy'] - df.loc[i - 1, 'vy']) / (df.loc[i + 1, 'time'] - df.loc[i - 1, 'time'])
        acceleration[i] = np.sqrt(np.square(ax) + np.square(ay))

    # Forward difference for the first data point
    if n > 1:  # Check to ensure there is more than one point
        ax = (df.loc[1, 'vx'] - df.loc[0, 'vx']) / (df.loc[1, 'time'] - df.loc[0, 'time'])
        ay = (df.loc[1, 'vy'] - df.loc[0, 'vy']) / (df.loc[1, 'time'] - df.loc[0, 'time'])
        acceleration[0] = np.sqrt(np.square(ax) + np.square(ay))
    
    # Backward difference for the last data point
    ax = (df.loc[n - 1, 'vx'] - df.loc[n - 2, 'vx']) / (df.loc[n - 1, 'time'] - df.loc[n - 2, 'time'])
    ay = (df.loc[n - 1, 'vy'] - df.loc[n - 2, 'vy']) / (df.loc[n - 1, 'time'] - df.loc[n - 2, 'time'])
    acceleration[-1] = np.sqrt(np.square(ax) + np.square(ay))
    
    return acceleration
