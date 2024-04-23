"""
Identify the force applied in the longitudinal direction *over the next timestep*
as a function of car's inputs.

In this analysis, I neglected that the force should be associated with the inputs
from the *previous* timestep, so that needs to be fixed. 
"""
from models.VehicleParameters import VehicleParameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv('throttle-sin.csv')
df['time'] = df['dt'].cumsum()

# Smoothing the velocity data using a simple moving average
window_size = 3  # The window size can be adjusted depending on the data
df['vx'] = df['vx'].rolling(window=window_size, center=True, min_periods=1).mean()

def calculate_acceleration(df):
    n = len(df)
    acceleration = np.zeros(n)
    
    # Calculate variable time steps for central difference
    df['dt_next'] = df['dt'].shift(-1).fillna(method='ffill')  # Shift dt upwards to use for forward time difference
    
    # Central difference for interior points
    for i in range(1, n-1):
        acceleration[i] = (df.loc[i + 1, 'vx'] - df.loc[i - 1, 'vx']) / (df.loc[i + 1, 'time'] - df.loc[i - 1, 'time'])
    
    # Forward difference for the first data point
    if n > 1:  # Check to ensure there is more than one point
        acceleration[0] = (df.loc[1, 'vx'] - df.loc[0, 'vx']) / (df.loc[1, 'time'] - df.loc[0, 'time'])
    
    # Backward difference for the last data point
    acceleration[-1] = (df.loc[n - 1, 'vx'] - df.loc[n - 2, 'vx']) / (df.loc[n - 1, 'time'] - df.loc[n - 2, 'time'])
    
    return acceleration

# Applying the function and adding to DataFrame
df['acceleration'] = calculate_acceleration(df)
df['Fx'] = df['acceleration'] * VehicleParameters().m
df_full = df[df['steps'] > 100]
df = df[(df['steps'] > 200) & (df['cmd_throttle'] >= 0)]

print(df)

# Features matrix and target vector
X = df[['cmd_throttle', 'cmd_steer']]
y = df['Fx']

# Create a Linear Regression model and fit it
model = LinearRegression().fit(X, y)

# Display the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

x = np.linspace(df['cmd_throttle'].min(), df['cmd_throttle'].max(), 100)
y = np.linspace(df['cmd_steer'].min(), df['cmd_steer'].max(), 100)
xx, yy = np.meshgrid(x, y)

# Predict z values for the meshgrid
zz = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data points
ax.scatter(df['cmd_throttle'], df['cmd_steer'], df['Fx'], c='r', marker='o', s=1)

# Surface plot for the regression plane
ax.plot_surface(xx, yy, zz, color='b', alpha=0.5)  # semi-transparent blue plane

ax.set_xlabel('Command Throttle')
ax.set_ylabel('Command Steer')
ax.set_zlabel('Fx')

plt.show()

X = df[['cmd_throttle', 'cmd_steer']]
y = df['Fx']

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(r2)

residuals = y - y_pred
residuals_mean = residuals.mean()
residuals_std = residuals.std()
print(f"residuals mean: {residuals_mean}, residuals_std: {residuals_std}")

plt.scatter(y_pred, residuals,  s=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at zero
plt.show()

def Fx(throttle, steer):
    return (14470.41348872 * abs(throttle) 
            + -47.77210478 * steer 
            - 4600.14374037663)

df = df.assign(pred_Fx=lambda row: predict(row['cmd_throttle'], row['cmd_steer']))
r22 = r2_score(y, df['pred_Fx'])
print(r22)

# df['diff'] = df['pred_Fx'] - df['Fx']
# bigs = (df['diff'] > 0.1).sum()
# breakpoint()
# print(bigs)
# print(len(df))

df_full = df_full.assign(pred_Fx=lambda row: predict(row['cmd_throttle'], row['cmd_steer']))
x = np.linspace(df_full['cmd_throttle'].min(), df_full['cmd_throttle'].max(), 100)
y = np.linspace(df_full['cmd_steer'].min(), df_full['cmd_steer'].max(), 100)
xx, yy = np.meshgrid(x, y)

# Predict z values for the meshgrid
# zz = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
zz = np.array([predict(throt, st) for throt, st in zip(np.ravel(xx), np.ravel(yy))]).reshape(xx.shape)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data points
ax.scatter(df_full['cmd_throttle'], df_full['cmd_steer'], df_full['Fx'], c='r', marker='o', s=1)

# Surface plot for the regression plane
ax.plot_surface(xx, yy, zz, color='b', alpha=0.5)  # semi-transparent blue plane

ax.set_xlabel('Command Throttle')
ax.set_ylabel('Command Steer')
ax.set_zlabel('Fx')

plt.title("Fx under given throttle and steer commands")

plt.show()
