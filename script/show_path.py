import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline

FP1 = "shanghai-handpicked.csv"
FP2 = "shanghai-ga.csv"

S = 500
LENGTH = 50

df1 = pd.read_csv(FP1)
df2 = pd.read_csv(FP2)

df1 = df1[ (df1['progress'] >= S) & (df1['progress'] < S) ]
df2 = df1[ (df1['progress'] >= S) & (df1['progress'] < S) ]
cl = ParameterizedCenterline()

plt.plot(df1['X'], df1['Y'], label="Normal")
plt.plot(df2['X'], df2['Y'], label="GA")
plt.plot([x for x,y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], color='b')
plt.plot([x for x,y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], color='b')
plt.title("Shanghai")
plt.legend()
plt.show()

plt.plot(range(len(df1)), df1['vx'], label='Normal')
plt.plot(range(len(df2)), df2['vx'], label='GA')
plt.grid(True)
plt.legend()
plt.title("Car Longitudinal Velocity over Shanghai")
plt.show()

plt.plot(range(len(df1)), df1['vy'], label='Normal')
plt.plot(range(len(df2)), df2['vy'], label='GA')
plt.grid(True)
plt.legend()
plt.title("Car Lateral Velocity over Shanghai")
plt.show()
