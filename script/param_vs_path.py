import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline


cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")
df = pd.read_csv("data-mydrive.csv")

ss = np.linspace(0, cl.length, 5000)
param_x = [cl.Gx(s) for s in ss]
param_y = [cl.Gy(s) for s in ss]
plot_points = list(zip(param_x, param_y))

param_x = [p[0] for p in plot_points]
param_y = [p[1] for p in plot_points]

car_x = [x for x in list(df["X"])]
car_y = list(df["Y"])

plt.scatter(car_x[0], car_y[0])
plt.plot(car_x, car_y, color='green', linewidth=3)
plt.plot(param_x, param_y)
#plt.scatter([w[0] for w in cl.waypoints], [w[1] for w in cl.waypoints], color='red', s=4)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
