import numpy as np
from matplotlib import pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline


cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

X, Y = 70, -115
s_hat, dist = cl.projection_local(X, Y)
x, y = cl.Gx(s_hat), cl.Gy(s_hat)

track_plot_points = list(np.linspace(s_hat-10, s_hat+10, 100))
track_x = [cl.Gx(s) for s in track_plot_points]
track_y = [cl.Gy(s) for s in track_plot_points]

plt.plot(track_x, track_y)
plt.plot(x, y, 'rx')
plt.plot(X, Y, 'bo')
plt.title(f"distance: {dist}")

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
