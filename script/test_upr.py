"""
Testing the unit principal normal vector function.

Conclusion: using the definition is ill-conditioned because of numerics
with the second derivatives (it seems). Just get the orthogonal vector
to the tangent.
"""

from matplotlib import pyplot as plt
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

S = 460
sx, sy = cl.Gx(S), cl.Gy(S)

upr = cl.unit_principal_normal(S)
upr_dx, upr_dy = upr[0]*10, upr[1]*10

upr1 = cl.unit_principal_normal1(S)
upr1_dx, upr1_dy = upr1[0]*5, upr1[1]*5

tn = cl.unit_tangent(S)
tn_dx, tn_dy = tn[0]*10, tn[1]*10

track_plot_points = list(np.linspace(S-30, S+30, 10000))
track_x = [cl.Gx(s) for s in track_plot_points]
track_y = [cl.Gy(s) for s in track_plot_points]

plt.plot(track_x, track_y)

plt.arrow(sx, sy, upr_dx, upr_dy, width=0.05, head_length=2)

# Note how this one is sometimes good, sometimes bad, sometimes horrific.
plt.arrow(sx, sy, upr1_dx, upr1_dy, width=0.05, head_length=2, color="red")

plt.arrow(sx, sy, tn_dx, tn_dy, width=0.05, head_length=2)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
