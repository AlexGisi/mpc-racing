import numpy as np
import matplotlib.pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline

LOOKAHEAD = 50

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

ss = np.linspace(0, cl.length, 50)

def plot(s, e):
    ss = np.arange(s, s+LOOKAHEAD, 0.5)
    es = np.ones_like(ss)*e

    norms = [cl.unit_principal_normal(s) for s in ss]
    error_vectors = [(x*e, y*e) for (x, y), e in zip(norms, es)]
    s_points = [(cl.Gx(s), cl.Gy(s)) for s in ss]
    e_points_right = [(x+ex, y+ey) for (x, y), (ex, ey) in zip(s_points, error_vectors)]
    e_points_left = [(x-ex, y-ey) for (x, y), (ex, ey) in zip(s_points, error_vectors)]

    plt.scatter([x for x, y in s_points], [y for x, y in s_points], label="s points")
    plt.scatter([x for x, y in e_points_right], [y for x, y in e_points_right], label="error right")
    plt.scatter([x for x, y in e_points_left], [y for x, y in e_points_left], label="error left")
    plt.plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r', label="left lane")
    plt.plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r', label="right lane")
    plt.grid(True)
    plt.xlim(min([x for x, y in s_points])-5, max([x for x, y in s_points])+5)
    plt.ylim(min([y for x, y in s_points])-5, max([y for x, y in s_points])+5)
    plt.legend()
    plt.show()

for s in ss:
    plot(s, cl.lookup_error(s, LOOKAHEAD))
