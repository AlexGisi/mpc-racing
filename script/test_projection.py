"""
Test the centerline projection method for reliability over different values.

Conclusion: works well everywhere as long as you give it bounds.
"""
import numpy as np
from math import sqrt
from splines.ParameterizedCenterline import ParameterizedCenterline

PERTURB_S = 20.0

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")
print("finish load")

true_s = np.linspace(0.2, cl.length, 1000)
true_xy = np.array([(cl.Gx(s), cl.Gy(s)) for s in list(true_s)])
uprs = [cl.unit_principal_normal(s) for s in true_s]
noise = np.random.normal(scale=5.0, size=(len(true_s), 1))
perturbed_xy = [(tx + pnx, ty + pny) for ((tx, ty), (pnx, pny), n) in zip(true_xy, uprs, noise)]
perturbed_s_plus = np.clip(true_s + np.ones_like(true_s)*PERTURB_S, 0, cl.length)
perturbed_s_minus = np.clip(true_s - np.ones_like(true_s)*PERTURB_S, 0, cl.length)
perturbed_s = list(zip(perturbed_s_minus, perturbed_s_plus))

proj = []

print("start opt")
for s, sp, (X, Y) in zip(true_s, perturbed_s, perturbed_xy):
    proj.append(cl.projection(X, Y, bounds=(sp[0], sp[1])))

for p, (X, Y), pXY, i in zip(proj, true_xy, perturbed_xy, range(len(true_s))):
    dist = p[1]
    print("proj dist from cl", dist)

    point = [cl.Gx(p[0]), cl.Gy(p[0])]
    error = sqrt((point[0] - X)**2 + (point[1] - Y)**2)

    print("projection distance from true: ", error)

    if error > 1.0:
        cl.plot(points=[[X, Y], point, pXY], labels=["true", "global", "perturbedXY"])
