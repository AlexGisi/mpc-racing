"""
Test the centerline projection method for reliability over different values.

Conclusion: works well everywhere as long as you give it bounds.
"""
import numpy as np
from math import sqrt
from splines.ParameterizedCenterline import ParameterizedCenterline
import time
PERTURB_S = 20.0

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")
print("finish load")

true_s = np.linspace(0.2, cl.length, 2000)
true_xy = np.array([(cl.Gx(s), cl.Gy(s)) for s in list(true_s)])
uprs = [cl.unit_principal_normal(s) for s in true_s]
noise = np.random.normal(scale=5.0, size=(len(true_s), 1))
perturbed_xy = [(tx + pnx, ty + pny) for ((tx, ty), (pnx, pny), n) in zip(true_xy, uprs, noise)]
perturbed_s_plus = np.clip(true_s + np.ones_like(true_s)*PERTURB_S, 0, cl.length)
perturbed_s_minus = np.clip(true_s - np.ones_like(true_s)*PERTURB_S, 0, cl.length)
perturbed_s = list(zip(perturbed_s_minus, perturbed_s_plus))

proj_local = []
proj_global = []
local_times = []
global_times = []

print("start opt")
for s, sp, (X, Y) in zip(true_s, perturbed_s, perturbed_xy):
    t0 = time.time()
    proj_local.append(cl.projection_local(X, Y, bounds=(0, cl.length)))
    t1 = time.time()
    local_times.append(float(t1-t0))
    
    t0 = time.time()
    proj_global.append(cl.projection_global(X, Y))
    t1 = time.time()
    global_times.append(float(t1-t0))

for p, pg, (X, Y), pXY, i in zip(proj_local, proj_global, true_xy, perturbed_xy, range(len(true_s))):
    dist = p[1]
    print("local proj dist from cl", dist)

    distg = pg[1]
    print("global proj dist from cl", dist)

    point = [cl.Gx(p[0]), cl.Gy(p[0])]
    local_error = sqrt((point[0] - X)**2 + (point[1] - Y)**2)

    point_global = [cl.Gx(pg[0]), cl.Gy(pg[0])]
    global_error = sqrt((point_global[0] - X)**2 + (point_global[1] - Y)**2)

    print("local projection distance from true: ", local_error)
    print("global projection distance from true: ", global_error)

    print("local_time: ", local_times[i])
    print("global_time: ", global_times[i])

    if global_error > 0.1:
        cl.plot(points=[[X, Y], point, pXY], labels=["true", "local", "car"])
