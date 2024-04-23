import numpy as np
import matplotlib.pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline

cl = ParameterizedCenterline()
cl.from_file("/home/alex/Projects/graic/autobots-race/waypoints/shanghai_intl_circuit")
ss = np.linspace(0, cl.length, 1000)
ks = [cl.curvature(s) for s in ss]

gx = [cl.Gx(s) for s in ss]
gy = [cl.Gy(s) for s in ss]

sss = np.linspace(0, cl.length, 10000)
plt.plot([cl.Gx(s) for s in sss], [cl.Gy(s) for s in sss])
plt.scatter(gx, gy, label="Points at which curvature computed")

# Annotate each point with its curvature
for x, y, k in zip(gx, gy, ks):
    plt.text(x, y, f"{k:.2f}", fontsize=9, ha='right', va='bottom')

# Set plot details
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Curvature Computation Along the Track')
plt.legend()
plt.grid(True)
plt.show()