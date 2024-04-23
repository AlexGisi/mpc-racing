"""
max curvature: 0.27
"""

import numpy as np
import matplotlib.pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline

def moving_average_lookahead(curvatures, lookahead=10):
    n = len(curvatures)
    moving_averages = []
    
    for i in range(n):
        lookahead_range = curvatures[i:min(i + lookahead, n)]
        average = sum(lookahead_range) / len(lookahead_range)
        moving_averages.append(average)
    
    return moving_averages


cl = ParameterizedCenterline()
cl.from_file("/home/alex/Projects/graic/autobots-race/waypoints/shanghai_intl_circuit")
ss = np.linspace(0, cl.length, 1000)
ks = [cl.curvature(s) for s in ss]
avgs = moving_average_lookahead(ks)

gx = [cl.Gx(s) for s in ss]
gy = [cl.Gy(s) for s in ss]

# sss = np.linspace(0, cl.length, 10000)
# plt.plot([cl.Gx(s) for s in sss], [cl.Gy(s) for s in sss])
# plt.scatter(gx, gy, label="Points at which curvature computed")

# Annotate each point with its curvature
# for x, y, k in zip(gx, gy, ks):
#     plt.text(x, y, f"{k:.2f}", fontsize=9, ha='right', va='bottom')


# Find max curvature
kmax = max(ks)
i = np.argmax(ks)

print(kmax)

max_k_x = gx[i]
max_k_y = gy[i]

# plt.plot(max_k_x, max_k_y, 'rx', markersize=13)

# Set plot details
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Curvature Computation Along the Track')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.plot(ss, avgs)
plt.xlabel('S')
plt.ylabel('k')
plt.show()


