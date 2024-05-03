import matplotlib.pyplot as plt
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from control.ControllerParameters import RuntimeControllerParameters
from control.MPC import MPC
from models.State import State

cl = ParameterizedCenterline(track="shanghai_intl_circuit")

plt.plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r')
plt.plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r')

print(np.min([x for x in cl.left_lane.waypoints]))
print(np.argmin([x for x in cl.left_lane.waypoints]))

breakpoint()

plt.show()