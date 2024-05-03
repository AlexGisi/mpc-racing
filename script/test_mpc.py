import matplotlib.pyplot as plt
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from control.ControllerParameters import RuntimeControllerParameters, FixedControllerParameters
from control.MPC import MPC
from models.State import State

cl = ParameterizedCenterline(track="shanghai_intl_circuit")
s = np.random.uniform(low=0, high=cl.length)
x, y = cl.Gx(s), cl.Gy(s)
yaw = cl.unit_tangent_yaw(s)
v_x, v_y, r = 8, 0, 0

cl_x_coeffs = cl.x_as_coeffs(s, FixedControllerParameters.lookahead_distance)
cl_y_coeffs = cl.y_as_coeffs(s, FixedControllerParameters.lookahead_distance)
max_err = cl.lookup_error(s, FixedControllerParameters.lookahead_distance)

state0 = State(x=x, y=y, yaw=yaw, v_x=v_x, v_y=v_y, yaw_dot=r)
states = [state0]

mpc = MPC(state=state0,
          s=s,
          centerline_x_poly_coeffs=cl_x_coeffs,
          centerline_y_poly_coeffs=cl_y_coeffs,
          max_error=max_err,
          runtime_params=RuntimeControllerParameters())
sol = mpc.solution()

breakpoint()

def state_arrow(state, length=3):
    dx = length * np.cos(state.yaw)
    dy = length * np.sin(state.yaw)
    return state.x, state.y, dx, dy

fig, ax = plt.subplots(2, 2, figsize=(10, 12))
ax[0, 0].plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r')
ax[0, 0].plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r')
ax[0, 0].arrow(*state_arrow(state0), width=0.2)
ax[0, 0].set_xlim(np.min([s.x for s in states])-25, np.max([s.x for s in states])+25)
ax[0, 0].set_ylim(np.min([s.y for s in states])-25, np.max([s.y for s in states])+25)

plt.show()
