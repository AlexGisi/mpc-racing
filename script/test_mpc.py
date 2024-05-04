import matplotlib.pyplot as plt
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from control.ControllerParameters import RuntimeControllerParameters, FixedControllerParameters
from control.MPC import MPC
from control.util import make_poly
from models.State import State

cl = ParameterizedCenterline(track="shanghai_intl_circuit")
# s = np.random.uniform(low=0, high=cl.length)
s0 = 0
ss = np.arange(s0, cl.length, 0.5)
sol = None

x, y = cl.Gx(s0), cl.Gy(s0)
yaw = cl.unit_tangent_yaw(s0)
v_x, v_y, r = 1, 0, 0
state0 = State(x=x, y=y, yaw=yaw, v_x=v_x, v_y=v_y, yaw_dot=r)
for s in ss:
    cl_x_coeffs = cl.x_as_coeffs(s, FixedControllerParameters.lookahead_distance)
    cl_y_coeffs = cl.y_as_coeffs(s, FixedControllerParameters.lookahead_distance)
    max_err = cl.lookup_error(s, FixedControllerParameters.lookahead_distance)

    ### check centerline
    # cl_x = lambda s: make_poly(s, cl_x_coeffs)
    # cl_y = lambda s: make_poly(s, cl_y_coeffs)
    # ss = np.linspace(s, s+FixedControllerParameters.lookahead_distance, 50)
    # plt.plot([cl_x(s) for s in ss], [cl_y(s) for s in ss])
    # plt.show()
    ###
    
    states = [state0]
    mpc = MPC(state=state0,
              sol0=None,
              s0=s,
              centerline_x_poly_coeffs=cl_x_coeffs,
              centerline_y_poly_coeffs=cl_y_coeffs,
              max_error=max_err,
              runtime_params=RuntimeControllerParameters())

    sol, ret = mpc.solution()
    States, U, S_hat, e_hat_c, e_hat_l = ret[0], ret[1], ret[2], ret[3], ret[4]
    states.extend([State(x=t[0], y=t[1], yaw=t[2], v_x=t[3], v_y=t[4], yaw_dot=t[5]) for t in zip(States[0, :], States[1, :], States[2, :], States[3, :], States[4, :], States[5, :])])

    def state_arrow(state, length=3):
        dx = length * np.cos(state.yaw)
        dy = length * np.sin(state.yaw)
        return state.x, state.y, dx, dy

    fig, axs = plt.subplots(2, 3, figsize=(10, 12))

    cl_x = lambda s: make_poly(s, cl_x_coeffs)
    cl_y = lambda s: make_poly(s, cl_y_coeffs)
    sss = np.linspace(s, s+FixedControllerParameters.lookahead_distance, 50)
    axs[0, 0].plot([cl_x(s) for s in sss], [cl_y(s) for s in sss])

    axs[0, 0].plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r')
    axs[0, 0].plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r')
    axs[0, 0].plot(state0.x, state0.y, marker='o', color='r', ms=10, label="initial state")
    # axs[0, 0].scatter(States[0, :], States[1, :], marker='o', linestyle='-', color='b')
    for idx, (x, y) in enumerate(zip(States[0, :], States[1, :])):
        axs[0, 0].text(x, y, str(idx), color='red', fontsize=8, ha='center', va='center')
    axs[0, 0].plot(States[0, :], States[1, :], linestyle='-', color='red')


    axs[0, 0].set_title(f"States     s={s}")
    axs[0, 0].set_xlim(np.min([s.x for s in states])-0.5, np.max([s.x for s in states])+0.5)
    axs[0, 0].set_ylim(np.min([s.y for s in states])-0.5, np.max([s.y for s in states])+0.5)
    axs[0, 0].legend()

    axs[0, 1].plot(range(len(S_hat)), S_hat)
    axs[0, 1].grid(True)
    axs[0, 1].set_title("S_hat")

    axs[1, 0].plot(range(len(U[0, :])), U[0, :])
    axs[1, 0].set_title("Throttles")
    axs[1, 0].grid(True)

    axs[1, 1].plot(range(len(U[1, :])), U[1, :])
    axs[1, 1].set_title("Steers")
    axs[1, 1].grid(True)

    axs[0, 2].plot(range(len(states)), [state.v_x for state in states], label="v_x")
    axs[0, 2].plot(range(len(states)), [state.v_y for state in states], label="v_y")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    axs[1, 2].plot(range(len(e_hat_l)), e_hat_l, label="e_hat_l")
    axs[1, 2].plot(range(len(e_hat_c)), e_hat_c, label="e_hat_c")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    plt.show()

    state0 = states[1]
