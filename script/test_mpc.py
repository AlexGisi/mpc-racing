import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from control.ControllerParameters import RuntimeControllerParameters, FixedControllerParameters
from control.MPC import MPC
from control.util import make_poly
from models.State import State

POLY_DEG = 4

cl = ParameterizedCenterline(track="shanghai_intl_circuit")
# s = np.random.uniform(low=0, high=cl.length)
s0 = 50
ss = np.arange(s0, cl.length, 0.5)
sol = None
dual = None

x, y = cl.Gx(s0), cl.Gy(s0)
yaw = cl.unit_tangent_yaw(s0)
v_x, v_y, r = 25, 0, 0
state0 = State(x=x, y=y, yaw=yaw, v_x=v_x, v_y=v_y, yaw_dot=r)

while True:
    dynamic_lookahead = FixedControllerParameters.Ts * (state0.v_x + (15/40)*FixedControllerParameters.N) * FixedControllerParameters.N
    cl_x_coeffs = cl.x_as_coeffs(s0, dynamic_lookahead, deg=POLY_DEG)
    cl_y_coeffs = cl.y_as_coeffs(s0-5, dynamic_lookahead, deg=POLY_DEG)
    max_err = cl.lookup_error(s0, dynamic_lookahead)
        
    states = [state0]
    mpc = MPC(state=state0,
              sol0=None,
              duals=None,
              s0=s0,
              centerline_x_poly_coeffs=cl_x_coeffs,
              centerline_y_poly_coeffs=cl_y_coeffs,
              max_error=max_err,
              runtime_params=RuntimeControllerParameters())

    sol, ret, dual = mpc.solution()
    States, U, S_hat, e_hat_c, e_hat_l = ret[0], ret[1], ret[2], ret[3], ret[4]
    states.extend([State(x=t[0], y=t[1], yaw=t[2], v_x=t[3], v_y=t[4], yaw_dot=t[5]) for t in zip(States[0, :], States[1, :], States[2, :], States[3, :], States[4, :], States[5, :])])

    ### Plotting
    def state_arrow(state, length=3):
        dx = length * np.cos(state.yaw)
        dy = length * np.sin(state.yaw)
        return state.x, state.y, dx, dy

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 3)

    # Assign subplots to the grid
    ax1 = fig.add_subplot(gs[0, :])  # Makes ax1 span all columns in the first row
    ax2 = fig.add_subplot(gs[1, 0])  # Remaining subplots in specified positions
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    cl_x = lambda s: make_poly(s, cl_x_coeffs)
    cl_y = lambda s: make_poly(s, cl_y_coeffs)
    ss = np.linspace(s0-5, s0+dynamic_lookahead, 25)
    ax1.plot([cl_x(s) for s in ss], [cl_y(s) for s in ss], linewidth=2, label="Polynomial centerline")

    ax1.plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r')
    ax1.plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r')
    ax1.plot(state0.x, state0.y, marker='o', zorder=100, color='g', ms=6, label="Initial position")
    ax1.scatter(States[0, :], States[1, :], zorder=50, s=6, linewidths=4, marker='o', linestyle='-', color='r', label="Predicted path")
    # for idx, (x, y) in enumerate(zip(States[0, :], States[1, :])):
    #     axs[0, 0].text(x, y, str(idx), color='red', fontsize=8, ha='center', va='center')

    ax1.set_title(r"States ($s_0$=" + str(s0) + ")")
    ax1.set_xlim(np.min([s.x for s in states])-60, np.max([s.x for s in states])+60)
    ax1.set_ylim(np.min([s.y for s in states])-20, np.max([s.y for s in states])+20)
    ax1.set_aspect('equal')

    ax1.legend(loc='upper center', fancybox=True, shadow=True, ncol=3, fontsize=6)

    ax5.plot(range(len(U[1, :])), U[1, :])
    ax5.set_title("Steer commands")
    ax5.set_xlabel("Step")
    ax5.grid(True)

    ax2.plot(range(len(U[0, :])), U[0, :])
    ax2.set_title("Throttle commands")
    ax2.set_xlabel("Step")
    ax2.grid(True)

    ax3.plot(range(len(states)), [state.v_x for state in states], label="v_x")
    ax3.plot(range(len(states)), [state.v_y for state in states], label="v_y")
    ax3.set_title("Predicted velocities")
    ax3.set_xlabel("Step")
    ax3.legend()
    ax3.grid(True)

    ax4.plot(range(len(S_hat)), S_hat, label="$\hat{s}$")
    ax4.plot(range(len(states)), [cl.projection_local(s.x, s.y, bounds=(s0-10, s0+100), warn=False)[0] for s in states], label="s")
    ax4.set_xlabel("Step")
    ax4.grid(True)
    ax4.legend()
    ax4.set_title(r"Estimated and actual centerline progress")

    ax6.plot(range(len(e_hat_l)), e_hat_l, label=r"$\hat{e}_l$")
    ax6.plot(range(len(e_hat_c)), e_hat_c, label=r"$\hat{e}_c$")
    ax6.set_title("Estimated centerline and lag errors")
    ax6.set_xlabel("Step")
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()

    s0=s0 + 15

    # Perturb the car position from the centerline
    upr = np.array(cl.unit_principal_normal(s0)) * np.random.uniform(-5, 5)
    state0.x, state0.y = cl.Gx(s0)+upr[0], cl.Gy(s0)+upr[1]
    state0.yaw = cl.unit_tangent_yaw(s0)
