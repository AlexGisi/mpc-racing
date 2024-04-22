"""
Verify generation of centerline and error polynomials.
"""
import numpy as np
import matplotlib.pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline
from matplotlib.animation import FuncAnimation

S = 0
LOOKAHEAD = 100

cl = ParameterizedCenterline()
cl.from_file("waypoints/shanghai_intl_circuit")

def poly_eval(coeffs, x):
    num = 0
    for i, c in enumerate(coeffs):
        num += c*(x**(len(coeffs)-i-1))
    return num

def rmse(plot_x, plot_y, true_x, true_y):
    plot = np.array(list(zip(plot_x, plot_y)))
    true = np.array(list(zip(true_x, true_y)))
    return np.sqrt( 1/len(plot_x) * np.square(np.linalg.norm(plot-true)) )

def trial(s):
    x_coeffs = cl.x_as_coeffs(s, LOOKAHEAD)
    y_coeffs = cl.y_as_coeffs(s, LOOKAHEAD)

    x = lambda s: poly_eval(x_coeffs, s)
    y = lambda s: poly_eval(y_coeffs, s)

    ss = np.arange(0, LOOKAHEAD, 1.5) + s
    plot_x = [x(s) for s in ss]
    plot_y = [y(s) for s in ss]
    points = zip(plot_x, plot_y)

    true_x = [cl.Gx(s) for s in ss]
    true_y = [cl.Gy(s) for s in ss]

    return rmse(plot_x, plot_y, true_x, true_y), list(zip(plot_x, plot_y)), list(zip(true_x, true_y)), s


trial_s = np.arange(0, cl.length-LOOKAHEAD-1, 20)
res = [trial(s) for s in trial_s]

### ANIMATION ###
# Animation
fig, ax = plt.subplots()
plt.axis('equal')

# Initial empty plots
line_plot, = ax.plot([], [], 'r-', label='Interpolated Path')
line_true, = ax.plot([], [], 'g-', label='True Path')
ax.legend()

# Set plot limits if known, or dynamically update within update function

def init():
    line_plot.set_data([], [])
    line_true.set_data([], [])
    return line_plot, line_true

def update(frame):
    rmse, plot_points, true_points = res[frame]
    plot_x, plot_y = zip(*plot_points)
    true_x, true_y = zip(*true_points)
    line_plot.set_data(plot_x, plot_y)
    line_true.set_data(true_x, true_y)
    ax.set_xlim(min(plot_x), max(plot_x))
    ax.set_ylim(min(plot_y), max(plot_y))
    return line_plot, line_true

# ani = FuncAnimation(fig, update, frames=len(res), init_func=init, blit=True, repeat=True, interval=100)

for r in res:
    rmse, plot_points, true_points, s = r
    print(f"s: {s}, error: {rmse}")

    plot_x, plot_y = zip(*plot_points)
    true_x, true_y = zip(*true_points)
    ax.set_xlim(min(plot_x), max(plot_x))
    ax.set_ylim(min(plot_y), max(plot_y))
    plt.plot(plot_x, plot_y)
    plt.plot(true_x, true_y)
    plt.show()


# plt.scatter(plot_x, plot_y)
# # plt.plot(plot_x, plot_y)
# plt.show()
# 
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# ax1.plot(plot_x, plot_y)
# 
# ax2.plot(ss, [dx(s) for s in ss])
# ax2.set_title("dx")
# 
# ax3.plot(ss, [dy(s) for s in ss])
# ax3.set_title("dy")
# 
# plt.show()

