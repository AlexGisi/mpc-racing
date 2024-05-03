import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from models.State import State
from models.VehicleParameters import VehicleParameters

###
def f_vehicle_(x_k, u_k):
    # Unpack parameters.
    m = VehicleParameters.m
    Iz = VehicleParameters.Iz
    lf = VehicleParameters.lf
    lr = VehicleParameters.lr
    Cf = VehicleParameters.Cf
    Cr = VehicleParameters.Cr
    Ts = VehicleParameters.Ts

    x, y, yaw, v_x, v_y, yaw_dot = x_k[0], x_k[1], x_k[2], x_k[3], x_k[4], x_k[5]

    # Convert commands to physical values.
    Fx = u_k[0]*1000.0
    delta = u_k[1]*(70.0 / 360 * 2 * 3.14)

    # Calculate slip angles.
    theta_Vf = ca.atan2((v_y + lf * yaw_dot), v_x+0.01)
    theta_Vr = ca.atan2((v_y - lr * yaw_dot), v_x+0.01)

    # Calculate lateral forces at front and rear using linear tire model.
    Fyf = Cf * (delta - theta_Vf)
    Fyr = Cr * (-theta_Vr)

    # Dynamics equations
    # See "Online Learning of MPC for Autonomous Racing" by Costa et al
    v_x_dot = ( (Fx - Fyf*ca.sin(delta)) / m ) + (v_y * yaw_dot)
    v_y_dot = ((Fyf*ca.cos(delta) + Fyr) / m) - (v_x * yaw_dot)
    yaw_dot_dot = ( (Fyf*ca.cos(delta)*lf) - (Fyr*lr)) / Iz

    # Integrate to find new state
    x_new = x + (v_x * ca.cos(yaw) - v_y * ca.sin(yaw)) * Ts
    y_new = y + (v_x * ca.sin(yaw) + v_y * ca.cos(yaw)) * Ts
    yaw_new = yaw + yaw_dot * Ts
    v_x_new = v_x + v_x_dot * Ts
    v_y_new = v_y + v_y_dot * Ts
    yaw_dot_new = yaw_dot + yaw_dot_dot * Ts

    state_new = ca.vertcat(x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new)

    return state_new
###

N = 50
min_steer = -1.0
max_steer = 1.0
min_throttle = -1.0
max_throttle = 1.0
state0 = State(x=0, y=0, yaw=0, v_x=5, v_y=0, yaw_dot=0)

state_direction = np.array([np.cos(state0.yaw), np.sin(state0.yaw)])
state_direction = state_direction / np.linalg.norm(state_direction)
dx = np.zeros(shape=(N, 1))
dy = np.zeros(shape=(N, 1))

opti = ca.Opti()
U = opti.variable(2, N)
X = opti.variable(6, N+1)
u = ca.MX.sym('u', 2, N)
x = ca.MX.sym('x', 6, N+1)

opti.subject_to(X[0, 0] == state0.x)
opti.subject_to(X[1, 0] == state0.y)
opti.subject_to(X[2, 0] == state0.yaw)
opti.subject_to(X[3, 0] == state0.v_x)
opti.subject_to(X[4, 0] == state0.v_y)
opti.subject_to(X[5, 0] == state0.yaw_dot)

f_vehicle = ca.Function('f_vehicle', [X, u], [f_vehicle_(X, u)])

for i in range(1, N+1):
    dx[i-1] = state_direction[0] * i * VehicleParameters.Ts * (VehicleParameters.max_vel / 5)
    dy[i-1] = state_direction[1] * i * VehicleParameters.Ts * (VehicleParameters.max_vel / 5)
    opti.set_initial(X[0, i], state0.x+dx[i-1])
    opti.set_initial(X[1, i], state0.y+dy[i-1])
    opti.set_initial(X[2, i], state0.yaw)
    opti.set_initial(X[3, i], state0.v_x)
    opti.set_initial(X[4, i], state0.v_y)
    opti.set_initial(X[5, i], state0.yaw_dot)

    opti.subject_to(X[:, i] == f_vehicle(X[:, i-1], U[:, i-1]))

for i in range(0, N):
    opti.subject_to(U[0, i] < max_throttle)
    opti.subject_to(U[0, i] > min_throttle)
    opti.subject_to(U[1, i] < max_steer)
    opti.subject_to(U[1, i] > min_steer)

J = (X[0, N] - 10)**2 + (X[1, N] - 10)**2

opti.minimize(J)
opti.solver('ipopt', {
    'ipopt': {
        'max_iter': 100,  # Maximum number of iterations
        'print_level': 5,  # Adjust to control the verbosity of IPOPT output
        'tol': 1e-5  # Solver tolerance
    }
})
try:
    sol = opti.solve()
    throttles = sol.value(U)[0]
    steers = sol.value(U)[1]

    xs = sol.value(X)[0]
    ys = sol.value(X)[1]
    yaws = sol.value(X)[2]
except RuntimeError as e:
    if opti.return_status() == 'Maximum_Iterations_Exceeded':
        throttles = opti.debug.value(U)[0]
        steers = opti.debug.value(U)[1]

        xs = opti.debug.value(X)[0]
        ys = opti.debug.value(X)[1]
        yaws = opti.debug.value(X)[2]

fig, axs = plt.subplots(3, 1)

axs[0].plot(state0.x, state0.y, 'o', ms=10, label="Initial")
axs[0].plot(10, 10, 'o', ms=10, label="Target")
axs[0].plot([x for x in xs], [y for y in ys])
axs[0].scatter([state0.x+dxx for dxx in dx], [state0.y+dyy for dyy in dy], s=4, label="Guess")
axs[0].legend()

axs[1].plot(range(N), throttles)
axs[1].set_title("Throttles")
axs[1].grid(True)

axs[2].plot(range(N), steers)
axs[2].grid(True)
axs[2].set_title("Steers")

plt.show()
