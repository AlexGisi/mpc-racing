import torch
from torch.nn import Module
from models.VehicleParameters import VehicleParameters
from control.util import deg2rad


class Vehicle(Module):
    def __init__(self, ts):
        super(Vehicle, self).__init__()
        self.register_buffer('ts', ts)

    def forward(self, x):
        """Predict next vehicle state.

        :param x: torch.tensor, (X, Y, yaw, vx, vy, yawdot, throttle, steer).
        """
        # Unpack parameters.
        m = VehicleParameters.m
        Iz = VehicleParameters.Iz
        lf = VehicleParameters.lf
        lr = VehicleParameters.lr
        Cf = VehicleParameters.Cf
        Cr = VehicleParameters.Cr

        X, Y, yaw, v_x, v_y, yaw_dot = x[0], x[1], x[2], x[3], x[4], x[5]
        throttle, steer = x[6], x[7]  # Normalized in [-1, 1].

        # Convert commands to physical values.
        Fx = self.Fx(throttle, v_x)
        delta = self.steer_cmd_to_angle(steer, v_x, v_y)

        # Calculate slip angles.
        theta_Vf = torch.atan2((v_y + lf * yaw_dot), v_x+0.1)
        theta_Vr = torch.atan2((v_y - lr * yaw_dot), v_x+0.1)

        # Calculate lateral forces at front and rear using linear tire model.
        Fyf = Cf * (delta - theta_Vf)
        Fyr = Cr * (-theta_Vr)

        # Dynamics equations
        # See "Online Learning of MPC for Autonomous Racing" by Costa et al
        v_x_dot = ( (Fx - Fyf*torch.sin(delta)) / m ) + (v_y * yaw_dot)
        v_y_dot = ((Fyf*torch.cos(delta) + Fyr) / m) - (v_x * yaw_dot)
        yaw_dot_dot = ( (Fyf*torch.cos(delta)*lf) - (Fyr*lr)) / Iz

        # Integrate to find new state
        x_new = X + (v_x * torch.cos(yaw) - v_y * torch.sin(yaw)) * self.ts
        y_new = Y + (v_x * torch.sin(yaw) + v_y * torch.cos(yaw)) * self.ts
        yaw_new = yaw + yaw_dot * self.ts
        v_x_new = v_x + v_x_dot * self.ts
        v_y_new = v_y + v_y_dot * self.ts
        yaw_dot_new = yaw_dot + yaw_dot_dot * self.ts

        state_new = torch.tensor([x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new])
        return state_new
    
    def steer_cmd_to_angle(self, steer_cmd, v_x, v_y):
        """
        Maps a steer command in [-1, 1] to the angle of the wheels over the next timestep.
        """
        vel = torch.sqrt(v_x**2 + v_y**2) * 3.6  # km/h

        # Define gain based on velocity via sim-defined steering curve.
        gain = -0.001971664699 * vel + 0.986547

        return deg2rad(steer_cmd * gain * VehicleParameters.max_steer)

    def Fx(self, throttle, v_x):
        wheel_rpm = (v_x / VehicleParameters.C_wheel) * 60
        rpm = wheel_rpm * VehicleParameters.R * 4.5  # 4.5 is an estimated adjust based on observation

        eta = -0.00004428225806 * rpm + 1.282413306
        
        wheel_force = throttle*eta*VehicleParameters.T_max*VehicleParameters.R / VehicleParameters.r_wheel
        drag_force = 0.5*VehicleParameters.rho*VehicleParameters.C_d*VehicleParameters.A_f*(v_x**2)
        rolling_resistance = VehicleParameters.C_roll*VehicleParameters.m*VehicleParameters.g

        return wheel_force - drag_force - rolling_resistance
