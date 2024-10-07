import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from models.VehicleParameters import VehicleParameters
from control.util import deg2rad
import pandas as pd


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, alpha):
        alpha = torch.unsqueeze(alpha, dim=1)
        y = self.fc1(alpha)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        y = self.fc4(y)
        y = F.relu(y)
        y = torch.squeeze(y)
        
        return y
    
class MLP2(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 1)
        # self.scale = nn.Parameter(torch.tensor([65_000.]))

    def forward(self, alpha):
        # alpha *= self.scale

        alpha = torch.unsqueeze(alpha, dim=1)
        y = self.fc1(alpha)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = torch.squeeze(y)
        
        return y * 1e3


class SectorBoundedMLP(MLP):
    def __init__(self):
        super().__init__()

    def forward(self, alpha):
        y = super().forward(alpha)
        return torch.abs(y) * torch.sign(alpha)


class LinearTire(Module):
    def __init__(self):
        super().__init__()
        self.stiffness = nn.Parameter(torch.tensor([100_000.], dtype=torch.float32))

    def forward(self, alpha):
        return self.stiffness * alpha


class Pacejka(Module):
    def __init__(self):
        super(Pacejka, self).__init__()

        # a0 - a8 in http://www-cdr.stanford.edu/dynamic/bywire/tires.pdf
        # Note the ref has a0 = 1.30.
        self.a = nn.Parameter(torch.tensor([1.3, -22.1, 1011, 1078, 1.82, 0.208, 0.0, -0.354, 0.707], dtype=torch.float32))
        self.register_buffer('Fz', torch.tensor([VehicleParameters.m * 9.81], dtype=torch.float32))

    def forward(self, alpha):
        C = self.a[0]
        D = (self.a[1] * self.Fz + self.a[2]) * self.Fz
        BCD = self.a[3] * torch.sin(self.a[4] * torch.atan(self.a[5] * self.Fz))
        B = BCD / (C * D)
        E = self.a[6] * self.Fz**2 + self.a[7] * self.Fz + self.a[8]
        
        Sh = 0.0  # Camber angle of zero.
        Sv = 0.0

        phi = (1 - E) * (alpha + Sh) + (E / B) * torch.atan(B * (alpha + Sh))
        Fy = D * torch.sin(C * torch.atan(B * phi)) + Sv

        return Fy
    
    def forward2(self, alpha):
        B = 0.714
        C = 1.4
        D = 1.0
        E = -0.2

        Sh = 0.0  # Camber angle of zero.
        Sv = 0.0

        phi = (1 - E) * (alpha + Sh) + (E / B) * torch.atan(B * (alpha + Sh))
        Fy = D * torch.sin(C * torch.atan(B * phi)) + Sv

        return Fy


class Vehicle(Module):
    def __init__(self, tires):
        """Vehicle model

        :param tires: 'pacejka', 'linear'
        :param dtype: defaults to torch.float32
        """
        super(Vehicle, self).__init__()
        self.i = 0

        self.tires = tires
        if tires == "pacejka":
            self.front_tire = Pacejka(VehicleParameters.m * 9.81)
            self.back_tire = Pacejka(VehicleParameters.m * 9.81)
        elif tires == "linear":
            self.front_tire = LinearTire()
            self.back_tire = LinearTire()
        elif tires == 'mlp':
            self.front_tire = MLP()
            self.back_tire = MLP()
        elif tires == 'mlp2':
            self.front_tire = MLP2()
            self.back_tire = MLP2()
        else:
            raise ValueError(f"tires not recognized")

    def forward(self, x):
        """Predict next vehicle state.

        :param x: torch.tensor, (X, Y, yaw, vx, vy, yawdot, throttle, steer, dt).
        """
        # Unpack parameters.
        m = VehicleParameters.m
        Iz = VehicleParameters.Iz
        lf = VehicleParameters.lf
        lr = VehicleParameters.lr

        X, Y, yaw, v_x, v_y, yaw_dot, dt = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 8]
        throttle, steer = x[:, 6], x[:, 7]  # Normalized in [-1, 1].

        # Convert commands to physical values.
        Fx = self.Fx(throttle, v_x)
        delta = self.steer_cmd_to_angle(steer, v_x, v_y)

        # Calculate slip angles.
        theta_Vf = torch.atan2((v_y + lf * yaw_dot), v_x+0.1)
        theta_Vr = torch.atan2((v_y - lr * yaw_dot), v_x+0.1)

        df = pd.DataFrame(torch.stack([throttle, steer, v_x, v_y, yaw_dot, dt, theta_Vf, theta_Vr], dim=1).detach().cpu().numpy(), columns=['throttle', 'steer', 'vx', 'vy', 'yawdot', 'dt', 'slip_front', 'slip_back'])
        df.to_csv(f'/home/alex/projects/graic/autobots-race/learning/slip.csv', index=False)
        # self.i += 1

        # Calculate lateral forces at front and rear.
        Fyf = self.front_tire(delta - theta_Vf)
        Fyr = self.back_tire(-theta_Vr)

        # Dynamics equations
        # See "Online Learning of MPC for Autonomous Racing" by Costa et al
        v_x_dot = ( (Fx - Fyf*torch.sin(delta)) / m ) + (v_y * yaw_dot)
        v_y_dot = ((Fyf*torch.cos(delta) + Fyr) / m) - (v_x * yaw_dot)
        yaw_dot_dot = ( (Fyf*torch.cos(delta)*lf) - (Fyr*lr)) / Iz

        # Integrate to find new state
        x_new = X + (v_x * torch.cos(yaw) - v_y * torch.sin(yaw)) * dt
        y_new = Y + (v_x * torch.sin(yaw) + v_y * torch.cos(yaw)) * dt
        yaw_new = yaw + yaw_dot * dt
        v_x_new = v_x + v_x_dot * dt
        v_y_new = v_y + v_y_dot * dt
        yaw_dot_new = yaw_dot + yaw_dot_dot * dt

        state_new = torch.stack([x_new, y_new, yaw_new, v_x_new, v_y_new, yaw_dot_new], dim=1)
        assert(len(state_new) == len(x))

        if torch.isnan(state_new).any():
            breakpoint()

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
