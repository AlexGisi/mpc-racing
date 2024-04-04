import pickle
from matplotlib import pyplot as plt
import scipy
from scipy.interpolate import make_interp_spline
from math import sqrt, sin, cos, floor
import numpy as np
 

class ParameterizedCenterline:
    def __init__(self):
        self.spline_x = None
        self.spline_y = None
        self.length = None

    def Gx(self, s):
        if s > self.length:
            raise ValueError
        return self.spline_x(s)
    
    def Gy(self, s):
        if s > self.length:
            raise ValueError
        return self.spline_y(s)
    
    def dGx(self, s):
        if s > self.length:
            raise ValueError
        return self.spline_x.derivative()(s)
    
    def dGy(self, s):
        if s > self.length:
            raise ValueError
        return self.spline_y.derivative()(s)
    
    def ddGx(self, s):
        if s > self.length:
            raise ValueError
        return self.spline_x.derivative().derivative()(s)
    
    def ddGy(self, s):
        if s > self.length:
            raise ValueError
        return self.spline_y.derivative().derivative()(s)
    
    def get_projection_local(self, X, Y, s0, alpha=0.01, epsilon=0.0001):
        """
        Orthogonal projection of the current position (X, Y) onto the centerline,
        which is the result of the optimization problem s* = \min_{s \in [0, L]}
        f(s), where f(s) = \frac{1}{2} || g(s) - [X Y]^T ||^2 is the cost, i.e.
        the distance from (X,Y) to the centerline. We minimize it with gradient descent.

        Distance is not convex in s! So if s0 is far from s* it will not converge correctly!
        """
        df = float('inf')
        iters = 0
        while abs(df) > epsilon:
            df = (self.Gx(s0) - X)*self.dGx(s0) + (self.Gy(s0) - Y)*self.dGy(s0)
            s1 = s0 - alpha * df  # Gradient descent step.
            s0 = s1
            iters += 1

        error_centerline = (self.Gx(s1) - X)**2 + (self.Gy(s1) - Y)**2
        return s1, error_centerline, iters


    def from_file(self, fp):
        with open(fp, 'rb') as f:
            waypoints = pickle.load(f)
        waypoints.pop(0)  # First element is the track id.

        x = [p[0] for p in waypoints]
        y = [p[1] for p in waypoints]

        ss = [0]  # Centerline distances corresponding to each waypoint.
        cum_dist = 0
        for i, waypoint in enumerate(waypoints[:-1]):
            current = waypoints[i]
            next = waypoints[i+1]
            cum_dist += sqrt((current[0] - next[0])**2 + (current[1] - next[1])**2)
            ss.append(cum_dist)

        s = np.array(ss)
        x = np.array(x)
        y = np.array(y)

        self.spline_x = make_interp_spline(s, x, bc_type="clamped")
        self.spline_y = make_interp_spline(s, y, bc_type="clamped")
        self.length = max(ss)
    
    def plot(self, d=False, dd=False, points=None):
        subplot_n = 1 + d + dd
        fig = plt.figure()

        ax1 = fig.add_subplot(subplot_n, 1, 1)
        plotx, ploty = self._get_plotxy(self.Gx, self.Gy)
        ax1.set_title("x and y wrt s")
        ax1.plot(plotx, ploty, 'r-')

        if d:
            ax2 = fig.add_subplot(subplot_n, 1, 2)
            plotdx, plotdy = self._get_plotxy(self.dGx, self.dGy)
            ax2.set_title("Partials of x and y wrt s")
            ax2.plot(plotdx, plotdy, 'b-')
        
        if dd:
            ax3 = fig.add_subplot(subplot_n, 1, subplot_n)
            plotddx, plotddy = self._get_plotxy(self.ddGx, self.ddGy)
            ax3.set_title("Double partials of x and y wrt s")
            ax3.plot(plotddx, plotddy, 'g-')
        
        if points is not None:
            colors = ['r', 'b']
            for i, p in enumerate(points):
                ax1.plot(p[0], p[1], f'{colors[i % len(colors)]}o', markersize=10)

        plt.tight_layout()
        plt.show()

    def _get_plotxy(self, funcx, funcy, points_per_unit=10):
        plots = np.linspace(0, self.length, num=floor(self.length)*points_per_unit)
        plotx = [funcx(s) for s in plots] 
        ploty = [funcy(s) for s in plots]
        return plotx, ploty


if __name__ == '__main__':
    cl = ParameterizedCenterline()
    cl.from_file("../waypoints/shanghai_intl_circuit")

    # Test centerline projection.
    X, Y = (151, 65)
    s, err, iter = cl.get_projection_local(X, Y, 310)

    print(iter)
    cl.plot(d=False, dd=False, points=[[X, Y], [cl.Gx(s), cl.Gy(s)]])
