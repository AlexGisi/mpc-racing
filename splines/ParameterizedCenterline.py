from typing import List
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar, dual_annealing
from scipy.interpolate import make_interp_spline
from math import sqrt, floor
import numpy as np
from numpy.linalg import norm
from splines.util import euclidean, midpoint


class ParameterizedCenterline:
    def __init__(self):
        self.spline_x = None
        self.spline_y = None
        self.length = None
        self.waypoints = None

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
    
    def x_as_coeffs(self, s, lookahead) -> List[float]:
        """
        s: current progress
        lookahead: how far ahead should I get points for interpolation
        """
        interps = np.arange(0, lookahead, 2) + s
        interpx = np.array([self.Gx(s) for s in interps])
        coeffs = np.polyfit(interps, interpx, deg=10)
        return list(coeffs)

    def y_as_coeffs(self, s, lookahead) -> List[float]:
        """
        s: current progress
        lookahead: how far ahead should I get points for interpolation
        """
        interps = np.arange(0, lookahead, 6) + s
        interpy = np.array([self.Gy(s) for s in interps])
        coeffs = np.polyfit(interps, interpy, deg=10)
        return list(coeffs)

    def projection(self, X, Y, bounds=None):
        """
        Interface which wraps both projection functions and chooses the
        right one, depending on the bounds situation.

        X: float
        Y: float
        bounds: 2-dim Tuple
        """
        if bounds is None or 5 < abs(bounds[1] - bounds[0]):
            return self.projection_global(X, Y)
        else:
            return self.projection_local(X, Y, bounds=bounds)

    def projection_local(self, X, Y, bounds=None):
        """
        Get the progress along the track when the car is at (X, Y) and
        the distance of (X, Y) from the centerline (absolute centerline error).
        
        Must use reasonable bounds (true projection +- 5m is a good
        reference) because the distance is nonconvex wrt s.

        Takes ~0.0002s.
        """
        if bounds is None:
            bounds = (0, self.length)
        if bounds[1] - bounds[0] > 10:
            print("Warning: local projection over with large bounds", bounds)

        dist = lambda s: sqrt((self.Gx(s) - X)**2 + (self.Gy(s) - Y)**2)
        ret = minimize_scalar(dist, bounds=bounds)
        return ret.x, dist(ret.x)
    
    def projection_global(self, X, Y):
        """
        Takes ~0.065s.
        """
        dist = lambda s: sqrt((self.Gx(s) - X)**2 + (self.Gy(s) - Y)**2)
        ret = dual_annealing(dist, bounds=[(0, self.length)])
        return ret.x[0], dist(ret.x)

    def error_sign(self, X, Y, s):
        """
        Based on the progress, get the sign of the centerline error of the car.

        To the right of the centerline is positive.
        """
        upn = np.array(self.unit_principal_normal(s))
        centerline_displacement = np.array([X - self.Gx(s), Y - self.Gy(s)])

        return 1 if norm(centerline_displacement - upn) < norm(centerline_displacement + upn) else -1
    
    def unit_tangent(self, s):
        dr = np.array([self.dGx(s), self.dGy(s)])
        ut = dr / np.linalg.norm(dr)
        return ut
    
    def curvature(self, s):
        dx = self.dGx(s)
        dy = self.dGy(s)

        ddx = self.ddGx(s)
        ddy = self.ddGy(s)

        kappa = np.abs(dx * ddy - dy * ddx)
        return kappa

    def unit_principal_normal(self, s):
        """
        Return the unit vector orthogonal to the curve at s which forms
        a right-hand coordinate system with the tangent vector.

        Using the defition, the unit principal normal vector is
        found with

        ddr = np.array([self.ddGx(s), self.ddGy(s)])
        upn = ddr / np.linalg.norm(ddr)

        There seems to be some numerical issue with this due to higher
        order derivatives in interpolation, very notable on horizontal 
        (across the x-axis) lines. The unit_tangent function seems
        to be well-conditioned, so we simply use it and compute the
        associated tangent.
        """
        ut_x, ut_y = self.unit_tangent(s)
        return ut_y, -ut_x
    
    def unit_principal_normal1(self, s):
        """
        Do not use! Use unit_principal_normal.
        """
        ddr = np.array([self.ddGx(s), self.ddGy(s)])
        upn = ddr / np.linalg.norm(ddr)
        return upn[0], upn[1]

    def from_file(self, fp):
        with open(fp, 'rb') as f:
            waypoints = pickle.load(f)
        waypoints.pop(0)  # First element is the track id.

        # If final waypoint not the same as start, add another one
        # to close the gap.
        if euclidean(waypoints[-1], waypoints[0]) > 0.1:
            waypoints.append(
                midpoint(waypoints[-1], waypoints[0], alpha=0.9)
            )

        ss = [0]  # Centerline distances corresponding to each waypoint.
        cum_dist = 0
        for i, waypoint in enumerate(waypoints[:-1]):
            current = waypoints[i]
            next = waypoints[i+1]
            cum_dist += euclidean(current, next)
            ss.append(cum_dist)

        s = np.array(ss)
        x = np.array([p[0] for p in waypoints])
        y = np.array([p[1] for p in waypoints])

        self.spline_x = make_interp_spline(s, x)
        self.spline_y = make_interp_spline(s, y)
        self.length = ss[-1]
        self.waypoints = list(zip(x, y))
    
    def plot(self,
             d=False, 
             dd=False,
             between=None,
             show=True,
             points=None,
             pointsize=10,
             labels=None, 
             waypoints=False):
        subplot_n = 1 + d + dd
        fig = plt.figure()

        ax1 = fig.add_subplot(subplot_n, 1, 1)
        plotx, ploty = self._get_plotxy(self.Gx, self.Gy, between=between)
        ax1.set_title("Waypoints")
        ax1.plot(plotx, ploty, 'r-')

        if waypoints:
            x = [p[0] for p in self.waypoints]
            y = [p[1] for p in self.waypoints]
            ax1.scatter(x, y)

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
            colors = ['r', 'b', 'g']
            for i, p in enumerate(points):
                ax1.plot(p[0], p[1], f'{colors[i % len(colors)]}o', markersize=pointsize)
                if labels is not None:
                    ax1.text(p[0], p[1], labels[i], fontsize=9, ha='right', va='bottom')

        plt.tight_layout()
        if show is True:
            plt.show()
            
        return fig

    def _get_plotxy(self, funcx, funcy, points_per_unit=10, between=None):
        lower = between[0] if between is not None else 0
        upper = between[1] if between is not None else self.length
        plots = np.linspace(lower, upper, num=floor(upper-lower)*points_per_unit)
        plotx = [funcx(s) for s in plots] 
        ploty = [funcy(s) for s in plots]
        return plotx, ploty


if __name__ == '__main__':
    cl = ParameterizedCenterline()
    cl.from_file("waypoints/shanghai_intl_circuit")
    breakpoint()

    cl.plot(waypoints=True)
    