from typing import List
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize_scalar, dual_annealing
from math import sqrt, floor
import numpy as np
from numpy.linalg import norm
from splines.util import euclidean, midpoint


class ParameterizedLine:
    def __init__(self):
        self.spline_x = None
        self.spline_y = None
        self.length = None
        self.waypoints = None
    
    def Gx(self, s):
        s = s % self.length
        return self.spline_x(s)
    
    def Gy(self, s):
        s = s % self.length
        return self.spline_y(s)
    
    def dGx(self, s):
        s = s % self.length
        return self.spline_x.derivative()(s)
    
    def dGy(self, s):
        s = s % self.length
        return self.spline_y.derivative()(s)
    
    def ddGx(self, s):
        s = s % self.length
        return self.spline_x.derivative().derivative()(s)
    
    def ddGy(self, s):
        s = s % self.length
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
        raise NotImplementedError()
    
    def from_waypoints(self, waypoints):
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
