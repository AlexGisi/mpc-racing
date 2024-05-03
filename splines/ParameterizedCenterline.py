from typing import List, Type
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from math import floor
import numpy as np
from numpy.linalg import norm
from splines.ParameterizedLine import ParameterizedLine
from splines.ParameterizedLane import ParameterizedLane
from splines.util import euclidean, midpoint

class ParameterizedCenterline(ParameterizedLine):
    def __init__(self, track: str = "shanghai_intl_circuit"):
        super().__init__()

        self.right_lane = ParameterizedLane()
        self.right_lane.from_file(f"lanes/{track}_left.csv")

        self.left_lane = ParameterizedLane()
        self.left_lane.from_file(f"lanes/{track}_right.csv")

        self.lane_error_table = pd.read_csv(f"lanes/{track}_max_error.csv",
                                            index_col='ss')
        self.from_file(f"waypoints/{track}")
    
    def e_as_coeffs(self, s, lookahead):
        """
        Too slow for MPC, lookup table used instead.

        Get the minimum over the centerlines of the maximum drivable track error.
        The track is symmetric enough over the centerline. 
        """
        right_errors, ss = self.get_errors(self.right_lane, s, lookahead)
        left_errors, ss = self.get_errors(self.left_lane, s, lookahead)
        errors = [min(l, r) for l, r in zip(right_errors, left_errors)]
        coeffs = np.polyfit(ss, errors, deg=3)
        return list(coeffs)

    def get_errors(self, lane: Type[ParameterizedLane], s, lookahead: float, step=0.5):
        """
        Too slow for MPC, lookup table used instead.
        """
        errors = []
        if lookahead > 0:
            ss = np.arange(s, s+lookahead, step)
        elif lookahead == 0:
            ss = [s]
        else:
            raise ValueError()
        
        for s in ss:
            _, dist = lane.projection(self.Gx(s), self.Gy(s), bounds=lane.progress_bounds(step=step))
            errors.append(dist)
        lane.last_progress = None

        return errors, ss
    
    def lookup_error(self, s, lookahead):
        """
        Use the lookup table to find the smallest error in the prediction horizon.
        """
        s_round = round(s * 2) / 2
        lookahead_round = round(lookahead * 2) / 2
        ss = np.arange(s_round, s+lookahead_round, 0.5)
        left_min = 10000
        right_min = 10000
        for s in ss:
            s = s % self.length
            r = self.lane_error_table.loc[s]
            if r['left'] < left_min:
                left_min = r['left']
            if r['right'] < right_min:
                right_min = r['right']

        return min(right_min, left_min)

    def error_sign(self, X, Y, s):
        """
        Based on the progress, get the sign of the centerline error of the car.

        To the right of the centerline is positive.
        """
        upn = np.array(self.unit_principal_normal(s))
        centerline_displacement = np.array([X - self.Gx(s), Y - self.Gy(s)])

        return 1 if norm(centerline_displacement - upn) < norm(centerline_displacement + upn) else -1

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

        self.from_waypoints(waypoints)
    
    def plot(self,
             d=False, 
             dd=False,
             between=None,
             show=True,
             points=None,
             pointsize=10,
             labels=None, 
             waypoints=False,
             lanes=True):
        subplot_n = 1 + d + dd
        fig = plt.figure()

        ax1 = fig.add_subplot(subplot_n, 1, 1)
        plotx, ploty = self._get_plotxy(self.Gx, self.Gy, between=between)
        ax1.set_title("Waypoints")
        ax1.plot(plotx, ploty, 'r-')

        if lanes:
            ax1.plot([x for x,y in self.left_lane.waypoints], [y for x,y in self.left_lane.waypoints], 'b')
            ax1.plot([x for x,y in self.right_lane.waypoints], [y for x,y in self.right_lane.waypoints], 'b')

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
    cl.plot(waypoints=True)
    