import scipy.integrate as integrate
import scipy.optimize as optimize
import math
from matplotlib import pyplot as plt
import ParameterizedCenterline

class TrackSegments:
    def __init__(self, centerline, n_cp, v_max, d_f, d_r, m):
        self.line = centerline
        self.n_cp = n_cp
        self.v_max = v_max
        self.curve_c = (2 * d_f + 2 * d_r) / m
        self.lap_time = self.segment_time(0, centerline.length)
        self.bounds = self.calculate_segment_bounds()

    def curve_radius(self, s):
        return 1 / abs((self.line.dGx(s) - self.line.dGy(s)) * self.line.ddGy(s))
    
    def curve_velocity(self, s):
        return min(self.v_max, math.sqrt(self.curve_radius(s) * self.curve_c))
    
    def segment_time(self, s_0, s_1):
        return integrate.quad(lambda x: 1 / self.curve_velocity(x), s_0, s_1, limit=500)[0]
    
    def calculate_segment_bounds(self):
        bounds = [0]
        for i in range(self.n_cp):
            seg_bound = optimize.fsolve(lambda x: self.segment_time(bounds[-1], x) - (self.lap_time / self.n_cp), bounds[-1])[0]
            bounds.append(seg_bound)
        return bounds
    
    def get_bound(self, i):
        assert(i >= 0)
        assert(i < self.n_cp)
        return self.bounds[i]
    
    def plot(self):
        subplot_n = 1
        fig = plt.figure()

        ax1 = fig.add_subplot(subplot_n, 1, 1)
        plotx, ploty = self.line._get_plotxy(self.line.Gx, self.line.Gy)
        ax1.set_title("x and y wrt s")
        ax1.plot(plotx, ploty, 'r-')

        cp_x = []
        cp_y = []

        for bound in self.bounds:
            cp_x.append(self.line.Gx(bound))
            cp_y.append(self.line.Gy(bound))
        ax1.scatter(cp_x, cp_y, s=50, c='blue')

        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    line = ParameterizedCenterline.ParameterizedCenterline()
    line.from_file("./waypoints/shanghai_intl_circuit")

    n_cp = 10

    segments = TrackSegments(line, n_cp, 30, 1500, 1500, 250)

    segments.plot()