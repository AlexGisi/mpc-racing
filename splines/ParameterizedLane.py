import pandas as pd
from matplotlib import pyplot as plt
from splines.ParameterizedLine import ParameterizedLine


class ParameterizedLane(ParameterizedLine):
    def __init__(self):
        super().__init__()
        self.last_progress = None  # Track progress when last queried for error.

    def progress_bounds(self, step=0.5):
        if self.last_progress is None:
            return None
        else:
            return (self.last_progress-(2*step), self.last_progress+(step*6))

    def projection(self, X, Y, bounds=None):
        ret = super().projection(X, Y, bounds)
        self.last_progress = ret[0]
        return ret
    
    def from_file(self, fp):
        df = pd.read_csv(fp)
        waypoints = list(zip(df['x'], df['y']))
        self.from_waypoints(waypoints)
