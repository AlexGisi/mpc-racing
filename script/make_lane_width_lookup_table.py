"""
Make a width lookup table with ss values every 0.5m,
so it is easy to round your value and index into the table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from splines.ParameterizedCenterline import ParameterizedCenterline
from multiprocessing import Pool

def process_errors(s):
    right, _ = cl.get_errors(cl.right_lane, s, 0)
    left, _ = cl.get_errors(cl.left_lane, s, 0)
    return right[0], left[0]

def init(args):
    # This is the initialization for the global variables in the pool workers
    global cl
    cl = args

def plot(ss, right_es, left_es):
    norms = [cl.unit_principal_normal(s) for s in ss]
    right_error_vectors = [(x*e, y*e) for (x, y), e in zip(norms, right_es)]
    left_error_vectors = [(x*e, y*e) for (x, y), e in zip(norms, left_es)]
    
    s_points = [(cl.Gx(s), cl.Gy(s)) for s in ss]
    e_points_right = [(x+ex, y+ey) for (x, y), (ex, ey) in zip(s_points, right_error_vectors)]
    e_points_left = [(x-ex, y-ey) for (x, y), (ex, ey) in zip(s_points, left_error_vectors)]

    plt.scatter([x for x, y in s_points], [y for x, y in s_points], label="s points")
    plt.scatter([x for x, y in e_points_right], [y for x, y in e_points_right], label="error right")
    plt.scatter([x for x, y in e_points_left], [y for x, y in e_points_left], label="error left")
    plt.plot([x for x, y in cl.left_lane.waypoints], [y for x, y in cl.left_lane.waypoints], 'r', label="left lane")
    plt.plot([x for x, y in cl.right_lane.waypoints], [y for x, y in cl.right_lane.waypoints], 'r', label="right lane")
    plt.grid(True)
    plt.legend()
    plt.show()

def to_file(ss, right_es, left_es):
    fp = f"lanes/{TRACK_NAME}_max_error.csv"
    df  = pd.DataFrame(data={'ss': ss, 'right': right_es, 'left': left_es})
    df.to_csv(fp, index=False)

if __name__ == "__main__":
    TRACK_NAME = "shanghai_intl_circuit"
    POINTS_COUNT = 4000

    cl = ParameterizedCenterline()
    cl.from_file(f"waypoints/{TRACK_NAME}")

    ss = np.arange(0, cl.length, step=0.5)
    num_cores = 14  # Adjust this to the number of cores you want to use

    # Create a Pool with the desired number of processes and initialize it
    with Pool(processes=num_cores, initializer=init, initargs=(cl,)) as pool:
        results = pool.map(process_errors, ss)

    # Extract right and left errors from results
    right_es, left_es = zip(*results)

    # Convert the results to numpy arrays
    right_es = np.array(right_es)
    left_es = np.array(left_es)

    plot(ss, right_es, left_es)
    to_file(ss, right_es, left_es)
