import pandas as pd
import learning.util as util
from models.VehicleParameters import VehicleParameters

PATH = 'data/big/slip_validate.csv'
path = util.get_abs_fp(__file__, PATH)

df = pd.read_csv(path)
df['ay'] = df['vy'].diff() / df['dt']
df['fy'] = df['ay'] * VehicleParameters.m

df.to_csv(path, index=False)
