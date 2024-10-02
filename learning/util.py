import os
import pandas as pd
import torch
import sys


def get_most_recent_subdirectory(parent_directory):
    subdirs = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]    
    subdirs.sort()
    
    # Return the most recent directory (last in the sorted list)
    if subdirs:
        return subdirs[-1]
    else:
        return None
    
def load_dataset(fp, features, targets, device, dtype):
    df = pd.read_csv(fp)
    X =  torch.tensor(df.loc[:, features].to_numpy(), device=device, dtype=dtype)
    y =  torch.tensor(df.loc[:, targets].to_numpy(), device=device, dtype=dtype)
    return X, y
    
class Writer:
    def __init__(self, fp):
        self.fp = fp

    def write(self, string):
        with open(self.fp, 'a') as f:
            print(string, file=sys.stdout)
            print(string, file=f) 

    def __call__(self, string):
        self.write(string)
