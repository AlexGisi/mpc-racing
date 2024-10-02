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

def get_abs_fp(file_fp, rel_fp):
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(file_fp)), rel_fp)
    )
    
class Writer:
    def __init__(self, fp, delete_if_exists=False):
        self.fp = fp
        
        if delete_if_exists and os.path.exists(fp):
            os.remove(fp)

    def write(self, string):
        with open(self.fp, 'a') as f:
            print(string, file=sys.stdout)
            print(string, file=f) 

    def __call__(self, string):
        self.write(string)
