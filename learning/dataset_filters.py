import pandas as pd
import numpy as np


def uniform_union(columns, n_bins, df):
    dfs = [uniform(c, n_bins, df) for c in columns]
    return pd.concat(dfs, ignore_index=True)


def uniform(column, n_bins, df):
    bins = np.linspace(df[column].min(), df[column].max(), n_bins + 1)

    df['bin'] = pd.cut(df[column], bins=bins, include_lowest=True)
    min_count = df['bin'].value_counts().min()

    df_uniform = df.groupby('bin').apply(lambda x: x.sample(n=min_count)).reset_index(drop=True)
    return df_uniform
