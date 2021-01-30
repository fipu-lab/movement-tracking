import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


def plot_initial(ts):
    ts = ts.set_index("timestamp")
    plt.figure(figsize=(18, 10))
    plt.plot(ts)
    mean = ts.mean()
    std = ts.std()
    print(f"Mean {mean}")
    print(f"Std {std}")
    return ts


def trim_outliers(ts, percentile):
    res = ts.clip(upper=ts.quantile(percentile), axis=1)
    return res


def smoothing(ts, lag):
    rolling_ts = ts.rolling(window=lag)
    rolling_mean = rolling_ts.mean()
    fig = plt.figure(figsize=(18, 6))
    plt.plot(ts, label="Original time series")
    plt.plot(rolling_mean, label="Smoothed Moving Average", color="red")
    plt.legend(loc="upper left")
    return rolling_mean


def normalize_series(ts):
    ts = ts.clip(lower=0)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    cols = ts.columns
    np_scaled = min_max_scaler.fit_transform(ts)
    ts = pd.DataFrame(np_scaled, columns=cols)
    return ts
