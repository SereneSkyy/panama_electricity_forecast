import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_clean_data(path):
    df = pd.read_csv(path, parse_dates=["datetime"], dayfirst=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Missing hour interpolation (From Page 4)
    df = df.set_index("datetime").asfreq("h")
    df["nat_demand"] = df["nat_demand"].interpolate(method="linear")
    return df.reset_index()

def add_features(df):
    # Cyclical Encoding (From Page 10)
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Columns to keep for the model (Total 21 features + Target)
    seq_cols = ["nat_demand", "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
                "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
                "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
                "holiday", "school", "hour_sin", "hour_cos", 
                "dow_sin", "dow_cos"]
    return df[seq_cols]

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0]) # target is 'nat_demand' at index 0
    return np.array(X), np.array(y)