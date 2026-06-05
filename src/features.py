import pandas as pd
import numpy as np

def add_time_features(df):
    df['hour']      = df['ts_min'] // 60
    df['minute']    = df['ts_min'] % 60
    df['time_slot'] = df['ts_min'] // 15   # 96 slots/day

    # Cyclical encoding (so 23:45 is close to 0:00)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['slot_sin']  = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['slot_cos']  = np.cos(2 * np.pi * df['time_slot'] / 96)

    # Peak hour flag
    df['is_peak'] = (
        ((df['hour'] >= 7)  & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)

    # Interaction: lanes × peak
    df['lanes_peak'] = df['NumberofLanes'] * df['is_peak']
    return df

def add_lag_features(train, test):
    """
    Key insight: test is all day 49.
    Day-48 demand at same (geohash, timestamp) is the strongest predictor.
    """
    train48 = train[train['day'] == 48][['geohash', 'ts_min', 'demand']].copy()

    lag = train48.rename(columns={'demand': 'demand_d48'})

    train['demand_d48'] = train.merge(lag, on=['geohash', 'ts_min'], how='left')['demand_d48'].values
    test['demand_d48']  = test.merge(lag,  on=['geohash', 'ts_min'], how='left')['demand_d48'].values

    return train, test

def add_aggregate_features(train, test):
    """
    Build aggregate stats from day-48 data only (no leakage).
    """
    train48 = train[train['day'] == 48].copy()
    train48['geo4'] = train48['geohash'].str[:4]

    # geohash × timeslot mean (captures location-specific rush-hour patterns)
    gh_ts = (train48.groupby(['geohash', 'ts_min'])['demand']
             .mean().reset_index()
             .rename(columns={'demand': 'gh_ts_mean'}))

    # geohash mean and std
    gh_mean = (train48.groupby('geohash')['demand']
               .mean().reset_index()
               .rename(columns={'demand': 'gh_mean'}))
    gh_std  = (train48.groupby('geohash')['demand']
               .std().reset_index()
               .rename(columns={'demand': 'gh_std'}))

    # global timeslot mean
    ts_mean = (train48.groupby('ts_min')['demand']
               .mean().reset_index()
               .rename(columns={'demand': 'ts_mean'}))

    # geo4-prefix mean (for unseen geohashes)
    geo4_mean = (train48.groupby('geo4')['demand']
                 .mean().reset_index()
                 .rename(columns={'demand': 'geo4_mean'}))

    for df in [train, test]:
        df['geo4'] = df['geohash'].str[:4]
        df['gh_ts_mean']  = df.merge(gh_ts,    on=['geohash', 'ts_min'], how='left')['gh_ts_mean'].values
        df['gh_mean']     = df.merge(gh_mean,  on='geohash',             how='left')['gh_mean'].values
        df['gh_std']      = df.merge(gh_std,   on='geohash',             how='left')['gh_std'].values
        df['ts_mean']     = df.merge(ts_mean,  on='ts_min',              how='left')['ts_mean'].values
        df['geo4_mean']   = df.merge(geo4_mean,on='geo4',                how='left')['geo4_mean'].values

    return train, test

def fill_lag_with_fallback(train, test):
    """
    Where day-48 lag is missing, cascade through fallback aggregates.
    """
    for df in [train, test]:
        m = df['demand_d48'].isna()
        df.loc[m, 'demand_d48'] = df.loc[m, 'gh_ts_mean']
        m = df['demand_d48'].isna()
        df.loc[m, 'demand_d48'] = df.loc[m, 'gh_mean']
        m = df['demand_d48'].isna()
        df.loc[m, 'demand_d48'] = df.loc[m, 'geo4_mean']
        m = df['demand_d48'].isna()
        df.loc[m, 'demand_d48'] = df.loc[m, 'ts_mean']
    return train, test

FEATURE_COLS = [
    'lat', 'lon',
    'ts_min', 'hour', 'time_slot',
    'hour_sin', 'hour_cos', 'slot_sin', 'slot_cos', 'is_peak',
    'NumberofLanes', 'lanes_peak',
    'demand_d48', 'gh_ts_mean', 'gh_mean', 'gh_std', 'ts_mean', 'geo4_mean',
    'Temperature',
    'RoadType_enc', 'LargeVehicles_enc', 'Landmarks_enc', 'Weather_enc',
]