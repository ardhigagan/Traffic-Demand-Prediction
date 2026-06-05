import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────
TRAIN_PATH  = '../data/train.csv'
TEST_PATH   = '../data/test.csv'
OUTPUT_PATH = '../outputs/submission.csv'
# ─────────────────────────────────────────────────────────

BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'

def parse_ts(ts):
    h, m = ts.split(':')
    return int(h) * 60 + int(m)

def decode_gh(gh):
    lat, lon = 0., 0.
    lat_e, lon_e = 90., 180.
    is_lon = True
    for c in gh:
        cd = BASE32.index(c)
        for i in range(4, -1, -1):
            bit = (cd >> i) & 1
            if is_lon:
                lon_e /= 2
                lon = lon + lon_e if bit else lon - lon_e
            else:
                lat_e /= 2
                lat = lat + lat_e if bit else lat - lat_e
            is_lon = not is_lon
    return lat, lon

def add_base_features(df):
    df['ts_min']    = df['timestamp'].apply(parse_ts)
    c               = df['geohash'].map(decode_gh)
    df['lat']       = c.map(lambda x: x[0])
    df['lon']       = c.map(lambda x: x[1])
    df['geo3']      = df['geohash'].str[:3]
    df['geo4']      = df['geohash'].str[:4]
    df['geo5']      = df['geohash'].str[:5]
    df['hour']      = df['ts_min'] // 60
    df['time_slot'] = df['ts_min'] // 15
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['slot_sin']  = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['slot_cos']  = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['is_peak']   = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                       ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    df['lanes_peak']= df['NumberofLanes'] * df['is_peak']
    return df

def add_lag_features(train, test, t48):
    lag = t48[['geohash','ts_min','demand']].rename(columns={'demand':'d48'})
    g5  = t48.groupby(['geo5','ts_min'])['demand'].median().reset_index().rename(columns={'demand':'g5lag'})
    g4  = t48.groupby(['geo4','ts_min'])['demand'].median().reset_index().rename(columns={'demand':'g4lag'})
    g3  = t48.groupby(['geo3','ts_min'])['demand'].median().reset_index().rename(columns={'demand':'g3lag'})

    for df in [train, test]:
        df['demand_d48'] = df[['geohash','ts_min']].merge(lag, on=['geohash','ts_min'], how='left')['d48'].values
        df['g5lag']      = df[['geo5','ts_min']].merge(g5, on=['geo5','ts_min'], how='left')['g5lag'].values
        df['g4lag']      = df[['geo4','ts_min']].merge(g4, on=['geo4','ts_min'], how='left')['g4lag'].values
        df['g3lag']      = df[['geo3','ts_min']].merge(g3, on=['geo3','ts_min'], how='left')['g3lag'].values
        # Fill missing exact lag with neighborhood fallback
        for fb in ['g5lag', 'g4lag', 'g3lag']:
            m = df['demand_d48'].isna()
            df.loc[m, 'demand_d48'] = df.loc[m, fb]

def add_agg_stats(train, test, t48):
    def astats(grp, pfx):
        s = t48.groupby(grp)['demand'].agg(['mean','median','std']).reset_index()
        s.columns = grp + [f'{pfx}_{x}' for x in ['mean','median','std']]
        for df in [train, test]:
            tmp = df[grp].merge(s, on=grp, how='left')
            for c in s.columns:
                if c not in grp:
                    df[c] = tmp[c].values

    astats(['geohash'],        'gh')
    astats(['geo4'],           'g4')
    astats(['ts_min'],         'ts')
    astats(['hour'],           'hr')
    astats(['geo4', 'ts_min'], 'g4ts')
    astats(['geo3', 'ts_min'], 'g3ts')

def impute_temperature(train, test):
    tgh = train.groupby(['geohash','day'])['Temperature'].median().reset_index()
    tg4 = train.groupby(['geo4','day'])['Temperature'].median().reset_index()
    for df in [train, test]:
        m = df['Temperature'].isna()
        if m.sum():
            df.loc[m, 'Temperature'] = df.loc[m, ['geohash','day']].merge(
                tgh, on=['geohash','day'], how='left')['Temperature'].values
        m = df['Temperature'].isna()
        if m.sum():
            df.loc[m, 'Temperature'] = df.loc[m, ['geo4','day']].merge(
                tg4, on=['geo4','day'], how='left')['Temperature'].values
        df['Temperature'].fillna(train['Temperature'].median(), inplace=True)

def encode_cats(train, test):
    for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
        le = LabelEncoder()
        le.fit(pd.concat([train[col].fillna('Unknown'), test[col].fillna('Unknown')]))
        train[col+'_enc'] = le.transform(train[col].fillna('Unknown'))
        test[col+'_enc']  = le.transform(test[col].fillna('Unknown'))

FEAT_COLS = [
    'lat', 'lon', 'ts_min', 'hour', 'time_slot',
    'hour_sin', 'hour_cos', 'slot_sin', 'slot_cos',
    'is_peak', 'lanes_peak', 'NumberofLanes', 'Temperature',
    'demand_d48', 'g5lag', 'g4lag', 'g3lag',
    'gh_mean', 'gh_median', 'gh_std',
    'g4_mean', 'g4_median',
    'ts_mean', 'ts_median', 'ts_std',
    'hr_mean', 'hr_median',
    'g4ts_mean', 'g4ts_median', 'g4ts_std',
    'g3ts_mean', 'g3ts_median',
    'RoadType_enc', 'LargeVehicles_enc', 'Landmarks_enc', 'Weather_enc',
]

if __name__ == '__main__':
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    print("Adding base features...")
    for df in [train, test]:
        add_base_features(df)

    t48 = train[train.day == 48].copy()
    t48['geo3'] = t48.geohash.str[:3]
    t48['geo4'] = t48.geohash.str[:4]
    t48['geo5'] = t48.geohash.str[:5]
    t48['hour'] = t48.ts_min // 60

    print("Adding lag features...")
    add_lag_features(train, test, t48)

    print("Adding aggregate stats...")
    add_agg_stats(train, test, t48)

    print("Imputing temperature...")
    impute_temperature(train, test)

    print("Encoding categoricals...")
    encode_cats(train, test)

    fcols = [c for c in FEAT_COLS if c in train.columns]
    print(f"Total features: {len(fcols)}")

    meds = train[fcols].median()
    X_tr = train[fcols].fillna(meds)
    y_tr = train['demand'].values
    X_te = test[fcols].fillna(meds)

    # 5-fold CV
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_tr))
    tep = np.zeros(len(X_te))

    for fold, (ti, vi) in enumerate(kf.split(X_tr)):
        print(f"Training fold {fold+1}...")
        m = LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, num_leaves=255,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=0.5,
            random_state=42+fold, n_jobs=-1, verbose=-1
        )
        m.fit(X_tr.iloc[ti], y_tr[ti])
        oof[vi] = m.predict(X_tr.iloc[vi])
        tep    += m.predict(X_te) / 5
        print(f"  Fold {fold+1} R²×100: {100*r2_score(y_tr[vi], oof[vi]):.2f}")

    print(f"\nOOF R²×100: {100*r2_score(y_tr, oof):.2f}")

    sub = pd.DataFrame({'Index': test['Index'], 'demand': np.clip(tep, 0, 1)})
    sub.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}  shape: {sub.shape}")