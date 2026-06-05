import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def parse_timestamp(ts):
    h, m = ts.split(':')
    return int(h) * 60 + int(m)

def decode_geohash(gh):
    BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat, lon = 0.0, 0.0
    lat_err, lon_err = 90.0, 180.0
    is_lon = True
    for c in gh:
        cd = BASE32.index(c)
        for i in range(4, -1, -1):
            bit = (cd >> i) & 1
            if is_lon:
                lon_err /= 2
                lon = lon + lon_err if bit else lon - lon_err
            else:
                lat_err /= 2
                lat = lat + lat_err if bit else lat - lat_err
            is_lon = not is_lon
    return lat, lon

def load_and_preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    for df in [train, test]:
        # Parse timestamp
        df['ts_min'] = df['timestamp'].apply(parse_timestamp)

        # Geohash → lat/lon
        coords = df['geohash'].map(decode_geohash)
        df['lat'] = coords.map(lambda x: x[0])
        df['lon'] = coords.map(lambda x: x[1])

        # Geohash prefix hierarchy
        df['geo3'] = df['geohash'].str[:3]
        df['geo4'] = df['geohash'].str[:4]
        df['geo5'] = df['geohash'].str[:5]

    return train, test

def encode_categoricals(train, test):
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([
            train[col].fillna('Unknown'),
            test[col].fillna('Unknown')
        ])
        le.fit(combined)
        train[col + '_enc'] = le.transform(train[col].fillna('Unknown'))
        test[col + '_enc']  = le.transform(test[col].fillna('Unknown'))
    return train, test