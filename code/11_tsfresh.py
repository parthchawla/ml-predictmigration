import os
import pandas as pd
import numpy as np
import re
from scipy.stats import skew, kurtosis

# 1) cwd → project root
os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration')
print("CWD:", os.getcwd())

# 2) load raw data (wide, one row per person‐year)
df = pd.read_stata('data/MexMigData_daily_weather.dta')
print("Loaded shape:", df.shape)

# 3) identify daily‐weather columns
daily_cols = [c for c in df.columns if re.match(r'^v\d+_e\d+_m\d+', c)]
print(f"Found {len(daily_cols)} daily cols")

# 4) static part (everything else)
static_cols = [c for c in df.columns if c not in daily_cols]
static      = df[static_cols]

# 5) build features per element code
feature_frames = []
for e in [1,2,3,5,18]:
    pattern = f"_e{e}_"
    cols = [c for c in daily_cols if pattern in c]
    print(f" Element e{e}: {len(cols)} cols")
    # select and clean
    arr = df[cols].replace(-99999, np.nan).astype(float)
    # compute stats row‐wise
    feats = pd.DataFrame({
        f"e{e}_mean":    arr.mean(axis=1),
        f"e{e}_std":     arr.std(axis=1),
        f"e{e}_skew":    arr.apply(lambda row: skew(row.dropna()), axis=1),
        f"e{e}_kurt":    arr.apply(lambda row: kurtosis(row.dropna()), axis=1),
        f"e{e}_q25":     arr.quantile(0.25, axis=1),
        f"e{e}_q50":     arr.quantile(0.50, axis=1),
        f"e{e}_q75":     arr.quantile(0.75, axis=1),
        f"e{e}_min":     arr.min(axis=1),
        f"e{e}_max":     arr.max(axis=1),
        f"e{e}_range":   arr.max(axis=1) - arr.min(axis=1),
    })
    feature_frames.append(feats)

# 6) concat all features
all_feats = pd.concat(feature_frames, axis=1)
print("All_feats shape:", all_feats.shape)

# 7) merge with static
out_df = pd.concat([static.reset_index(drop=True), all_feats.reset_index(drop=True)], axis=1)
print("Final shape:", out_df.shape)

# 8) save
out_path = 'data/MexMigData_manual_features.parquet'
out_df.to_parquet(out_path, index=False)
size_mb = os.path.getsize(out_path)/1e6
print(f"Saved {out_path} ({size_mb:.1f} MB)")
