import os
import numpy as np
import pandas as pd
import lightgbm as lgb

# 1) SETUP & LOAD MODEL --------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# paths
path    = '/Users/parthchawla1/GitHub/ml-predictmigration/'
out_dir = os.path.join(path, 'output', 'with_weather')
model_file = os.path.join(out_dir, 'final_model1_nm1_w.txt')

# load the trained LightGBM model
model = lgb.Booster(model_file=model_file)

# 2) LOAD & PREPARE TEST DATA --------------------------------------------------
df = pd.read_csv(os.path.join(path, 'data', 'data_cohort_analysis_weather.csv'))

# remap gender strings to numeric
gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
df['male'] = df['male'].map(gender_map).astype(float)

# identify feature columns exactly as in training
vill_cols = [c for c in df.columns if c.startswith('vill_')]
spatial   = ['latitude_std','longitude_std','distkm_std','avtimeloc02_std','local_wage_std']
tmp       = ['avgtemp5','precip_tot5','GDD5','HDD5',
             'avgtemp6','precip_tot6','GDD6','HDD6',
             'avgtemp7','precip_tot7','GDD7','HDD7',
             'avgtemp8','precip_tot8','GDD8','HDD8']
season    = ['precip_tot_MDagseason','HDD_MDagseason','GDD_MDagseason',
             'precip_tot_nonagseason','HDD_nonagseason','GDD_nonagseason']
weather_cols = tmp + season

base = ['male','age','L1_hhchildren','L1_hhworkforce',
        'L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
        'L1_ag','L1_nonag','L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc',
        'L1_nonag_inc','L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_trans_inc']

x_cols = base + vill_cols + weather_cols
y_col  = 'work_us'

# filter to 2007 outcome cohort
test_data = df[df['cohort']=='2003-2007 Outcome Period'].copy()
X_test    = test_data[x_cols].fillna(0)      # fill any NaNs
y_test    = test_data[y_col].fillna(0).astype(int)

# identify all the weather columns to shock
temp_cols   = [c for c in weather_cols if c.startswith('avgtemp')]
precip_cols = [c for c in weather_cols if 'precip' in c]
gdd_cols    = [c for c in weather_cols if c.startswith('GDD')]
hdd_cols    = [c for c in weather_cols if c.startswith('HDD')]

all_weather_cols = temp_cols + precip_cols + gdd_cols + hdd_cols

# 3) SIMULATE SCENARIOS & PREDICT ----------------------------------------------
scenarios = {
    'original':           1.00,
    'temp_plus_10pct':    1.10,
    'precip_plus_10pct':  1.10,
    'gdd_plus_10pct':     1.10,
    'hdd_plus_10pct':     1.10,
    'all_weather_plus_10pct': 1.10
}

for name, factor in scenarios.items():
    X_mod = X_test.copy()
    
    if name == 'temp_plus_10pct':
        X_mod[temp_cols] *= factor
    elif name == 'precip_plus_10pct':
        X_mod[precip_cols] *= factor
    elif name == 'gdd_plus_10pct':
        X_mod[gdd_cols] *= factor
    elif name == 'hdd_plus_10pct':
        X_mod[hdd_cols] *= factor
    elif name == 'all_weather_plus_10pct':
        X_mod[all_weather_cols] *= factor
    # else: original, do nothing

    probs = model.predict(X_mod)
    preds = (probs > 0.5).astype(int)
    
    test_data[f'pred_prob_{name}'] = probs
    test_data[f'pred_{name}']      = preds

# 4) SAVE RESULTS --------------------------------------------------------------
output_csv = os.path.join(out_dir, 'test_predictions_2007_temp_shock.csv')
test_data.to_csv(output_csv, index=False)

print(f"Saved counterfactual predictions to {output_csv}")
