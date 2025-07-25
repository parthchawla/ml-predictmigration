####################################################################################################
## Description: Single‐cohort (2003–2007 → 2007) LightGBM version with weather & spatial data
## Author:      Parth Chawla
## Date:        Jun 21, 2025
####################################################################################################

import os
import numpy as np
import pandas as pd
import random
import lightgbm as lgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, precision_recall_fscore_support,
    precision_recall_curve, auc, classification_report,
    confusion_matrix
)

# 1) SETUP ----------------------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Change to project directory as needed
os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration/')

# Prepare output folder
out_dir = 'output/with_weather'
os.makedirs(out_dir, exist_ok=True)

# 2) LOAD & CLEAN DATA ---------------------------------------------------------------------------
df = pd.read_csv('data/data_cohort_analysis_weather.csv')
# map gender codes
gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
df['male'] = df['male'].map(gender_map).astype(float)

# 3) FEATURE & TARGET -----------------------------------------------------------------------------
vill_cols = [c for c in df.columns if c.startswith('vill_')]
# Spatial & distance
spatial = ['latitude_std','longitude_std','distkm_std','avtimeloc02_std','local_wage_std']
# Weather variables
tmp = ['avgtemp5','precip_tot5','GDD5','HDD5',
       'avgtemp6','precip_tot6','GDD6','HDD6',
       'avgtemp7','precip_tot7','GDD7','HDD7',
       'avgtemp8','precip_tot8','GDD8','HDD8']
season = ['precip_tot_MDagseason','HDD_MDagseason','GDD_MDagseason',
          'precip_tot_nonagseason','HDD_nonagseason','GDD_nonagseason']
weather_cols = tmp + season

# Base features
base = ['male','age','L1_hhchildren','L1_hhworkforce',
        'L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
        'L1_ag','L1_nonag','L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc',
        'L1_nonag_inc','L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_trans_inc']

x_cols = base + vill_cols + weather_cols

y_col = 'work_us'

# 4) SINGLE‐COHORT SPLIT ---------------------------------------------------------------------------
# 2003–2006 as train, 2007 as test
train_data = df[df['cohort']=='2003-2007 Pre-Period']
test_data  = df[df['cohort']=='2003-2007 Outcome Period']

X_train = train_data[x_cols]
y_train = train_data[y_col].fillna(0)

X_test  = test_data[x_cols]
y_test  = test_data[y_col].fillna(0)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 5) HYPERPARAMETER TUNING (RANDOM SEARCH) --------------------------------------------------------
best_precision = 0.0
best_params    = None
best_model     = None

# define search space
param_space = {
    'num_leaves':       np.random.randint(20,150,100),
    'min_data_in_leaf': np.random.randint(10,100,100),
    'learning_rate':    np.random.uniform(0.01,0.1,100),
    'feature_fraction': np.random.uniform(0.5,1.0,100),
    'bagging_fraction': np.random.uniform(0.5,1.0,100),
}

# class‐imbalance weight
pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

for j in range(100):
    if j % 10 == 0:
        print(f" Tuning iteration {j+1}/100")
    params = {
        'objective':        'binary',
        'metric':           'binary_logloss',
        'num_leaves':       int(param_space['num_leaves'][j]),
        'min_data_in_leaf': int(param_space['min_data_in_leaf'][j]),
        'learning_rate':    float(param_space['learning_rate'][j]),
        'feature_fraction': float(param_space['feature_fraction'][j]),
        'bagging_fraction': float(param_space['bagging_fraction'][j]),
        'scale_pos_weight': pos_weight,
        'seed':             SEED,
        'verbose':          -1
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_test,  label=y_test, reference=dtrain)
    mdl = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    y_val_pred = mdl.predict(X_test, num_iteration=mdl.best_iteration)
    prec = precision_score(y_test, (y_val_pred>0.5).astype(int))
    if prec > best_precision:
        best_precision = prec
        best_params    = params
        best_model     = mdl

print(f"\nBest validation precision = {best_precision:.4f}")
print("Best params:", best_params)

# 6) FINAL EVALUATION ON 2007 ----------------------------------------------------------------------
final_model = best_model
final_model.save_model(f'{out_dir}/final_model1_nm1_w.txt')

# predictions
y_prob   = final_model.predict(X_test)
y_pred   = (y_prob > 0.5).astype(int)

# classification metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"2007 Test → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 7) PRECISION–RECALL CURVE & CSV -----------------------------------------------------------------
prec_vals, rec_vals, thr = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec_vals, prec_vals)
pd.DataFrame({
    'precision': prec_vals,
    'recall':    rec_vals,
    'threshold': np.append(thr, np.nan)
}).to_csv(f'{out_dir}/lightgbm_nm1_w_precision_recall_curve.csv', index=False)

plt.figure(figsize=(6,4))
plt.plot(rec_vals, prec_vals, label=f'AUC={pr_auc:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curve (2007 Test)'); plt.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_nm1_w_precision_recall_curve.png')
plt.close()

# 8) FEATURE IMPORTANCE & CSV ---------------------------------------------------------------------
importance_df = pd.DataFrame({
    'feature':    x_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
importance_df.to_csv(f'{out_dir}/lightgbm_nm1_w_feature_importance.csv', index=False)

plt.figure(figsize=(12,6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Features (gain)'); plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_nm1_w_feature_importance.png')
plt.close()

# 9) SHAP VALUES & CSV ---------------------------------------------------------------------------
explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
shap_df     = pd.DataFrame(shap_values, columns=x_cols)
shap_df.to_csv(f'{out_dir}/lightgbm_nm1_w_shap.csv', index=False)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP summary (2007 Test)')
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_nm1_w_shap.png')
plt.close()

# 10) FINAL TEST PREDICTIONS ----------------------------------------------------------------------
out = test_data.copy()
out['predicted_y']    = y_pred
out['predicted_prob'] = y_prob
out.to_csv(f'{out_dir}/test_nm1_w_predictions_2007.csv', index=False)

# classification report & confusion matrix
cr  = classification_report(y_test, y_pred, target_names=["No US","Worked US"])
cm  = confusion_matrix(y_test, y_pred)

with open(f'{out_dir}/lightgbm_nm1_w_report.txt','w') as f:
    f.write("Classification Report\n\n"+cr+"\nConfusion Matrix\n\n"+np.array2string(cm))

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix (2007 Test)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_nm1_w_confusion_matrix.png')
plt.close()

print("Single‐cohort analysis complete. All outputs saved.")
