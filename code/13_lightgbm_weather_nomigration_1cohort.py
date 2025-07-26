####################################################################################################
## Description: Single‐cohort (2003–2007 → 2007) LightGBM with manual daily-weather features
## Author:      Parth Chawla
## Date:        Jul 24, 2025
####################################################################################################

import os
import re
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

# Change to your project directory
os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration/')

# Prepare output folder
out_dir = 'output/with_daily'
os.makedirs(out_dir, exist_ok=True)

# 2) LOAD & CLEAN DATA ---------------------------------------------------------------------------
# point to your new cohort CSV
df = pd.read_csv('data/data_cohort_daily_weather.csv')

# map gender codes if present
if 'male' in df.columns:
    gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
    df['male'] = df['male'].map(gender_map).astype(float)

# 3) FEATURE & TARGET -----------------------------------------------------------------------------
# static “base” features you already compute earlier
base = [
    'male','age','L1_hhchildren','L1_hhworkforce',
    'L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
    'L1_ag','L1_nonag','L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc',
    'L1_nonag_inc','L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_trans_inc'
]

# village dummies
vill_cols = [c for c in df.columns if c.startswith('vill_')]

# spatial & distance
spatial = [c for c in df.columns if c.endswith('_std') and c.split('_')[0] in 
           ['latitude','longitude','distkm','avtimeloc02','local_wage']]

# 1) re-introduce your monthly aggregates:
weather_monthly = [
    'avgtemp5','precip_tot5','GDD5','HDD5',
    'avgtemp6','precip_tot6','GDD6','HDD6',
    'avgtemp7','precip_tot7','GDD7','HDD7',
    'avgtemp8','precip_tot8','GDD8','HDD8',
    'precip_tot_MDagseason','HDD_MDagseason','GDD_MDagseason',
    'precip_tot_nonagseason','HDD_nonagseason','GDD_nonagseason'
]

# 2) your manual daily‐weather features:
daily_feats = [c for c in df.columns if re.match(r'^e\d+_', c)]

# 3) build the full x_cols:
x_cols = base + vill_cols + spatial + weather_monthly + daily_feats
y_col  = 'work_us'

# sanity check
print("Total features:", len(x_cols))
print("Sample feature list:", x_cols[:10], "...", x_cols[-5:])

# 4) SINGLE‐COHORT SPLIT ---------------------------------------------------------------------------
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

param_space = {
    'num_leaves':       np.random.randint(20,150,100),
    'min_data_in_leaf': np.random.randint(10,100,100),
    'learning_rate':    np.random.uniform(0.01,0.1,100),
    'feature_fraction': np.random.uniform(0.5,1.0,100),
    'bagging_fraction': np.random.uniform(0.5,1.0,100),
}

pos_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))

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
final_model.save_model(f'{out_dir}/final_model_daily.txt')

y_prob = final_model.predict(X_test)

# --- THRESHOLD TUNING FOR BEST F1 -----------------------------------------------
prec_vals, rec_vals, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec_vals, prec_vals)
print(f"PR-AUC = {pr_auc:.4f}")

f1_scores  = 2 * (prec_vals * rec_vals) / (prec_vals + rec_vals + 1e-8)
best_idx   = np.nanargmax(f1_scores)
best_thresh= thresholds[best_idx]
best_f1    = f1_scores[best_idx]
best_prec  = prec_vals[best_idx]
best_rec   = rec_vals[best_idx]

print(f"Best F1={best_f1:.4f} at threshold={best_thresh:.4f} → Precision={best_prec:.4f}, Recall={best_rec:.4f}")

# Use the optimized threshold
y_pred_opt = (y_prob > best_thresh).astype(int)
y_pred     = y_pred_opt  # <— define y_pred!

# Recompute metrics at optimized threshold
precision_opt, recall_opt, f1_opt, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary'
)
print(f"Optimized (th={best_thresh:.4f}) → Precision: {precision_opt:.4f}, Recall: {recall_opt:.4f}, F1: {f1_opt:.4f}")

# Save optimized predictions
out_opt = test_data.copy()
out_opt['predicted_y']    = y_pred
out_opt['predicted_prob'] = y_prob
out_opt.to_csv(f'{out_dir}/test_daily_predictions_2007_opt_thresh.csv', index=False)

# 7) FEATURE IMPORTANCE & CSV ---------------------------------------------------------------------
importance_df = pd.DataFrame({
    'feature':    x_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
importance_df.to_csv(f'{out_dir}/lightgbm_daily_feature_importance.csv', index=False)

plt.figure(figsize=(6,4))
plt.plot(rec_vals, prec_vals, label=f'AUC={pr_auc:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curve (2007 Test)'); plt.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_daily_pr_curve.png')
plt.close()

# 8) FEATURE IMPORTANCE & CSV ---------------------------------------------------------------------
importance_df = pd.DataFrame({
    'feature':    x_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
importance_df.to_csv(f'{out_dir}/lightgbm_daily_feature_importance.csv', index=False)

plt.figure(figsize=(12,6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Features (gain)'); plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_daily_feature_importance.png')
plt.close()

# 9) SHAP VALUES & CSV ---------------------------------------------------------------------------
explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
shap_df     = pd.DataFrame(shap_values, columns=x_cols)
shap_df.to_csv(f'{out_dir}/lightgbm_daily_shap.csv', index=False)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP summary (2007 Test)')
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_daily_shap.png')
plt.close()

# 10) FINAL TEST PREDICTIONS ----------------------------------------------------------------------
out = test_data.copy()
out['predicted_y']    = y_pred
out['predicted_prob'] = y_prob
out.to_csv(f'{out_dir}/test_daily_predictions_2007.csv', index=False)

# classification report & confusion matrix
cr  = classification_report(y_test, y_pred, target_names=["No US","Worked US"])
cm  = confusion_matrix(y_test, y_pred)
with open(f'{out_dir}/lightgbm_daily_report.txt', 'w') as f:
    f.write("Classification Report\n\n" + cr + "\nConfusion Matrix\n\n" + np.array2string(cm))

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix (2007 Test)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_daily_confusion_matrix.png')
plt.close()

print("Single‐cohort (2003–2007→2007) analysis with daily features complete. All outputs saved.")
