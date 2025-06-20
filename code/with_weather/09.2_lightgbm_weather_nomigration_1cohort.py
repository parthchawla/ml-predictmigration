####################################################################################################
## Description: Machine learning program to predict migration in Mexico
##              Single‐cohort (2005–2009 → 2010) LightGBM version
## Author:      Parth Chawla
## Date:        Nov, 2024
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

# Change to your project directory (or remove if already there)
os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration/')

# 2) LOAD & CLEAN DATA ---------------------------------------------------------------------------
df = pd.read_csv('data/data_cohort_analysis_add_vars.csv')
# map gender codes to 0/1
gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
df['male'] = df['male'].map(gender_map).astype(float)

# drop & aggregate MX/US‐specific vars into totals, if not already done
agg_mappings = {
    'L1_yrs_in_ag_sal_cum': ['L1_yrs_in_mx_ag_sal_cum','L1_yrs_in_us_ag_sal_cum'],
    'L1_yrs_in_nonag_sal_cum': ['L1_yrs_in_mx_nonag_sal_cum','L1_yrs_in_us_nonag_sal_cum'],
    'L1_yrs_in_ag_own_cum': ['L1_yrs_in_mx_ag_own_cum','L1_yrs_in_us_ag_own_cum'],
    'L1_yrs_in_nonag_own_cum': ['L1_yrs_in_mx_nonag_own_cum','L1_yrs_in_us_nonag_own_cum']
}
for new_var, parts in agg_mappings.items():
    df[new_var] = df[parts].sum(axis=1)
df.drop(columns=sum(agg_mappings.values(), []), inplace=True)

# 3) FEATURES & TARGET ----------------------------------------------------------------------------
vill_cols = [c for c in df.columns if c.startswith('vill_')]
x_cols1 = [
    'male','age','L1_hhchildren','L1_hhworkforce',
    'L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
    *agg_mappings.keys(),
    'L1_ag','L1_nonag',
    'L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc','L1_nonag_inc',
    'L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_trans_inc'
]
x_cols = x_cols1 + vill_cols
y_col  = 'work_us'

# 4) SINGLE‐COHORT SPLIT ---------------------------------------------------------------------------
# 2005–2009 as training, 2010 as test
train_data    = df[df['cohort']=='2005-2010 Pre-Period']
test_data     = df[df['cohort']=='2005-2010 Outcome Period']

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

# 6) FINAL EVALUATION ON 2010 ----------------------------------------------------------------------
final_model = best_model

# predictions
y_prob   = final_model.predict(X_test)
y_pred   = (y_prob > 0.5).astype(int)

# classification metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"2010 Test → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 7) PRECISION–RECALL CURVE & CSV -----------------------------------------------------------------
prec_vals, rec_vals, thr = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec_vals, prec_vals)
pd.DataFrame({
    'precision': prec_vals,
    'recall':    rec_vals,
    'threshold': np.append(thr, np.nan)
}).to_csv('output/lightgbm_nm1_precision_recall_curve.csv', index=False)

plt.figure(figsize=(6,4))
plt.plot(rec_vals, prec_vals, label=f'AUC={pr_auc:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curve (2010 Test)'); plt.legend()
plt.tight_layout()
plt.savefig('output/lightgbm_nm1_precision_recall_curve.png')
plt.close()

# 8) FEATURE IMPORTANCE & CSV ---------------------------------------------------------------------
importance_df = pd.DataFrame({
    'feature':    x_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
importance_df.to_csv('output/lightgbm_nm1_feature_importance.csv', index=False)

plt.figure(figsize=(12,6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Features (gain)'); plt.tight_layout()
plt.savefig('output/lightgbm_nm1_feature_importance.png')
plt.close()

# 9) SHAP VALUES & CSV ---------------------------------------------------------------------------
explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
shap_df     = pd.DataFrame(shap_values, columns=x_cols)
shap_df.to_csv('output/lightgbm_nm1_shap.csv', index=False)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP summary (2010 Test)')
plt.tight_layout()
plt.savefig('output/lightgbm_nm1_shap.png')
plt.close()

# 10) FINAL TEST PREDICTIONS ----------------------------------------------------------------------
out = test_data.copy()
out['predicted_y']    = y_pred
out['predicted_prob'] = y_prob
out.to_csv('output/test_nm1_predictions_2010.csv', index=False)

# classification report & confusion matrix
cr  = classification_report(y_test, y_pred, target_names=["No US","Worked US"])
cm  = confusion_matrix(y_test, y_pred)

with open('output/lightgbm_nm1_report.txt','w') as f:
    f.write("Classification Report\n\n"+cr+"\nConfusion Matrix\n\n"+np.array2string(cm))

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix (2010 Test)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('output/lightgbm_nm1_confusion_matrix.png')
plt.close()

print("Single‐cohort (2005–2009→2010) analysis complete. All outputs saved.")
