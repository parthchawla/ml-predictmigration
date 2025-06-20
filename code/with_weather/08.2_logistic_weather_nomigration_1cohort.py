####################################################################################################
## Description: Single‐cohort logistic regression (2003–2007 → 2007) with weather & spatial data
## Author: Parth Chawla
## Date: Jun 21, 2025
####################################################################################################

import os
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# 1) SETUP
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Project directory and output
os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration/')
out_dir = 'output/with_weather'
os.makedirs(out_dir, exist_ok=True)

# 2) LOAD DATA
# Use updated cohort-weather file
df = pd.read_csv('data/data_cohort_analysis_weather.csv')
# Map gender to 0/1
gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
df['male'] = df['male'].map(gender_map).astype(float)

# 3) FEATURES & TARGET
# Village dummies
vill_cols = [c for c in df.columns if c.startswith('vill_')]
# Spatial features
spatial_cols = ['latitude_std','longitude_std','distkm_std','avtimeloc02_std','local_wage_std']
# Weather features
monthly = [
    'avgtemp5','precip_tot5','GDD5','HDD5',
    'avgtemp6','precip_tot6','GDD6','HDD6',
    'avgtemp7','precip_tot7','GDD7','HDD7',
    'avgtemp8','precip_tot8','GDD8','HDD8'
]
seasonal = [
    'precip_tot_MDagseason','HDD_MDagseason','GDD_MDagseason',
    'precip_tot_nonagseason','HDD_nonagseason','GDD_nonagseason'
]
weather_cols = monthly + seasonal

# Base lagged & control features
base_cols = [
    'male','age','L1_hhchildren','L1_hhworkforce',
    'L1_yrs_in_mx_cum','L1_yrs_in_us_cum','L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
    'L1_ag','L1_nonag',
    'L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc','L1_nonag_inc',
    'L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_trans_inc'
]
# Final feature list\ nx_cols = base_cols + spatial_cols + weather_cols + vill_cols
y_col = 'work_us'

# 4) SINGLE-COHORT SPLIT
# Train: 2003-2007 Pre-Period; Test: 2003-2007 Outcome Period
t = df['cohort']=='2003-2007 Pre-Period'
train_df = df[t]
t = df['cohort']=='2003-2007 Outcome Period'
test_df  = df[t]
X_train = train_df[nx_cols]
y_train = train_df[y_col].fillna(0)
X_test  = test_df[nx_cols]
y_test  = test_df[y_col].fillna(0)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 5) PIPELINE: IMPUTER + LOGISTIC
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', LogisticRegression(
        solver='liblinear', random_state=SEED,
        class_weight='balanced', max_iter=500
    ))
])

# 6) TRAIN
pipeline.fit(X_train, y_train)

# 7) PREDICT & EVALUATE
y_pred = pipeline.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary'
)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 8) FEATURE IMPORTANCE (|coef|) & CSV
coef = pipeline.named_steps['clf'].coef_[0]
imp_df = (
    pd.DataFrame({'feature': nx_cols, 'importance': np.abs(coef)})
      .sort_values('importance', ascending=False)
)
imp_df.to_csv(f'{out_dir}/logistic_nm1_feature_importance.csv', index=False)
plt.figure(figsize=(12,6))
sns.barplot(data=imp_df.head(20), x='importance', y='feature')
plt.title('Top 20 Features (|coef|)')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_nm1_feature_importance.png')
plt.close()

# 9) SHAP SUMMARY & CSV
X_test_imp = pipeline.named_steps['imputer'].transform(X_test)
explainer = shap.LinearExplainer(
    pipeline.named_steps['clf'],
    pipeline.named_steps['imputer'].transform(X_train)
)
shap_values = explainer.shap_values(X_test_imp)
# save SHAP values
shap_df = pd.DataFrame(shap_values, columns=nx_cols)
shap_df.to_csv(f'{out_dir}/logistic_nm1_shap.csv', index=False)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test_imp, feature_names=nx_cols, show=False)
plt.title('SHAP summary (2007 Test)')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_nm1_shap.png')
plt.close()

# 10) FINAL REPORT & CONFUSION MATRIX
report = classification_report(y_test, y_pred, target_names=["No US","Worked US"])
cm     = confusion_matrix(y_test, y_pred)
with open(f'{out_dir}/logistic_nm1_report.txt','w') as f:
    f.write('Classification Report\n\n'+report+"\nConfusion Matrix\n"+np.array2string(cm))
plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix (2007 Test)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_nm1_confusion_matrix.png')
plt.close()

print("Single‐cohort logistic regression (2003–2007→2007) complete.")
