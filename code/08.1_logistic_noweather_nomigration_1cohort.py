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

x_cols = base + vill_cols

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

# 5) PIPELINE: IMPUTER + LOGISTIC
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', LogisticRegression(
        solver='liblinear',
        random_state=SEED,
        class_weight='balanced',
        max_iter=500
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
importance_df = (
    pd.DataFrame({'feature': x_cols, 'importance': np.abs(coef)})
      .sort_values('importance', ascending=False)
)
importance_df.to_csv(f'{out_dir}/logistic_nm1_feature_importance.csv', index=False)

plt.figure(figsize=(12,6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Features (|coef|)')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_nm1_feature_importance.png')
plt.close()

# 9) SHAP SUMMARY & CSV
X_test_imp = pipeline.named_steps['imputer'].transform(X_test)
explainer   = shap.LinearExplainer(
    pipeline.named_steps['clf'],
    pipeline.named_steps['imputer'].transform(X_train)
)
shap_values = explainer.shap_values(X_test_imp)

# save the raw SHAP values
shap_df = pd.DataFrame(shap_values, columns=x_cols)
shap_df.to_csv(f'{out_dir}/logistic_nm1_shap.csv', index=False)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test_imp, feature_names=x_cols, show=False)
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_nm1_shap.png')
plt.close()

# 10) FINAL REPORT & CONFUSION MATRIX
report = classification_report(y_test, y_pred, target_names=["No US","Worked US"])
cm     = confusion_matrix(y_test, y_pred)
with open(f'{out_dir}/logistic_nm1_report.txt','w') as f:
    f.write("Classification Report\n\n" + report + "\nConfusion Matrix\n\n" + np.array2string(cm))

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix (2007 Test)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_nm1_confusion_matrix.png')
plt.close()

print("Single‐cohort logistic regression complete. Outputs saved.")
