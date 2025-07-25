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

# Change to project directory as needed
os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration/')

# Prepare output folder
out_dir = 'output/with_weather'
os.makedirs(out_dir, exist_ok=True)

# 2) LOAD DATA
# Use weather-enhanced cohort file
df = pd.read_csv('data/data_cohort_analysis_weather.csv')
# Map gender to 0/1
gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
df['male'] = df['male'].map(gender_map).astype(float)

# 3) DEFINE FEATURES & TARGET
vill_cols = [c for c in df.columns if c.startswith('vill_')]
# Base lagged features
base_cols = [
    'male','age','L1_hhchildren','L1_hhworkforce',
    'L1_yrs_in_mx_cum','L1_yrs_in_us_cum','L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
    'L1_hh_yrs_in_us_cum','L1_hh_migrant',
    'L1_work_us','L1_work_mx','L1_ag','L1_nonag',
    'L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc','L1_nonag_inc',
    'L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_rem_mx','L1_rem_us','L1_trans_inc'
]
# Spatial & distance\ spatial_cols = ['latitude_std','longitude_std','distkm_std','avtimeloc02_std','local_wage_std']
# Weather: monthly lags
weather_monthly = [
    'avgtemp5','precip_tot5','GDD5','HDD5',
    'avgtemp6','precip_tot6','GDD6','HDD6',
    'avgtemp7','precip_tot7','GDD7','HDD7',
    'avgtemp8','precip_tot8','GDD8','HDD8'
]
# Weather: seasonal aggregates
weather_seasonal = [
    'precip_tot_MDagseason','HDD_MDagseason','GDD_MDagseason',
    'precip_tot_nonagseason','HDD_nonagseason','GDD_nonagseason'
]
# Combine feature list
x_cols = base_cols + vill_cols # NO WEATHER OR SPATIAL
y_col = 'work_us'

# 4) ARDL-STYLE SPLIT (train on <2007, test =2007)
train_df = df[df['year'] < 2007]
test_df  = df[df['year'] == 2007]

X_train = train_df[x_cols]
y_train = train_df[y_col].fillna(0)
X_test  = test_df[x_cols]
y_test  = test_df[y_col].fillna(0)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}")

# 5) PIPELINE: IMPUTER + LOGISTIC REGRESSION
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
y_prob = pipeline.predict_proba(X_test)[:,1]
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 8) FEATURE IMPORTANCE (|coef|) & CSV
coef = pipeline.named_steps['clf'].coef_[0]
importance_df = pd.DataFrame({'feature': x_cols,'importance': np.abs(coef)})
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df.to_csv(f'{out_dir}/logistic_feature_importance.csv', index=False)

plt.figure(figsize=(12,6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Most Important Features (|coef|)')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_feature_importance.png')
plt.close()

# 9) SHAP SUMMARY & CSV
X_test_imp = pipeline.named_steps['imputer'].transform(X_test)
explainer = shap.LinearExplainer(pipeline.named_steps['clf'], pipeline.named_steps['imputer'].transform(X_train))
shap_values = explainer.shap_values(X_test_imp)
shap_df = pd.DataFrame(shap_values, columns=x_cols)
shap_df.to_csv(f'{out_dir}/logistic_shap.csv', index=False)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test_imp, feature_names=x_cols, show=False)
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_shap.png')
plt.close()

# 10) SAVE OUTPUTS
report = classification_report(y_test, y_pred, target_names=["No US","Worked US"])
cm = confusion_matrix(y_test, y_pred)
with open(f'{out_dir}/logistic_report.txt', 'w') as f:
    f.write('Classification Report\n\n'+report+"\nConfusion Matrix\n\n"+np.array2string(cm))

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix (ARDL-style split)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{out_dir}/logistic_confusion_matrix.png')
plt.close()

print("ARDL-style logistic regression complete. Outputs saved.")
