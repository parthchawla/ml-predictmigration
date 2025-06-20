####################################################################################################
## Description: Machine learning program to predict migration in Mexico with weather & spatial data
## Author: Parth Chawla
## Date: Nov, 2024 (updated Jun 21, 2025)
####################################################################################################

import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, precision_recall_fscore_support
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap

# Set a random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
lgb_params = {'seed': SEED}

# Change working directory
path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

# Create output directory for weather runs
out_dir = 'output/with_weather'
os.makedirs(out_dir, exist_ok=True)

# Load prepared cohort data with weather & spatial features
df = pd.read_csv('data/data_cohort_analysis_weather.csv')
vill_cols = [col for col in df.columns if col.startswith('vill_')]

# Map gender to 0/1
gender_map = {'M':1,'m':1,'Male':1,'male':1,'F':0,'f':0,'Female':0,'female':0}
df['male'] = df['male'].map(gender_map).astype(float)

# Define feature columns (lagged, dummies, spatial, and weather)
x_cols1 = [
    'male', 'age',
    'L1_hhchildren', 'L1_hhworkforce',
    'L1_yrs_in_mx_cum', 'L1_yrs_in_us_cum', 'L1_yrs_in_ag_cum', 'L1_yrs_in_nonag_cum',
    'L1_hh_yrs_in_us_cum', 'L1_hh_migrant',
    'L1_ag_inc', 'L1_asset_inc', 'L1_farmlab_inc', 'L1_liv_inc', 'L1_nonag_inc',
    'L1_plot_inc_renta_ag', 'L1_plot_inc_renta_nonag', 'L1_rec_inc', 'L1_rem_mx', 'L1_rem_us', 'L1_trans_inc',
    'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag',
    # spatial & distance
    'latitude_std', 'longitude_std', 'distkm_std', 'avtimeloc02_std', 'local_wage_std',
    # weather: monthly lags
    'avgtemp5', 'precip_tot5', 'GDD5', 'HDD5',
    'avgtemp6', 'precip_tot6', 'GDD6', 'HDD6',
    'avgtemp7', 'precip_tot7', 'GDD7', 'HDD7',
    'avgtemp8', 'precip_tot8', 'GDD8', 'HDD8',
    # weather: seasonal aggregates
    'precip_tot_MDagseason', 'HDD_MDagseason', 'GDD_MDagseason',
    'precip_tot_nonagseason', 'HDD_nonagseason', 'GDD_nonagseason'
]
x_cols = x_cols1 + vill_cols
y_col = 'work_us'

# Define cohorts for 1980-2007
pre_periods = [
    '1980-1984 Pre-Period','1985-1989 Pre-Period','1990-1994 Pre-Period',
    '1995-1999 Pre-Period','2000-2004 Pre-Period','2003-2007 Pre-Period'
]
outcome_periods = [
    '1980-1984 Outcome Period','1985-1989 Outcome Period','1990-1994 Outcome Period',
    '1995-1999 Outcome Period','2000-2004 Outcome Period','2003-2007 Outcome Period'
]

# Calculate class imbalance weight
pos_weight = len(df[df[y_col]==0]) / len(df[df[y_col]==1])

# Hyperparameter search space
param_space = {
    'num_leaves': np.random.randint(20,150,100),
    'min_data_in_leaf': np.random.randint(10,100,100),
    'learning_rate': np.random.uniform(0.01,0.1,100),
    'feature_fraction': np.random.uniform(0.5,1.0,100),
    'bagging_fraction': np.random.uniform(0.5,1.0,100),
}

# Track best model
best_precision = 0
best_params = None
best_model = None

print("Starting cohort-based hyperparam search with weather & spatial features...")
for i in range(len(pre_periods)):
    print(f"\nTraining on {pre_periods[i]}, validating on {outcome_periods[i]}...")
    train_df = df[df['cohort']==pre_periods[i]]
    val_df   = df[df['cohort']==outcome_periods[i]]

    X_train = train_df[x_cols]
    y_train = train_df[y_col]
    X_val   = val_df[x_cols]
    y_val   = val_df[y_col].fillna(0)

    print(f" X_train: {X_train.shape}, X_val: {X_val.shape}")

    for j in range(100):
        if j%10==0: print(f"  Hyperparam iter {j+1}/100")
        params = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'scale_pos_weight':pos_weight,'seed':SEED,
            'num_leaves':param_space['num_leaves'][j],
            'min_data_in_leaf':param_space['min_data_in_leaf'][j],
            'learning_rate':param_space['learning_rate'][j],
            'feature_fraction':param_space['feature_fraction'][j],
            'bagging_fraction':param_space['bagging_fraction'][j]
        }
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1000,
                          callbacks=[lgb.early_stopping(stopping_rounds=50)])
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        prec = precision_score(y_val, preds.round())
        if prec>best_precision:
            print(f"  New best precision {prec:.4f} at iter {j+1}")
            best_precision = prec
            best_params = params
            best_model = model

print(f"\nBest precision: {best_precision:.4f}")
print(f"Best params: {best_params}")

# Final training on all pre-periods
print("\nRetraining final model on all pre-periods...")
final_train = df[df['cohort'].isin(pre_periods)]
X_comb = final_train[x_cols]
y_comb = final_train[y_col]
print(f" Final train shape: {X_comb.shape}")
dtrain = lgb.Dataset(X_comb, label=y_comb)
final_model = lgb.train(best_params, dtrain, num_boost_round=best_model.best_iteration)

# Save model and importances
final_model.save_model(f'{out_dir}/final_model1.txt')
imp_df = pd.DataFrame({'feature':x_cols,'importance':final_model.feature_importance(importance_type='gain')})
imp_df.sort_values('importance',ascending=False).to_csv(f'{out_dir}/lightgbm_feature_importance.csv',index=False)
plt.figure(figsize=(12,6))
sns.barplot(data=imp_df.head(20), x='importance', y='feature')
plt.title('Top 20 Features')
plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_feature_importance.png')
plt.close()

# Evaluate on final test cohort
print("\nEvaluating on test cohort...")
test_df = df[df['cohort']==outcome_periods[-1]]
X_test = test_df[x_cols]
y_test = test_df[y_col].fillna(0)
print(f" Test cohort: {outcome_periods[-1]}, shape: {X_test.shape}")

probs = final_model.predict(X_test)
preds = (probs>0.5).astype(int)

# Precision-recall curve
from sklearn.metrics import precision_recall_curve, auc
prec_vals, rec_vals, thresh = precision_recall_curve(y_test, probs)
pr_auc = auc(rec_vals, prec_vals)
pd.DataFrame({'precision':prec_vals,'recall':rec_vals,'threshold':np.append(thresh,np.nan)}).to_csv(f'{out_dir}/lightgbm_precision_recall_curve.csv',index=False)
plt.figure(figsize=(6,4))
plt.plot(rec_vals,prec_vals,label=f'AUC={pr_auc:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.legend(); plt.tight_layout()
plt.savefig(f'{out_dir}/lightgbm_precision_recall_curve.png'); plt.close()

# Test predictions
test_df['actual_y'] = y_test
test_df['predicted_y'] = preds
test_df['predicted_prob'] = probs
test_df.to_csv(f'{out_dir}/test_predictions_2010.csv',index=False)

# Metrics and SHAP
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
print(f"Test Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
explainer = shap.TreeExplainer(final_model)
shap_vals = explainer.shap_values(X_test)
pd.DataFrame(shap_vals, columns=x_cols).to_csv(f'{out_dir}/lightgbm_shap.csv',index=False)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_vals, X_test, show=False)
plt.tight_layout(); plt.savefig(f'{out_dir}/lightgbm_shap.png'); plt.close()

# Classification report & confusion matrix
report = metrics.classification_report(y_test, preds, target_names=["No","Yes"])
cm = metrics.confusion_matrix(y_test, preds)
with open(f'{out_dir}/lightgbm_report.txt','w') as f:
    f.write('Classification Report\n\n'+report+"\nConfusion Matrix:\n"+np.array2string(cm))

# Confusion matrix heatmap
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='g')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig(f'{out_dir}/lightgbm_confusion_matrix.png'); plt.show()
