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
        'L1_yrs_in_mx_cum','L1_yrs_in_us_cum','L1_yrs_in_ag_cum','L1_yrs_in_nonag_cum',
        'L1_ag','L1_nonag','L1_ag_inc','L1_asset_inc','L1_farmlab_inc','L1_liv_inc',
        'L1_nonag_inc','L1_plot_inc_renta_ag','L1_plot_inc_renta_nonag','L1_rec_inc','L1_trans_inc']

x_cols = base + spatial + weather_cols + vill_cols

y_col = 'work_us'

# 4) SINGLE‐COHORT SPLIT ---------------------------------------------------------------------------
# 2003–2006 as train, 2007 as test
train_data = df[df['cohort']=='2003-2007 Pre-Period']
test_data  = df[df['cohort']=='2003-2007 Outcome Period']

X_train = train_data[x_cols]
y_train = train_data[y_col].fillna(0)
X_test  = test_data[x_cols]
y_test  = test_data[y_col].fillna(0)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# 5) TUNING ---------------------------------------------------------------------------------------
best_precision, best_params, best_model = 0, None, None
# search space
param_space = {
    'num_leaves':       np.random.randint(20,150,100),
    'min_data_in_leaf': np.random.randint(10,100,100),
    'learning_rate':    np.random.uniform(0.01,0.1,100),
    'feature_fraction': np.random.uniform(0.5,1.0,100),
    'bagging_fraction': np.random.uniform(0.5,1.0,100),
}
pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

dtrain = lgb.Dataset(X_train, label=y_train)
for j in range(100):
    if j%10==0: print(f' Iter {j+1}/100')
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'seed':SEED,'scale_pos_weight':pos_weight,
        'num_leaves':int(param_space['num_leaves'][j]),
        'min_data_in_leaf':int(param_space['min_data_in_leaf'][j]),
        'learning_rate':float(param_space['learning_rate'][j]),
        'feature_fraction':float(param_space['feature_fraction'][j]),
        'bagging_fraction':float(param_space['bagging_fraction'][j])
    }
    m = lgb.train(params, dtrain, valid_sets=[(X_test,y_test)],
                  num_boost_round=1000,
                  callbacks=[lgb.early_stopping(stopping_rounds=50)])
    preds = m.predict(X_test, num_iteration=m.best_iteration)
    prec = precision_score(y_test, (preds>0.5).astype(int))
    if prec>best_precision:
        best_precision, best_params, best_model = prec, params, m
        print(f' New best prec {prec:.4f}')
print(f'Best val precision: {best_precision:.4f}\nParams: {best_params}')

# 6) FINAL EVAL 2007 -------------------------------------------------------------------------------
final_model = best_model
pam = final_model.predict(X_test)
pred = (pam>0.5).astype(int)
p, r, f1, _ = precision_recall_fscore_support(y_test,pred,average='binary')
print(f'2007 Test → Prec: {p:.3f}, Rec: {r:.3f}, F1: {f1:.3f}')

# 7) PR CURVE -------------------------------------------------------------------------------------
prec_vals, rec_vals, th = precision_recall_curve(y_test, pam)
pr_auc = auc(rec_vals,prec_vals)
pd.DataFrame({'precision':prec_vals,'recall':rec_vals,'threshold':np.append(th,np.nan)})\
  .to_csv(f'{out_dir}/lightgbm_nm1_precision_recall_curve.csv',index=False)
plt.figure(figsize=(6,4));plt.plot(rec_vals,prec_vals,label=f'AUC={pr_auc:.2f}');
plt.xlabel('Recall');plt.ylabel('Precision');plt.title('PR Curve 2007');plt.legend();plt.tight_layout();
plt.savefig(f'{out_dir}/lightgbm_nm1_precision_recall_curve.png');plt.close()

# 8) IMPORTANCE & SHAP -----------------------------------------------------------------------------
imp = pd.DataFrame({'feature':x_cols,'importance':final_model.feature_importance('gain')})\
   .sort_values('importance',ascending=False)
imp.to_csv(f'{out_dir}/lightgbm_nm1_feature_importance.csv',index=False)
plt.figure(figsize=(12,6));sns.barplot(data=imp.head(20),x='importance',y='feature');
plt.title('Top 20 Features');plt.tight_layout();
plt.savefig(f'{out_dir}/lightgbm_nm1_feature_importance.png');plt.close()

sh = shap.TreeExplainer(final_model).shap_values(X_test)
pd.DataFrame(sh,columns=x_cols).to_csv(f'{out_dir}/lightgbm_nm1_shap.csv',index=False)
plt.figure(figsize=(10,6));shap.summary_plot(sh,X_test,show=False);
plt.title('SHAP 2007');plt.tight_layout();
plt.savefig(f'{out_dir}/lightgbm_nm1_shap.png');plt.close()

# 9) PREDICTIONS & REPORT --------------------------------------------------------------------------
out = test_data.copy(); out['pred']=pred; out['prob']=pam
out.to_csv(f'{out_dir}/test_nm1_predictions_2007.csv',index=False)
cr = classification_report(y_test,pred,target_names=['No US','Yes US'])
cm = confusion_matrix(y_test,pred)
with open(f'{out_dir}/lightgbm_nm1_report.txt','w') as f:
    f.write('Report\n\n'+cr+"\nCM:\n"+np.array2string(cm))
plt.figure(figsize=(6,4));sns.heatmap(cm,annot=True,fmt='g');
plt.title('CM 2007');plt.ylabel('Actual');plt.xlabel('Pred');
plt.tight_layout();plt.savefig(f'{out_dir}/lightgbm_nm1_confusion_matrix.png');plt.close()

print('Single-cohort (2003–2007→2007) analysis complete.')
