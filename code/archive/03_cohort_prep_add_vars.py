####################################################################################################
## Description: Machine learning program to predict migration in Mexico
## Author: Parth Chawla
## Date: Aug 25, 2024
####################################################################################################

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # scikit-learn package
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

def assign_cohort(year):
    if 1980 <= year <= 1983:
        return "1980-1984 Pre-Period"
    elif year == 1984:
        return "1980-1984 Outcome Period"
    elif 1985 <= year <= 1988:
        return "1985-1989 Pre-Period"
    elif year == 1989:
        return "1985-1989 Outcome Period"
    elif 1990 <= year <= 1993:
        return "1990-1994 Pre-Period"
    elif year == 1994:
        return "1990-1994 Outcome Period"
    elif 1995 <= year <= 1998:
        return "1995-1999 Pre-Period"
    elif year == 1999:
        return "1995-1999 Outcome Period"
    elif 2000 <= year <= 2003:
        return "2000-2004 Pre-Period"
    elif year == 2004:
        return "2000-2004 Outcome Period"
    elif 2005 <= year <= 2009:
        return "2005-2010 Pre-Period"  # Pre-period includes 2005-2009
    elif year == 2010:
        return "2005-2010 Outcome Period"  # 2010 as outcome period
    else:
        return "Outside Range"

dtafile = 'data/MexMigData_merged.dta'
#df, meta = pyreadstat.read_dta(dtafile) (encoding error)
df = pd.read_stata(dtafile)

cols_to_move = ['ind', 'year']
df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]] # cols to front
df = df.sort_values(by=['ind', 'year'], ascending=[True, True]) # sort

# Create cumulative variables excluding the current year
df['work_in_mx'] = np.where((df['work_loc'] == 1) | (df['work_mx'] == 1), 1, 0)
df['L1_yrs_in_mx_cum'] = df.groupby('ind')['work_in_mx'].cumsum().shift(1)
df['L1_yrs_in_us_cum'] = df.groupby('ind')['work_us'].cumsum().shift(1)
df['L1_yrs_in_ag_cum'] = df.groupby('ind')['ag'].cumsum().shift(1)
df['L1_yrs_in_nonag_cum'] = df.groupby('ind')['nonag'].cumsum().shift(1)

df['work_mx_ag_sal'] = np.where((df['loc_ag_sal_'] == 1) | (df['mx_ag_sal_'] == 1), 1, 0)
df['work_mx_nonag_sal'] = np.where((df['loc_nonag_sal_'] == 1) | (df['mx_nonag_sal_'] == 1), 1, 0)
df['L1_yrs_in_mx_ag_sal_cum'] = df.groupby('ind')['work_mx_ag_sal'].cumsum().shift(1)
df['L1_yrs_in_mx_nonag_sal_cum'] = df.groupby('ind')['work_mx_nonag_sal'].cumsum().shift(1)

df['work_mx_ag_own'] = np.where((df['loc_ag_own_'] == 1) | (df['mx_ag_own_'] == 1), 1, 0)
df['work_mx_nonag_own'] = np.where((df['loc_nonag_own_'] == 1) | (df['mx_nonag_own_'] == 1), 1, 0)
df['L1_yrs_in_mx_ag_own_cum'] = df.groupby('ind')['work_mx_ag_own'].cumsum().shift(1)
df['L1_yrs_in_mx_nonag_own_cum'] = df.groupby('ind')['work_mx_nonag_own'].cumsum().shift(1)

df['L1_yrs_in_us_ag_sal_cum'] = df.groupby('ind')['us_ag_sal_'].cumsum().shift(1)
df['L1_yrs_in_us_nonag_sal_cum'] = df.groupby('ind')['us_nonag_sal_'].cumsum().shift(1)
df['L1_yrs_in_us_ag_own_cum'] = df.groupby('ind')['us_ag_own_'].cumsum().shift(1)
df['L1_yrs_in_us_nonag_own_cum'] = df.groupby('ind')['us_nonag_own_'].cumsum().shift(1)

# Entire household cumsum, lags
df = df.sort_values(by=['numc', 'year'], ascending=[True, True]) # sort
df['L1_hh_yrs_in_us_cum'] = df.groupby('numc')['work_us'].cumsum().shift(1)
df['L1_hhchildren'] = df.groupby('numc')['hhchildren'].shift(1)
df['L1_hhworkforce'] = df.groupby('numc')['hhworkforce'].shift(1)
df['L1_ag_inc'] = df.groupby('numc')['ag_inc'].shift(1)
df['L1_asset_inc'] = df.groupby('numc')['asset_inc'].shift(1)
df['L1_farmlab_inc'] = df.groupby('numc')['farmlab_inc'].shift(1)
df['L1_liv_inc'] = df.groupby('numc')['liv_inc'].shift(1)
df['L1_nonag_inc'] = df.groupby('numc')['nonag_inc'].shift(1)
df['L1_plot_inc_renta_ag'] = df.groupby('numc')['plot_inc_renta_ag'].shift(1)
df['L1_plot_inc_renta_nonag'] = df.groupby('numc')['plot_inc_renta_nonag'].shift(1)
df['L1_rec_inc'] = df.groupby('numc')['rec_inc'].shift(1)
df['L1_rem_mx'] = df.groupby('numc')['rem_mx'].shift(1)
df['L1_rem_us'] = df.groupby('numc')['rem_us'].shift(1)
df['L1_trans_inc'] = df.groupby('numc')['trans_inc'].shift(1)

df['cohort'] = df['year'].apply(assign_cohort)
cohort_counts = df['cohort'].value_counts().sort_index()
print(cohort_counts)

# Create village dummies:
vill_dummies = pd.get_dummies(df['villageid'], drop_first=True, prefix="vill", dtype=int)
df = pd.concat([df, vill_dummies], axis=1)

# Individual lags
df = df.sort_values(by=['ind', 'year'], ascending=[True, True]) # sort
df['L1_work_us'] = df.groupby('ind')['work_us'].shift(1)
df['L1_work_mx'] = df.groupby('ind')['work_in_mx'].shift(1)
df['L1_ag'] = df.groupby('ind')['ag'].shift(1)
df['L1_nonag'] = df.groupby('ind')['nonag'].shift(1)
df['L2_work_us'] = df.groupby('ind')['work_us'].shift(2)
df['L2_work_mx'] = df.groupby('ind')['work_in_mx'].shift(2)
df['L2_ag'] = df.groupby('ind')['ag'].shift(2)
df['L2_nonag'] = df.groupby('ind')['nonag'].shift(2)

# Check if any household member was a migrant
df['L1_hh_migrant_any_year'] = df.groupby('numc')['L1_work_us'].transform('max')
df['L1_hh_migrant'] = df.groupby(['numc', 'year'])['L1_work_us'].transform('max')
df['L2_hh_migrant'] = df.groupby(['numc', 'year'])['L2_work_us'].transform('max')

# Move 'cohort' to the first column:
cols = ['cohort'] + [col for col in df.columns if col != 'cohort']
df = df[cols]

print(df.head())
df.to_csv('data/data_cohort_analysis_add_vars.csv', index=False)
