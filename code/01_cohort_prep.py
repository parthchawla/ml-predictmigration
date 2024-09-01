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
    if 1980 <= year <= 1987:
        return "1980-1989 Pre-Period"
    elif 1988 <= year <= 1989:
        return "1980-1989 Outcome Period"
    elif 1990 <= year <= 1997:
        return "1990-1999 Pre-Period"
    elif 1998 <= year <= 1999:
        return "1990-1999 Outcome Period"
    elif 2000 <= year <= 2008:
        return "2000-2010 Pre-Period"
    elif 2009 <= year <= 2010:
        return "2000-2010 Outcome Period"
    else:
        return "Outside Range"

dtafile = 'data/MexMigData.dta'
df, meta = pyreadstat.read_dta(dtafile)

cols_to_move = ['ind', 'year']
df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]] # cols to front
df = df.sort_values(by=['ind', 'year'], ascending=[True, True]) # sort

# Create cumulative variables:
df['work_in_mx'] = np.where((df['work_loc']==1) | (df['work_mx']==1), 1, 0) # worked in mx in year t
df['yrs_in_mx_cum'] = df.groupby(['ind'])['work_in_mx'].cumsum() # years worked in mx till t
df['yrs_in_us_cum'] = df.groupby('ind')['work_us'].cumsum() # years worked in the US till t
df['yrs_in_ag_cum'] = df.groupby('ind')['ag'].cumsum() # years worked in ag till t
df['yrs_in_nonag_cum'] = df.groupby('ind')['nonag'].cumsum() # years worked in ag till t

df['work_mx_ag_sal'] = np.where((df['loc_ag_sal_']==1) | (df['mx_ag_sal_']==1), 1, 0)
df['work_mx_nonag_sal'] = np.where((df['loc_nonag_sal_']==1) | (df['mx_nonag_sal_']==1), 1, 0)
df['yrs_in_mx_ag_sal_cum'] = df.groupby(['ind'])['work_mx_ag_sal'].cumsum()
df['yrs_in_mx_nonag_sal_cum'] = df.groupby(['ind'])['work_mx_nonag_sal'].cumsum()

df['work_mx_ag_own'] = np.where((df['loc_ag_own_']==1) | (df['mx_ag_own_']==1), 1, 0)
df['work_mx_nonag_own'] = np.where((df['loc_nonag_own_']==1) | (df['mx_nonag_own_']==1), 1, 0)
df['yrs_in_mx_ag_own_cum'] = df.groupby(['ind'])['work_mx_ag_own'].cumsum()
df['yrs_in_mx_nonag_own_cum'] = df.groupby(['ind'])['work_mx_nonag_own'].cumsum()

df['yrs_in_us_ag_sal_cum'] = df.groupby(['ind'])['us_ag_sal_'].cumsum()
df['yrs_in_us_nonag_sal_cum'] = df.groupby(['ind'])['us_nonag_sal_'].cumsum()
df['yrs_in_us_ag_own_cum'] = df.groupby(['ind'])['us_ag_own_'].cumsum()
df['yrs_in_us_nonag_own_cum'] = df.groupby(['ind'])['us_nonag_own_'].cumsum()

print(df[['yrs_in_mx_cum', 'yrs_in_us_cum', 'yrs_in_ag_cum', 'yrs_in_nonag_cum', 
          'yrs_in_mx_ag_sal_cum', 'yrs_in_mx_nonag_sal_cum', 'yrs_in_mx_ag_own_cum', 
          'yrs_in_mx_nonag_own_cum', 'yrs_in_us_ag_sal_cum', 'yrs_in_us_nonag_sal_cum', 
          'yrs_in_us_ag_own_cum', 'yrs_in_us_nonag_own_cum']].head())

df['cohort'] = df['year'].apply(assign_cohort)
cohort_counts = df['cohort'].value_counts().sort_index()
print(cohort_counts)

# Create village dummies:
vill_dummies = pd.get_dummies(df['villageid'], drop_first=True, prefix="vill", dtype=int)
df = pd.concat([df, vill_dummies], axis=1)

df['L1_work_us'] = df.groupby('ind')['work_us'].shift(1)
df['L1_work_mx'] = df.groupby('ind')['work_in_mx'].shift(1)
df['L1_ag'] = df.groupby('ind')['ag'].shift(1)
df['L1_nonag'] = df.groupby('ind')['nonag'].shift(1)

print(df.head())
df.to_csv('data/data_cohort_analysis.csv', index=False)
