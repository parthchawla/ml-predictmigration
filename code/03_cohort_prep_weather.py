####################################################################################################
## Description: Machine learning program to predict migration in Mexico with weather & spatial data
## Author: Parth Chawla
## Date: Aug 25, 2024 (updated Jun 20, 2025)
####################################################################################################

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pyreadstat
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

#-----------------------------------------------------------------------------------
# Set working directory
#-----------------------------------------------------------------------------------
path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

#-----------------------------------------------------------------------------------
# Cohort assignment function (1980–2007 data)
#-----------------------------------------------------------------------------------
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
    elif 2003 <= year <= 2006:
        return "2003-2007 Pre-Period"
    elif year == 2007:
        return "2003-2007 Outcome Period"
    else:
        return "Outside Range"

#-----------------------------------------------------------------------------------
# Load new data with weather & spatial vars
#-----------------------------------------------------------------------------------
dtafile = 'data/MexMigData_merged_weather.dta'
#df, meta = pyreadstat.read_dta(dtafile) (encoding error)
df = pd.read_stata(dtafile)

# Rename ID columns for consistency
if 'indid' in df.columns:
    df.rename(columns={'indid':'ind'}, inplace=True)
if 'householdid' in df.columns:
    df.rename(columns={'householdid':'numc'}, inplace=True)

# Move key cols to front and sort
cols_to_move = ['ind', 'year']
df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
df = df.sort_values(by=['ind', 'year'], ascending=[True, True])

#-----------------------------------------------------------------------------------
# Derive basic flags
#-----------------------------------------------------------------------------------
# Employment flags
if 'local_ag' in df.columns and 'local_nonag' in df.columns:
    df['ag'] = df['local_ag']
    df['nonag'] = df['local_nonag']
else:
    raise KeyError("Expected 'local_ag' and 'local_nonag' in dataset")
# Work in Mexico or US
df['work_in_mx'] = ((df['work_loc'] == 1) | (df['work_mx'] == 1)).astype(int)

#-----------------------------------------------------------------------------------
# Standardize spatial, distance, and wage features
#-----------------------------------------------------------------------------------
for col in ['latitude', 'longitude', 'distkm', 'avtimeloc02', 'local_wage']:
    if col in df.columns:
        df[col + '_std'] = (df[col] - df[col].mean()) / df[col].std()
    else:
        raise KeyError(f"Expected '{col}' in dataset")

#-----------------------------------------------------------------------------------
# Create cumulative history variables with one-year lag
#-----------------------------------------------------------------------------------
df['L1_yrs_in_mx_cum']   = df.groupby('ind')['work_in_mx'].cumsum().shift(1) # check missing/zero
df['L1_yrs_in_us_cum']   = df.groupby('ind')['work_us'].cumsum().shift(1)
df['L1_yrs_in_ag_cum']   = df.groupby('ind')['ag'].cumsum().shift(1)
df['L1_yrs_in_nonag_cum']= df.groupby('ind')['nonag'].cumsum().shift(1)

#-----------------------------------------------------------------------------------
# Income & remittance lags
#-----------------------------------------------------------------------------------
df = df.sort_values(by=['numc', 'year'], ascending=[True, True])
df['L1_hh_yrs_in_us_cum'] = df.groupby('numc')['work_us'].cumsum().shift(1)
for inc in ['hhchildren','hhworkforce','ag_inc','asset_inc','farmlab_inc','liv_inc',
            'nonag_inc','plot_inc_renta_ag','plot_inc_renta_nonag','rec_inc',
            'rem_mx','rem_us','trans_inc']:
    if inc in df.columns:
        df['L1_' + inc] = df.groupby('numc')[inc].shift(1)
    else:
        raise KeyError(f"Expected '{inc}' in dataset")

#-----------------------------------------------------------------------------------
# Assign cohort and generate village dummies
#-----------------------------------------------------------------------------------
df['cohort'] = df['year'].apply(assign_cohort)
vill_dummies = pd.get_dummies(df['villageid'], drop_first=True, prefix="vill", dtype=int)
df = pd.concat([df, vill_dummies], axis=1)

#-----------------------------------------------------------------------------------
# Individual lags for dynamic features
#-----------------------------------------------------------------------------------
df = df.sort_values(by=['ind', 'year'], ascending=[True, True])
for lag in [1,2]:
    df[f'L{lag}_work_us'] = df.groupby('ind')['work_us'].shift(lag)
    df[f'L{lag}_work_mx'] = df.groupby('ind')['work_in_mx'].shift(lag)
    df[f'L{lag}_ag']      = df.groupby('ind')['ag'].shift(lag)
    df[f'L{lag}_nonag']   = df.groupby('ind')['nonag'].shift(lag)

#-----------------------------------------------------------------------------------
# Household migrant indicators
#-----------------------------------------------------------------------------------
df['L1_hh_migrant_any_year'] = df.groupby('numc')['L1_work_us'].transform('max')
df['L1_hh_migrant']          = df.groupby(['numc','year'])['L1_work_us'].transform('max')
if 'L2_work_us' in df.columns:
    df['L2_hh_migrant']      = df.groupby(['numc','year'])['L2_work_us'].transform('max')

#-----------------------------------------------------------------------------------
# Reorder, save, and export
#-----------------------------------------------------------------------------------
cols = ['cohort'] + [c for c in df.columns if c != 'cohort']
df = df[cols]
df.to_csv('data/data_cohort_analysis_weather.csv', index=False)
print("Cohort prep for 1980–2007 with weather & spatial features complete. Rows:", df.shape[0])
