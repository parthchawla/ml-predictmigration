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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import lightgbm as lgb

path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

df = pd.read_csv('data/data_cohort_analysis.csv')
vill_cols = [col for col in df if col.startswith('vill_')]

#                   cohort |      Freq.     Percent        Cum.
# -------------------------+-----------------------------------
# 1980-1989 Outcome Period |     10,444        4.83        4.83
#     1980-1989 Pre-Period |     41,776       19.30       24.13
# 1990-1999 Outcome Period |     15,466        7.15       31.27
#     1990-1999 Pre-Period |     61,864       28.58       59.85
# 2000-2010 Outcome Period |     11,982        5.54       65.39
#     2000-2010 Pre-Period |     74,920       34.61      100.00
# -------------------------+-----------------------------------
#                    Total |    216,452      100.00

pre_periods = ['1980-1989 Pre-Period', '1990-1999 Pre-Period', '2000-2010 Pre-Period']
outcome_periods = ['1980-1989 Outcome Period', '1990-1999 Outcome Period', '2000-2010 Outcome Period']

# Initialize variables to store the best model and its parameters
best_precision = 0
best_params = None
best_model = None

# Define the hyperparameter space for random search
param_space = {
    'num_leaves': np.random.randint(20, 100, size=100),  # Max number of leaves in one tree
    'min_data_in_leaf': np.random.randint(10, 100, size=100),  # Min samples required in a leaf
    'learning_rate': np.random.uniform(0.01, 0.3, size=100),  # Step size for each boosting step
    'feature_fraction': np.random.uniform(0.5, 1.0, size=100),  # Fraction of features used per tree
    'bagging_fraction': np.random.uniform(0.5, 1.0, size=100)  # Fraction of data used per iteration
}

# # Split the data into pre-period and outcome period data
# pre_data = df[df['cohort'].isin(pre_periods)]
# outcome_data = df[df['cohort'].isin(outcome_periods)]

# # Example: Training on the first cohort's pre-period and testing on its outcome period
# train_data = pre_data[pre_data['cohort'] == '1980-1989 Pre-Period']
# test_data = outcome_data[outcome_data['cohort'] == '1980-1989 Outcome Period']

# # Create x and y variables:
# x_cols1 = ['male', 'age', 'hhchildren', 'hhworkforce', 'ag', 'nonag', 
#            'yrs_in_mx_cum', 'yrs_in_us_cum', 'yrs_in_ag_cum', 'yrs_in_nonag_cum', 
#            'yrs_in_mx_ag_sal_cum', 'yrs_in_mx_nonag_sal_cum', 'yrs_in_mx_ag_own_cum', 
#            'yrs_in_mx_nonag_own_cum', 'yrs_in_us_ag_sal_cum', 'yrs_in_us_nonag_sal_cum', 
#            'yrs_in_us_ag_own_cum', 'yrs_in_us_nonag_own_cum',
#            'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag']
# x_cols = x_cols1 + vill_cols
# y_cols = ['work_us']

# x_train = train_data[x_cols]
# y_train = train_data[y_cols]
# x_test = test_data[x_cols]
# y_test = test_data[y_cols]

# Hyperparameter tuning for Gradient Booster:
# gbc = GradientBoostingClassifier()
# parameters = {
#     "n_estimators":[50,100,250],
#     "max_depth":[1,3,5],
#     "learning_rate":[0.01,0.1,1]
# }
# cv = GridSearchCV(gbc,parameters,cv=2)
# cv.fit(x_train, y_train.values.ravel())
# print(f'Best parameters are: {results.best_params_}')
# Best parameters are: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}

# Create x and y variables:
x_cols1 = ['male', 'age', 'hhchildren', 'hhworkforce', 'ag', 'nonag', 
           'yrs_in_mx_cum', 'yrs_in_us_cum', 'yrs_in_ag_cum', 'yrs_in_nonag_cum', 
           'yrs_in_mx_ag_sal_cum', 'yrs_in_mx_nonag_sal_cum', 'yrs_in_mx_ag_own_cum', 
           'yrs_in_mx_nonag_own_cum', 'yrs_in_us_ag_sal_cum', 'yrs_in_us_nonag_sal_cum', 
           'yrs_in_us_ag_own_cum', 'yrs_in_us_nonag_own_cum',
           'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag']
x_cols = x_cols1 + vill_cols
y_cols = ['work_us']

# Define training data (from the pre-period of the current cohort)
train_data = df[df['cohort'] == pre_periods[0]]
X_train = train_data[x_cols]
y_train = train_data[y_cols]

# Define validation data (from the outcome period of the next cohort)
validate_data = df[df['cohort'] == outcome_periods[0]]
X_validate = validate_data[x_cols]
y_validate = validate_data[y_cols]

# Perform random search over 100 hyperparameter configurations
for j in range(100):
    # Sample a unique set of hyperparameters
    params = {
        'objective': 'binary',  # Binary classification objective
        'num_leaves': param_space['num_leaves'][j],
        'min_data_in_leaf': param_space['min_data_in_leaf'][j],
        'learning_rate': param_space['learning_rate'][j],
        'feature_fraction': param_space['feature_fraction'][j],
        'bagging_fraction': param_space['bagging_fraction'][j],
        'metric': 'binary_logloss',  # Use log loss as the evaluation metric
        'verbose': -1  # Suppress all LightGBM output
    }
    
    # Create LightGBM datasets for training and validation
    d_train = lgb.Dataset(X_train, label=y_train)
    d_validate = lgb.Dataset(X_validate, label=y_validate, reference=d_train)
    
    # Train the model with early stopping
    model = lgb.train(
        params,
        d_train,
        valid_sets=[d_validate],
        num_boost_round=1000,  # Maximum number of boosting rounds
        callbacks=[lgb.early_stopping(stopping_rounds=50)]  # Early stopping callback, stop if no improvement for 50 rounds
    )
    
    # Predict on the validation set and calculate precision
    y_validate_pred = model.predict(X_validate, num_iteration=model.best_iteration)
    precision = precision_score(y_validate, y_validate_pred.round())
    
    # Update the best model if the current one has higher precision
    if precision > best_precision:
        best_precision = precision
        best_params = params
        best_model = model

# Print the best hyperparameters and the corresponding precision
print(f"Best Precision: {best_precision}")
print(f"Best Parameters: {best_params}")

exit()

## GPT:
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_score

# Assuming df is your DataFrame with a 'year' column and cohort labels already assigned

# Define the cohorts for pre-periods and outcome periods
pre_periods = ['1980-1989 Pre-Period', '1990-1999 Pre-Period', '2000-2009 Pre-Period']
outcome_periods = ['1980-1989 Outcome Period', '1990-1999 Outcome Period', '2000-2009 Outcome Period']

# Initialize variables to store the best model and its parameters
best_precision = 0
best_params = None
best_model = None

# Define the hyperparameter space for random search
param_space = {
    'num_leaves': np.random.randint(20, 100, size=100),  # Max number of leaves in one tree
    'min_data_in_leaf': np.random.randint(10, 100, size=100),  # Min samples required in a leaf
    'learning_rate': np.random.uniform(0.01, 0.3, size=100),  # Step size for each boosting step
    'feature_fraction': np.random.uniform(0.5, 1.0, size=100),  # Fraction of features used per tree
    'bagging_fraction': np.random.uniform(0.5, 1.0, size=100)  # Fraction of data used per iteration
}

# Loop through each cohort (excluding the last one for testing)
for i in range(len(pre_periods) - 1):
    # Define training data (from the pre-period of the current cohort)
    train_data = df[df['cohort'] == pre_periods[i]]
    X_train = train_data.drop(columns=['target_column', 'cohort'])  # Drop target and cohort labels
    y_train = train_data['target_column']
    
    # Define validation data (from the outcome period of the next cohort)
    validate_data = df[df['cohort'] == outcome_periods[i]]
    X_validate = validate_data.drop(columns=['target_column', 'cohort'])
    y_validate = validate_data['target_column']

    # Perform random search over 100 hyperparameter configurations
    for j in range(100):
        # Sample a unique set of hyperparameters
        params = {
            'objective': 'binary',  # Binary classification objective
            'num_leaves': param_space['num_leaves'][j],
            'min_data_in_leaf': param_space['min_data_in_leaf'][j],
            'learning_rate': param_space['learning_rate'][j],
            'feature_fraction': param_space['feature_fraction'][j],
            'bagging_fraction': param_space['bagging_fraction'][j],
            'metric': 'binary_logloss'  # Use log loss as the evaluation metric
        }
        
        # Create LightGBM datasets for training and validation
        d_train = lgb.Dataset(X_train, label=y_train)
        d_validate = lgb.Dataset(X_validate, label=y_validate, reference=d_train)
        
        # Train the model with early stopping
        model = lgb.train(
            params,
            d_train,
            valid_sets=[d_validate],
            num_boost_round=1000,  # Maximum number of boosting rounds
            early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
            verbose_eval=False  # Suppress output for each iteration
        )
        
        # Predict on the validation set and calculate precision
        y_validate_pred = model.predict(X_validate, num_iteration=model.best_iteration)
        precision = precision_score(y_validate, y_validate_pred.round())
        
        # Update the best model if the current one has higher precision
        if precision > best_precision:
            best_precision = precision
            best_params = params
            best_model = model

# Print the best hyperparameters and the corresponding precision
print(f"Best Precision: {best_precision}")
print(f"Best Parameters: {best_params}")

# Final Training on the combined training and validation data using the best hyperparameters
final_train_data = df[df['cohort'].isin(pre_periods[:-1])]  # Combine all training and validation pre-periods
X_combined = final_train_data.drop(columns=['target_column', 'cohort'])
y_combined = final_train_data['target_column']

# Create a dataset for final training
d_combined = lgb.Dataset(X_combined, label=y_combined)
final_model = lgb.train(best_params, d_combined, num_boost_round=best_model.best_iteration)

# Evaluate the final model on the test cohort (last cohort's outcome period)
test_data = df[df['cohort'] == outcome_periods[-1]]
X_test = test_data.drop(columns=['target_column', 'cohort'])
y_test = test_data['target_column']

# Predict on the test set and calculate precision
y_test_pred = final_model.predict(X_test)
test_precision = precision_score(y_test, y_test_pred.round())
print(f"Test Precision: {test_precision}")
