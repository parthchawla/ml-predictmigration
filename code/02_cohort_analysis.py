####################################################################################################
## Description: Machine learning program to predict migration in Mexico
## Author: Parth Chawla
## Date: Aug 25, 2024
####################################################################################################

import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

df = pd.read_csv('data/data_cohort_analysis.csv')
vill_cols = [col for col in df if col.startswith('vill_')]

# Create x and y variables:
x_cols1 = ['male', 'age', 'hhchildren', 'hhworkforce', 'ag', 'nonag', 
           'yrs_in_mx_cum', 'yrs_in_us_cum', 'yrs_in_ag_cum', 'yrs_in_nonag_cum', 
           'yrs_in_mx_ag_sal_cum', 'yrs_in_mx_nonag_sal_cum', 'yrs_in_mx_ag_own_cum', 
           'yrs_in_mx_nonag_own_cum', 'yrs_in_us_ag_sal_cum', 'yrs_in_us_nonag_sal_cum', 
           'yrs_in_us_ag_own_cum', 'yrs_in_us_nonag_own_cum',
           'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag']

# Assuming vill_cols is a list of village dummies already defined somewhere in your code
x_cols = x_cols1 + vill_cols  # Final feature columns
y_cols = ['work_us']  # Define the target column

# Define the cohorts for pre-periods and outcome periods
pre_periods = ['1980-1984 Pre-Period', '1985-1989 Pre-Period', '1990-1994 Pre-Period', 
               '1995-1999 Pre-Period', '2000-2004 Pre-Period', '2005-2010 Pre-Period']
outcome_periods = ['1980-1984 Outcome Period', '1985-1989 Outcome Period', '1990-1994 Outcome Period', 
                   '1995-1999 Outcome Period', '2000-2004 Outcome Period', '2005-2010 Outcome Period']

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

# Print start of the process
print("Starting the cohort-based training and validation process...")

# Loop through each cohort (excluding the last one for testing)
for i in range(len(pre_periods)):
    # Define training data (from the pre-period of the current cohort)
    print(f"\nProcessing cohort {pre_periods[i]} for training and {outcome_periods[i]} for validation...")

    train_data = df[df['cohort'] == pre_periods[i]]
    X_train = train_data[x_cols]  # Features
    y_train = train_data[y_cols]  # Target

    # Print the head of the training data
    print("\nTraining Data Head:")
    print(train_data.head())
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Define validation data (from the outcome period of the next cohort)
    validate_data = df[df['cohort'] == outcome_periods[i]]
    X_validate = validate_data[x_cols]  # Features
    y_validate = validate_data[y_cols]  # Target

    y_validate = y_validate.fillna(0)  # Fill NaNs with 0

    # Print the head of the validation data
    print("\nValidation Data Head:")
    print(validate_data.head())
    print(f"X_validate shape: {X_validate.shape}, y_validate shape: {y_validate.shape}")

    # Perform random search over 100 hyperparameter configurations
    for j in range(2):
        if j % 10 == 0:  # Print progress every 10 iterations
            print(f"Hyperparameter configuration {j+1}/100")

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
        
        # Train the model with early stopping using a callback
        model = lgb.train(
            params,
            d_train,
            valid_sets=[d_validate],
            num_boost_round=1000,  # Maximum number of boosting rounds
            callbacks=[lgb.early_stopping(stopping_rounds=50)]  # Early stopping callback
        )
        
        # Predict on the validation set and calculate precision
        y_validate_pred = model.predict(X_validate, num_iteration=model.best_iteration)
        precision = precision_score(y_validate, y_validate_pred.round())
        
        # Update the best model if the current one has higher precision
        if precision > best_precision:
            print(f"New best precision found: {precision:.4f} at configuration {j+1}")
            best_precision = precision
            best_params = params
            best_model = model

# Print the best hyperparameters and the corresponding precision
print(f"\nBest Precision: {best_precision}")
print(f"Best Parameters: {best_params}")

# Final Training on the combined training and validation data using the best hyperparameters
print("\nTraining the final model on the combined training and validation data...")
final_train_data = df[df['cohort'].isin(pre_periods)]  # Combine all pre-periods (now includes up to 2005-2009)
X_combined = final_train_data[x_cols]  # Features
y_combined = final_train_data[y_cols]  # Target

# Print cohorts included in X_combined and y_combined
print(f"\nCohorts included in X_combined and y_combined: {final_train_data['cohort'].unique()}")
print(f"X_combined shape: {X_combined.shape}, y_combined shape: {y_combined.shape}")
print("\nCombined Training Data Head:")
print(final_train_data.head())

# Create a dataset for final training
d_combined = lgb.Dataset(X_combined, label=y_combined)
final_model = lgb.train(best_params, d_combined, num_boost_round=best_model.best_iteration)

# Evaluate the final model on the test cohort (last cohort's outcome period)
print("\nEvaluating the final model on the test cohort...")
test_data = df[df['cohort'] == outcome_periods[-1]]
X_test = test_data[x_cols]  # Features
y_test = test_data[y_cols]  # Target

y_test = y_validate.fillna(0)  # Fill NaNs with 0

# Print cohorts included in X_test and y_test
print(f"\nCohorts included in X_test and y_test: {test_data['cohort'].unique()}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("\nTest Data Head:")
print(test_data.head())

# Predict on the test set and calculate precision
y_test_pred = final_model.predict(X_test)
test_precision = precision_score(y_test, y_test_pred.round())
print(f'Test Precision: {test_precision}')
