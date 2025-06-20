####################################################################################################
## Description: Machine learning program to predict migration in Mexico
## Author: Parth Chawla
## Date: Dec 31, 2024
####################################################################################################

import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import random
import shap

# Set a random seed for reproducibility
SEED = 42
np.random.seed(SEED)   # Set seed for NumPy
random.seed(SEED)      # Set seed for Python's random module
lgb_params = {'seed': SEED}  # Use seed in LightGBM parameters

path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

df = pd.read_csv('data/data_cohort_analysis_add_vars.csv')
vill_cols = [col for col in df if col.startswith('vill_')]

# Create x and y variables (most lagged to avoid leakage):
x_cols1 = ['male', 'age', 'L1_hhchildren', 'L1_hhworkforce', 
           'L1_yrs_in_mx_cum', 'L1_yrs_in_us_cum', 'L1_yrs_in_ag_cum', 'L1_yrs_in_nonag_cum', 
           'L1_yrs_in_mx_ag_sal_cum', 'L1_yrs_in_mx_nonag_sal_cum', 'L1_yrs_in_mx_ag_own_cum', 
           'L1_yrs_in_mx_nonag_own_cum', 'L1_yrs_in_us_ag_sal_cum', 'L1_yrs_in_us_nonag_sal_cum', 
           'L1_yrs_in_us_ag_own_cum', 'L1_yrs_in_us_nonag_own_cum',
           'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag',
           'L1_ag_inc', 'L1_asset_inc', 'L1_farmlab_inc', 'L1_liv_inc', 'L1_nonag_inc', 
           'L1_plot_inc_renta_ag', 'L1_plot_inc_renta_nonag', 'L1_rec_inc', 
           'L1_rem_mx', 'L1_rem_us', 'L1_trans_inc',
           'L1_hh_yrs_in_us_cum', 'L1_hh_migrant']

# Assuming vill_cols is a list of village dummies already defined somewhere in your code
x_cols = x_cols1 + vill_cols  # Final feature columns
y_cols = ['work_us']  # Define the target column

# Define the cohorts for pre-periods and outcome periods
pre_periods = ['1980-1984 Pre-Period', '1985-1989 Pre-Period', '1990-1994 Pre-Period', 
               '1995-1999 Pre-Period', '2000-2004 Pre-Period', '2005-2010 Pre-Period']
outcome_periods = ['1980-1984 Outcome Period', '1985-1989 Outcome Period', '1990-1994 Outcome Period', 
                   '1995-1999 Outcome Period', '2000-2004 Outcome Period', '2005-2010 Outcome Period']

df['male'] = pd.to_numeric(df['male'], errors='coerce')  # Convert strings or mixed types to numeric

# Inputs: final_model, X_test

# ---- Load Trained Model ----
final_model = lgb.Booster(model_file='output/final_model1.txt')

# ---- Filter X_test of non-migrants ----
test_data = df[df['cohort'] == outcome_periods[-1]]
X_test = test_data[x_cols]
y_test = test_data[y_cols]
non_migrants = test_data[y_test.values.flatten() == 0]  # Filter where y_test is 0

# Function to calculate probabilities for a given input
def predict_proba(model, X):
    return model.predict(X)

# Gradient ascent to maximize probability of work_us = 1
def maximize_probability(model, X_initial, max_iter=100, learning_rate=0.01, constraints=None, exclude_cols=None):
    X = X_initial.copy()
    
    # Predict the initial probability
    y_pred_initial = predict_proba(model, X)
    print("Initial Predicted Probability:\n", y_pred_initial)

    for i in range(max_iter):
        # Predict probability
        y_pred = predict_proba(model, X)
        
        # Compute gradients (using SHAP as a proxy)
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(X)
        # Check if SHAP values are in a list format (LightGBM binary classifier)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take the first (and only) class if binary classification
        
        # Normalize SHAP values
        shap_values = shap_values / np.linalg.norm(shap_values, axis=1, keepdims=True)
        
        # Exclude specified columns (e.g., village dummies) from updates
        if exclude_cols is not None:
            for col in exclude_cols:
                shap_values[:, X.columns.get_loc(col)] = 0

        # Identify the top 5 features for each individual
        for row_idx in range(len(X)):
            top_features = np.argsort(-np.abs(shap_values[row_idx, :]))[:5]
            # Update only the top 5 features for the current individual
            X.iloc[row_idx, top_features] += learning_rate * shap_values[row_idx, top_features]
        
        # X += learning_rate * normalized_shap_values
        
        # Apply constraints (e.g., age >= 0)
        if constraints:
            X = constraints(X)
        
        # Stop if the probability is sufficiently close to 1
        if np.all(y_pred > 0.5):
            print(f"Converged at iteration {i}")
            break
    
    return X, y_pred

# Define constraints function (example)
def apply_constraints(X):
    X['age'] = np.clip(X['age'], 0, 120)  # Age should be between 0 and 120
    return X

# Example usage
X_initial = non_migrants[x_cols].iloc[:5, :]  # Select one non-migrant
print("Initial Features:\n", X_initial)

X_adjusted, y_pred_adjusted = maximize_probability(
    final_model,
    X_initial,
    max_iter=100,
    learning_rate=0.05,
    constraints=apply_constraints,
    exclude_cols=vill_cols  # Exclude village dummies from updates
)

print("Final Predicted Probability:\n", y_pred_adjusted)

# Export initial and adjusted values
output_file_initial = 'output/initial_features.csv'
output_file_adjusted = 'output/adjusted_features.csv'
X_initial.to_csv(output_file_initial, index=False)
X_adjusted.to_csv(output_file_adjusted, index=False)
