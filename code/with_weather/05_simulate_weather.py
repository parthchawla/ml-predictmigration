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

# ---- Load Trained Model ----
final_model = lgb.Booster(model_file='output/final_model1.txt')

# ---- Remittance Shock Simulations ----
print("\nEvaluating the final model on the test cohort with shocks...")
test_data = df[df['cohort'] == outcome_periods[-1]]
X_test = test_data[x_cols]
y_test = test_data[y_cols]

y_test = y_test.fillna(0)  # Fill NaNs with 0

# Print cohorts included in X_test and y_test
print(f"\nCohorts included in X_test and y_test: {test_data['cohort'].unique()}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Define shock scenarios
shocks = {
    "original": 1.0,
    "eliminate_remittances": 0.0,
    "double_L1_rem_us": 2.0,
    "double_L1_ag_inc": 2.0,
    "double_L1_farmlab_inc": 2.0,
    "double_L1_nonag_inc": 2.0,
    "double_L1_trans_inc": 2.0,
    "double_L1_hhchildren": 2.0,
    "double_L1_hhworkforce": 2.0,
    "double_age": 2.0,
    "halve_L1_rem_us": 0.5,
    "halve_L1_ag_inc": 0.5,
    "halve_L1_farmlab_inc": 0.5,
    "halve_L1_nonag_inc": 0.5,
    "halve_L1_trans_inc": 0.5,
    "halve_L1_hhchildren": 0.5,
    "halve_L1_hhworkforce": 0.5,
    "halve_age": 0.5,
    "no_L1_work_us": 0.0,
    "no_L1_ag": 0.0,
    "no_L1_nonag": 0.0,
    "yes_L1_work_mx": 1.0,
    "no_male": 0.0,
    'no_L1_hh_migrant': 0.0
}

shock_results = {}

for scenario, value in shocks.items():
    print(f"\nSimulating scenario: {scenario}")

    # Create a modified copy of X_test
    X_test_shocked = X_test.copy()

    # Apply the shock to the relevant variable(s)
    if scenario.startswith("double_"):
        variable = scenario.replace("double_", "")  # Extract the variable name
        X_test_shocked[variable] *= value  # Double the variable
    elif scenario.startswith("halve_"):
        variable = scenario.replace("halve_", "")  # Extract the variable name
        X_test_shocked[variable] *= value  # Double the variable
    elif scenario.startswith("no_"):
        variable = scenario.replace("no_", "")  # Extract the variable name
        X_test_shocked[variable] = value  # Set the variable to 0
    elif scenario.startswith("yes_"):
        variable = scenario.replace("yes_", "")  # Extract the variable name
        X_test_shocked[variable] = value  # Set the variable to 1
    elif scenario == "eliminate_remittances":
        X_test_shocked['L1_rem_us'] *= value  # Set remittances to 0
    else:  # Default to adjusting remittances, handles original
        X_test_shocked['L1_rem_us'] *= value

    # Predict migration probabilities under the shock scenario
    y_test_pred_shocked = final_model.predict(X_test_shocked)
    y_test_pred_binary_shocked = (y_test_pred_shocked > 0.5).astype(int)  # Convert to binary (0 or 1)

    # Add predictions for this scenario to the test dataset
    test_data[f'predicted_prob_{scenario}'] = y_test_pred_shocked
    test_data[f'predicted_y_{scenario}'] = y_test_pred_binary_shocked

    # Calculate and print performance metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred_binary_shocked, average='binary')
    print(f"Scenario: {scenario}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Store results for analysis
    shock_results[scenario] = {
        "probabilities": y_test_pred_shocked,
        "binary_predictions": y_test_pred_binary_shocked,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Add actual target values to the dataset for reference
test_data['actual_y'] = y_test.values

# Save the test dataset with predictions for all scenarios
test_data.to_csv('output/test_predictions_2010_shocks1.csv', index=False)
