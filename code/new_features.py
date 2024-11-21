import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import random
import shap

# Set a random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
lgb_params = {'seed': SEED}

path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

# Load data
df = pd.read_csv('data/data_cohort_analysis.csv')

# Feature engineering
df['age_squared'] = df['age'] ** 2  # Non-linear age effects
df['workforce_ratio'] = df['hhworkforce'] / (df['hhchildren'] + 1)  # Family workforce ratio
df['total_experience'] = df['yrs_in_mx_cum'] + df['yrs_in_us_cum']  # Total work experience
df['ag_experience'] = df['yrs_in_ag_cum'] * df['ag']  # Agricultural experience interaction
df['nonag_experience'] = df['yrs_in_nonag_cum'] * df['nonag']  # Non-agricultural experience interaction
df['recent_us_work'] = (df['L1_work_us'] > 0).astype(int)  # Recent US work indicator

vill_cols = [col for col in df if col.startswith('vill_')]

# Updated feature columns including new engineered features
x_cols1 = ['male', 'age', 'age_squared', 'hhchildren', 'hhworkforce', 'workforce_ratio',
           'ag', 'nonag', 'ag_experience', 'nonag_experience',
           'yrs_in_mx_cum', 'yrs_in_us_cum', 'total_experience',
           'yrs_in_ag_cum', 'yrs_in_nonag_cum',
           'yrs_in_mx_ag_sal_cum', 'yrs_in_mx_nonag_sal_cum', 'yrs_in_mx_ag_own_cum',
           'yrs_in_mx_nonag_own_cum', 'yrs_in_us_ag_sal_cum', 'yrs_in_us_nonag_sal_cum',
           'yrs_in_us_ag_own_cum', 'yrs_in_us_nonag_own_cum',
           'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag', 'recent_us_work']

x_cols = x_cols1 + vill_cols
y_cols = ['work_us']

# Define periods (same as before)
pre_periods = ['1980-1984 Pre-Period', '1985-1989 Pre-Period', '1990-1994 Pre-Period',
               '1995-1999 Pre-Period', '2000-2004 Pre-Period', '2005-2010 Pre-Period']
outcome_periods = ['1980-1984 Outcome Period', '1985-1989 Outcome Period', '1990-1994 Outcome Period',
                   '1995-1999 Outcome Period', '2000-2004 Outcome Period', '2005-2010 Outcome Period']

# Enhanced hyperparameter space
param_space = {
    'num_leaves': np.random.randint(20, 150, size=100),
    'min_data_in_leaf': np.random.randint(10, 100, size=100),
    'learning_rate': np.random.uniform(0.01, 0.1, size=100),  # Narrower range for more stable training
    'feature_fraction': np.random.uniform(0.6, 1.0, size=100),
    'bagging_fraction': np.random.uniform(0.6, 1.0, size=100),
    'reg_alpha': np.random.uniform(0, 10.0, size=100),  # L1 regularization
    'reg_lambda': np.random.uniform(0, 10.0, size=100),  # L2 regularization
    'min_split_gain': np.random.uniform(0, 1.0, size=100),  # Minimum gain for splitting
    'max_depth': np.random.randint(5, 20, size=100)  # Control tree depth
}

best_precision = 0
best_params = None
best_model = None

print("Starting the cohort-based training and validation process...")

# Training loop (modified for better validation)
for i in range(len(pre_periods)):
    print(f"\nProcessing cohort {pre_periods[i]} for training and {outcome_periods[i]} for validation...")

    train_data = df[df['cohort'] == pre_periods[i]]
    X_train = train_data[x_cols]
    y_train = train_data[y_cols]

    validate_data = df[df['cohort'] == outcome_periods[i]]
    X_validate = validate_data[x_cols]
    y_validate = validate_data[y_cols].fillna(0)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_validate shape: {X_validate.shape}, y_validate shape: {y_validate.shape}")

    # Enhanced random search
    for j in range(100):
        if j % 10 == 0:
            print(f"Hyperparameter configuration {j+1}/100")

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': param_space['num_leaves'][j],
            'min_data_in_leaf': param_space['min_data_in_leaf'][j],
            'learning_rate': param_space['learning_rate'][j],
            'feature_fraction': param_space['feature_fraction'][j],
            'bagging_fraction': param_space['bagging_fraction'][j],
            'reg_alpha': param_space['reg_alpha'][j],
            'reg_lambda': param_space['reg_lambda'][j],
            'min_split_gain': param_space['min_split_gain'][j],
            'max_depth': param_space['max_depth'][j],
            'metric': ['binary_logloss', 'auc'],  # Multiple metrics for monitoring
            'verbose': -1,
            'seed': SEED
        }

        d_train = lgb.Dataset(X_train, label=y_train)
        d_validate = lgb.Dataset(X_validate, label=y_validate, reference=d_train)

        # Enhanced training with multiple evaluation metrics
        model = lgb.train(
            params,
            d_train,
            valid_sets=[d_validate],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Suppress logging
            ]
        )

        # Evaluate using multiple metrics
        y_validate_pred = model.predict(X_validate, num_iteration=model.best_iteration)
        precision = precision_score(y_validate, y_validate_pred.round())
        auc = roc_auc_score(y_validate, y_validate_pred)
        avg_precision = average_precision_score(y_validate, y_validate_pred)

        # Use combined metric for model selection
        combined_metric = (precision + auc + avg_precision) / 3
        
        if combined_metric > best_precision:
            print(f"New best combined metric: {combined_metric:.4f} at configuration {j+1}")
            best_precision = combined_metric
            best_params = params
            best_model = model

# Final model training and evaluation (same as before, with additional metrics)
print("\nTraining final model...")
final_train_data = df[df['cohort'].isin(pre_periods)]
X_combined = final_train_data[x_cols]
y_combined = final_train_data[y_cols]

d_combined = lgb.Dataset(X_combined, label=y_combined)
final_model = lgb.train(best_params, d_combined, num_boost_round=best_model.best_iteration)

# Calculate feature importance
importance_df = pd.DataFrame({
    'feature': x_cols,
    'importance': final_model.feature_importance(importance_type='gain')
})
importance_df = importance_df.sort_values('importance', ascending=False)

# Save feature importance plot
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('output/feature_importance.png')
plt.close()

# Final evaluation
print("\nEvaluating final model...")
test_data = df[df['cohort'] == outcome_periods[-1]]
X_test = test_data[x_cols]
y_test = test_data[y_cols].fillna(0)

y_test_pred = final_model.predict(X_test)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Calculate and save comprehensive metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred_binary, average='binary')
auc = roc_auc_score(y_test, y_test_pred)
avg_precision = average_precision_score(y_test, y_test_pred)

print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')
print(f'Test F1 Score: {f1}')
print(f'Test AUC: {auc}')
print(f'Test Average Precision: {avg_precision}')

# Generate and save SHAP values for interpretability
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('output/shap_summary.png')
plt.close()

# Save confusion matrix and classification report (same as before)
target_names = ["Didn't work in the US", "Worked in the US"]
cr = metrics.classification_report(y_test, y_test_pred_binary, target_names=target_names)
cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred_binary)

with open('output/report_lightgbm.txt', 'w') as f:
    f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, str(cnf_matrix)))

# Create confusion matrix heatmap
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('output/confusion_matrix_lightgbm.png', bbox_inches='tight')
plt.close()
