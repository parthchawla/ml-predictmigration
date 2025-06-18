import os
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt

# 1) SETUP
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

os.chdir('/Users/parthchawla1/GitHub/ml-predictmigration/')

# 2) LOAD DATA
df = pd.read_csv('data/data_cohort_analysis_add_vars.csv')
vill_cols = [col for col in df.columns if col.startswith('vill_')]

# 3) DEFINE FEATURES & TARGET
x_cols1 = [
    'male', 'age', 'L1_hhchildren', 'L1_hhworkforce',
    'L1_yrs_in_mx_cum', 'L1_yrs_in_us_cum', 'L1_yrs_in_ag_cum', 'L1_yrs_in_nonag_cum',
    'L1_yrs_in_mx_ag_sal_cum', 'L1_yrs_in_mx_nonag_sal_cum',
    'L1_yrs_in_mx_ag_own_cum', 'L1_yrs_in_mx_nonag_own_cum',
    'L1_yrs_in_us_ag_sal_cum', 'L1_yrs_in_us_nonag_sal_cum',
    'L1_yrs_in_us_ag_own_cum', 'L1_yrs_in_us_nonag_own_cum',
    'L1_work_us', 'L1_work_mx', 'L1_ag', 'L1_nonag',
    'L1_ag_inc', 'L1_asset_inc', 'L1_farmlab_inc', 'L1_liv_inc', 'L1_nonag_inc',
    'L1_plot_inc_renta_ag', 'L1_plot_inc_renta_nonag', 'L1_rec_inc',
    'L1_rem_mx', 'L1_rem_us', 'L1_trans_inc',
    'L1_hh_yrs_in_us_cum', 'L1_hh_migrant'
]
x_cols = x_cols1 + vill_cols
y_col = 'work_us'

# Map gender to 0/1:
mapping = {
    'M': 1, 'm': 1, 'Male': 1, 'male': 1,
    'F': 0, 'f': 0, 'Female': 0, 'female': 0
}
df['male'] = df['male'].map(mapping)
df['male'] = pd.to_numeric(df['male'], errors='coerce')

# 4) SPLIT TRAIN/TEST (ARDL-style)
train_df = df[df['year'] < 2010]
test_df  = df[df['year'] == 2010]

X_train = train_df[x_cols]
y_train = train_df[y_col].fillna(0)
X_test  = test_df[x_cols]
y_test  = test_df[y_col].fillna(0)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}")

# 5) PIPELINE: IMPUTER + LOGISTIC REGRESSION
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', LogisticRegression(
        solver='liblinear',
        random_state=SEED,
        class_weight='balanced',
        max_iter=500
    ))
])

# 6) TRAIN
pipeline.fit(X_train, y_train)

# 7) PREDICT & EVALUATE
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 8) SAVE OUTPUTS
report = classification_report(y_test, y_pred, target_names=["No US work", "Worked US"])
cm = confusion_matrix(y_test, y_pred)
with open('output/logistic_report.txt', 'w') as f:
    f.write("Classification Report\n\n" + report + "\nConfusion Matrix\n\n" + np.array2string(cm))

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix\n(Logistic ARDL-Style Split)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('output/logistic_confusion_matrix.png')
plt.close()

print("ARDL-style logistic regression complete. Outputs saved.")
