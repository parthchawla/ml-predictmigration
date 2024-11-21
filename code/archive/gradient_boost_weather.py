####################################################################################################
## Description: Machine learning program to predict migration in Mexico
## Author: Parth Chawla
## Date: Nov 8, 2023
####################################################################################################

import os
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

datafile = 'data/LaborWeather_EJ_Main.csv'
data = pd.read_csv(datafile)

data = data.sort_values(by=['indid', 'year'], ascending=[True, True]) # sort

data['work_in_mx'] = np.where((data['work_loc']==1) | (data['work_mx']==1), 1, 0) # worked in mx in year t
data['yrs_worked_in_mx_cum'] = data.groupby(['indid'])['work_in_mx'].cumsum() # years worked in mx till t

data['l1_local_ag'] = data['local_ag'].shift(1)
data['l2_local_ag'] = data['local_ag'].shift(2)
data['l3_local_ag'] = data['local_ag'].shift(3)

data['l1_local_nonag'] = data['local_nonag'].shift(1)
data['l2_local_nonag'] = data['local_nonag'].shift(2)
data['l3_local_nonag'] = data['local_nonag'].shift(3)

print(data.head())

# We want training and testing data to not have the same people

# Get unique list of individuals:
data_copy = data
ind = data_copy.drop_duplicates(subset='indid')
ind = ind[['indid']]

# Create a new variable with a random number between 0 and [no. of unique IDs]:
np.random.seed(42)
ind['rannum'] = (np.random.randint(0, 10000, ind.shape[0]))/10000
ind = ind[['indid', 'rannum']]

# Give each ind a P% chance of being in training and (100-P)% chance of being in test:
ind['MODELING_GROUP'] = np.where((ind.rannum <= 0.75), 'TRAINING', 'TESTING')

# Merge modeling group with data:
data = data.sort_values(by=['indid'], ascending=[True])
ind = ind.sort_values(by=['indid'], ascending=[True])
data = data.merge(ind, on=['indid'], how='inner')
tips_summed = data.groupby(['MODELING_GROUP'])['rannum'].count()
print(tips_summed)

# Create year dummies:
year_dummies = pd.get_dummies(data['year'], drop_first=True, prefix="y", dtype=int)
data = pd.concat([data, year_dummies], axis=1)
year_cols = [col for col in data if col.startswith('y_')]

# Create village dummies:
vill_dummies = pd.get_dummies(data['villageid'], drop_first=True, prefix="vill", dtype=int)
data = pd.concat([data, vill_dummies], axis=1)
vill_cols = [col for col in data if col.startswith('vill_')]

# Create x and y variables:
#x_cols1 = ["male","age","hhchildren","hhworkforce","ag","nonag","yrs_worked_in_mx_cum","MODELING_GROUP"]
x_cols1 = ["age","yrs_worked_in_mx_cum","l1_local_ag","l2_local_ag","l3_local_ag","l1_local_nonag","l2_local_nonag","l3_local_nonag","avgtemp5","precip_tot5","GDD5","HDD5","avgtemp6","precip_tot6","GDD6","HDD6","avgtemp7","precip_tot7","GDD7","HDD7","avgtemp8","precip_tot8","GDD8","HDD8","precip_tot","bint_LT12","bint_GT32","HDD","GDD","GDD_30C","GDD_34C","HDD_30C","HDD_34C","GDDwithin","HDDwithin","avgtemp","bint_12","bint_14","bint_16","bint_18","bint_20","bint_22","bint_24","bint_26","bint_28","bint_30","precip_tot_MDagseason","HDD_MDagseason","GDD_MDagseason","precip_tot_nonagseason","HDD_nonagseason","GDD_nonagseason","precip_totsq","precip_totsq_MDagseason","precip_totsq_nonagseason","distkm","avtimeloc02","MODELING_GROUP"]
x_cols = x_cols1 + year_cols + vill_cols
y_cols = ["work_us", "MODELING_GROUP"]
x = data[x_cols]
y = data[y_cols]

# Split the data into a training set and a test set:
x_train = x.loc[x['MODELING_GROUP'] == 'TRAINING']
y_train = y.loc[y['MODELING_GROUP'] == 'TRAINING']
x_test = x.loc[x['MODELING_GROUP'] == 'TESTING']
y_test = y.loc[y['MODELING_GROUP'] == 'TESTING']

x_train = x_train.drop('MODELING_GROUP', axis=1)
y_train = y_train.drop('MODELING_GROUP', axis=1)
x_test = x_test.drop('MODELING_GROUP', axis=1)
y_test = y_test.drop('MODELING_GROUP', axis=1)

# Replace missing with zero:
x_train = x_train.fillna(0)
y_train = y_train.fillna(0)
x_test = x_test.fillna(0)
y_test = y_test.fillna(0)

# Summarize class distribution in training data:
counter = Counter(y_train['work_us'])
print(counter)

# Transform the dataset using SMOTE:
oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

# Summarize class distribution after transformation:
counter = Counter(y_train['work_us'])
print(counter)

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
# exit()
# Best parameters are: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}

# Train the AI model on the training set:
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=50, random_state=16)
model.fit(x_train, y_train.values.ravel())

# Evaluate the AI model on the test set:
y_pred = model.predict(x_test)

# Create classification report:
target_names = ["Didn't work in the US", "Worked in the US"]
cr = metrics.classification_report(y_test, y_pred, target_names=target_names)

# Create Confusion matrix:
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Write to txt:
cm = np.array2string(cnf_matrix)
f = open('output/report_gradboost_weather.txt', 'w')
f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
f.close()

# Create heatmap:
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('output/confusion_matrix_gradboost_weather.png', bbox_inches='tight')
