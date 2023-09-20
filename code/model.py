####################################################################################################
## Description: Machine learning program to predict migration in Mexico
## Author: Parth Chawla
## Date: Sep 2, 2023
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

path = '/Users/parthchawla1/GitHub/ml-predictmigration/'
os.chdir(path)

dtafile = 'data/MexMigData.dta'
data, meta = pyreadstat.read_dta(dtafile)

cols_to_move = ['ind', 'year']
data = data[cols_to_move + [col for col in data.columns if col not in cols_to_move]] # cols to front
data = data.sort_values(by=['ind', 'year'], ascending=[True, True]) # sort

data['work_in_mx'] = np.where((data['work_loc']==1) | (data['work_mx']==1), 1, 0) # worked in mx in year t
data['yrs_worked_in_mx_cum'] = data.groupby(['ind'])['work_in_mx'].cumsum() # years worked in mx till t
print(data.head())

# We want training and testing data to not have the same people

# Get unique list of individuals:
data_copy = data
ind = data_copy.drop_duplicates(subset='ind')
ind = ind[['ind']]

# Create a new variable with a random number between 0 and [no. of unique IDs]:
np.random.seed(42)
ind['rannum'] = (np.random.randint(0, 10000, ind.shape[0]))/10000
ind = ind[['ind', 'rannum']]

# Give each ind a P% chance of being in training and (100-P)% chance of being in test:
ind['MODELING_GROUP'] = np.where((ind.rannum <= 0.75), 'TRAINING', 'TESTING')

# Merge modeling group with data:
data = data.sort_values(by=['ind'], ascending=[True])
ind = ind.sort_values(by=['ind'], ascending=[True])
data = data.merge(ind, on=['ind'], how='inner')
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
x_cols1 = ["male","age","hhchildren","hhworkforce","ag","nonag","yrs_worked_in_mx_cum","MODELING_GROUP"]
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

# Train the AI model on the training set:
# model = LogisticRegression(max_iter=10000, random_state=16)
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=50, random_state=16)
model.fit(x_train, y_train.values.ravel())

# coefs = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(model.coef_))], axis = 1)
# coefs.to_csv('output/coefs.csv')

# Evaluate the AI model on the test set:
y_pred = model.predict(x_test)

# Create classification report:
target_names = ["Didn't work in the US", "Worked in the US"]
cr = metrics.classification_report(y_test, y_pred, target_names=target_names)

# Create Confusion matrix:
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Write to txt:
cm = np.array2string(cnf_matrix)
f = open('output/report.txt', 'w')
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
plt.savefig('output/confusion_matrix.png', bbox_inches='tight')
