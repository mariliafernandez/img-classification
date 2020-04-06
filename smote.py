from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

data = pd.read_csv('deepFeatures.csv')
x = data.T[1:].T 
y = data.T[0:1].T # Target: first row

# Spliting the data into train and test set Dataframes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Column identifier
col = y.columns.tolist()[0]

print('\nNumber of data in imbalanced dataset:')
for i in range(7):
    print('Group #', i,': ', len(y_train[y_train[col] == i]))

smt = SMOTE()
# x_train and y_train are now arrays
x_train, y_train = smt.fit_sample(x_train,  np.ravel(y_train,order='C'))

print('\nNumber of data in balanced dataset using SMOTE:')
for i in range(7):
    print('Group #', i,': ', len(y_train[y_train == i])) 