from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

data = pd.read_csv('deepFeatures.csv')
x = data.T[1:].T 
y = data.T[0:1].T # Target: first row
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2)


col = y.columns.tolist()[0]

print('\nNumbers of items in each group before SMOTE')
for i in range(7):
    print('Group #', i,': ', len(y[y[col] == i]))

smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train,  np.ravel(y_train,order='C'))

print('\n\nNumbers of items in each group after SMOTE')
for i in range(7):
    print('Group #', i,': ', len(y_train == i))
