from imblearn.over_sampling import SMOTE
import numpy as np

def print_imbalanced(y_train):
    # Dataframe column identifier
    col = y_train.columns.tolist()[0]
    print('\nNumber of data in imbalanced dataset:')
    for i in range(7):
        print('Group #', i,': ', len(y_train[y_train[col] == i]))

def print_balanced(y_train):
    print('\nNumber of data in balanced dataset using SMOTE:')
    for i in range(7):
        print('Group #', i,': ', len(y_train[y_train == i])) 

# Returns 2D array (x, y) of balanced train data
def smote(x_train, y_train):
    smt = SMOTE()
    return smt.fit_sample(x_train,  np.ravel(y_train,order='C'))
    


