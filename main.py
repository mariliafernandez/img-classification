import itertools
import classifier as cl
import pandas as pd
import numpy as np
from utils import split_classes, split_train_test, print_size, print_size_smote
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


data = pd.read_csv('deepFeatures.csv')

x = data.T[1:].T 
y = data.T[0:1].T # Target: first row

x_list = []
y_list = []

x_list, y_list = split_classes(x, y)
x_train, x_test, y_train, y_test = split_train_test(x_list, y_list)

x_train_all = pd.concat(x_train)
y_train_all = pd.concat(y_train)

print('Imbalanced dataset')
print_size(y_train_all)

smt = SMOTE()
x_smote, y_smote = smt.fit_sample(x_train_all, np.ravel(y_train_all, order='C'))

print('\nOversampled dataset')
print_size_smote(y_smote)
# print('accuracy (imbalanced): ', accuracy_score(y_test, cl.knn(x_train, y_train))
# print('accuracy (balanced): ', accuracy_score(y_test_smote, cl.knn(x_train_smote, y_train_smote))
