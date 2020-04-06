import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def split_classes(x, y):
    x_split = []
    y_split = []

    col = y.columns.tolist()[0] # Get column index
    
    for i in range(7):
        x_split.append(x[y[col] == i])
        y_split.append(y[y[col] == i])
    
    return x_split, y_split

def split_train_test(x, y):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(7):
        x1, x2, y1, y2 = train_test_split(x[i], y[i], test_size=0.2)
        x_train.append(x1)
        x_test.append(x2)
        y_train.append(y1)
        y_test.append(y2)

    return x_train, x_test, y_train, y_test

def print_size(train):
    col = train.columns.tolist()[0]
    for i in range(7):
        print('Size of class ',i,': ',len(train[train[col] == i]))

def print_size_smote(train):
    for i in range(7):
        print('Size of class ', i, ': ', len(train[train == i]))
