import balance_dataset as bd
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('deepFeatures.csv')
x = data.T[1:].T 
y = data.T[0:1].T # Target: first row

# Spliting the data into train and test set Dataframes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
bd.print_imbalanced(y_train)

# Overbalancing dataset with SMOTE
x_train_smote, y_train_smote = bd.smote(x_train, y_train)
bd.print_balanced(y_train_smote)