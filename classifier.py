from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def knn(x_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    return neigh.predict(x_test)
