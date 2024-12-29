import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  

iris = pd.read_csv("Iris.csv")

iris = iris.drop('Id',axis=1)
x = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20) 

no_neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)



plt.figure(figsize=(10, 6))  
plt.plot(no_neighbors, test_accuracy, marker='o', linestyle='-', color='blue', label='Testing Accuracy', linewidth=2)
plt.plot(no_neighbors, train_accuracy, marker='s', linestyle='--', color='green', label='Training Accuracy', linewidth=2)

plt.title('k-NN: Accuracy vs Number of Neighbors', fontsize=16, fontweight='bold')
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)
plt.show()