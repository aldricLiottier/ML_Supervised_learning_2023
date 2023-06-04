import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load X inputs and y labels from .npy files
X = np.load('./ex_3_classification_NBA/inputs.npy')
Y = np.load('./ex_3_classification_NBA/labels.npy')

Y_1d = np.ravel(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y_1d, test_size=0.2, random_state=42)

model = SVC()
model1 = KNeighborsClassifier(n_neighbors=7)
model2 = LinearRegression()
model3 = RandomForestClassifier(500)
model4 = RandomForestClassifier(50)

model.fit(X_train, y_train)
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Accuracy support Vector machine:", accuracy)
accuracy1 = model1.score(X_test, y_test)
print("Accuracy K nearest neighbor:", accuracy1)

accuracy2 = model2.score(X_test, y_test)
print("Accuracy Linear Regression:", accuracy2)

accuracy3 = model3.score(X_test, y_test)
print("Accuracy random forest 500 depth:", accuracy3)

accuracy4 = model4.score(X_test, y_test)
print("Accuracy random forest 50 depth:", accuracy4)
