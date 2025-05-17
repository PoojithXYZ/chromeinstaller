import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train
y_train

from sklearn.ensemble import AdaBoostClassifier


from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=1)

ada_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=1.0, random_state=42)
ada_clf.fit(X_train, y_train)


y_pred = ada_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of AdaBoost classifier: {accuracy * 100:.2f}%")

