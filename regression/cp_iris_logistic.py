import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg = LogisticRegression(C = 99999).fit(X_train, Y_train)

# Evaluate the model
print("Training scores: {:.2f}".format(logreg.score(X_train, Y_train)))
print("Test scores: {:.2f}".format(logreg.score(X_test,Y_test)))
