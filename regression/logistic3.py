import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# Split the data
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, confusion_matrix,  f1_score, precision_score, recall_score, roc_curve
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(random_state = 0, solver = "liblinear")
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

df = pd.DataFrame({'actual': y_test, 'predicted': y_pred}, columns=['actual', 'predicted'])
confusion_matrix = pd.crosstab(df['actual'], df['predicted'], rownames=['actual'], colnames=['predicted'])
print(confusion_matrix)
