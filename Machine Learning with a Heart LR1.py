# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:40:38 2019

@author: ruban
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\train_values.csv", index_col=0)
y = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\train_labels.csv", index_col=0)

X = pd.get_dummies(X, drop_first=True)

scale = StandardScaler()
logreg = LogisticRegression()
pca = PCA(n_components=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 21)

pipe = Pipeline(steps=[('scaler', scale),('components', pca), ('logistic', logreg)])

param_grid = {'logistic__C': [0.0001, 0.001, 0.01, 1, 10], 
              'logistic__penalty': ['l1', 'l2']}

gs = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)

gs.fit(X_train, y_train.values.reshape(-1,))

y_pred = gs.predict(X_test)

y_score = accuracy_score(y_test, y_pred)

y_pred_prob = gs.predict_proba(X_test)[ :, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, linestyle='solid')
plt.show()

print("Logistic Regression : {:.3f} , Log Loss : {:.3f}".format(y_score, log_loss(y_test, y_pred_prob)))
