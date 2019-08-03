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
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\train_values.csv", index_col=0)
y = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\train_labels.csv", index_col=0)

X = pd.get_dummies(X, drop_first=True)

seed = 42

scale = StandardScaler()
logreg = LogisticRegression(C=0.4393970560760795,random_state=seed)
pca = PCA(n_components=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = seed)

c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

pipe_log = Pipeline(steps=[('scaler', scale), ('principal components', pca), ('logistic', logreg)])

pipe_log.fit(X_train, y_train)

y_pred = pipe_log.predict(X_test)

y_pred_prob = pipe_log.predict_proba(X_test)

print('Accuracy : {:.3f} , loss log : {:.3f}'.format(accuracy_score(y_test, y_pred), log_loss(y_test, y_pred_prob)))

