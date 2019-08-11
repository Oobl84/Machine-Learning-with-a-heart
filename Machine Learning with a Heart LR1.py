# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:40:38 2019

@author: ruban
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\train_values.csv", index_col=0)
y = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\train_labels.csv", index_col=0)

X = pd.get_dummies(X, drop_first=True)

sns.set()

seed = 42

scale = MinMaxScaler()
logreg = LogisticRegression(C=0.432, penalty='l1',random_state=seed)
pca = PCA(n_components=7)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = seed)

pipe_log = Pipeline(steps=[('scaler', scale), ('pca', pca), ('logistic', logreg)])

pipe_log.fit(X_train, y_train)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

y_pred = pipe_log.predict(X_test)

y_pred_prob = pipe_log.predict_proba(X_test)[:, 1]


print('Accuracy : {:.3f} , loss log : {:.3f}'.format(accuracy_score(y_test, y_pred), log_loss(y_test, y_pred_prob)))

test_values = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\test_values.csv", index_col=0)

test_values = pd.get_dummies(test_values, drop_first=True)

submission_format = pd.read_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\submission_format.csv", index_col=0)

predictions = pipe_log.predict_proba(test_values)[:, 1]


#my_submission_7 = pd.DataFrame(data=predictions, columns = submission_format.columns, index=submission_format.index)

#my_submission_7.to_csv(r"C:\Users\ruban\OneDrive\Documents\Python Notes\Machine Learning with a Heart\my_submission_7.csv")
