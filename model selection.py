# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:40:47 2020

@author: liu
"""

from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from utils import Config
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

# load data
data = pd.read_csv(Config['feature_path'])
feature1 = data[Config['selected_features']].sample(frac=1)
label = feature1['Class']
groups = feature1['user']
feature = feature1.drop(columns=['Class','user'])
gss = LeaveOneGroupOut()
x = gss.split(feature, label, groups=groups)


#optimaze linear kernel SVM
##param_grid = [
##  {'C': np.logspace(-1, 1, num=10), 'kernel': ['linear']},
##  {'C': np.logspace(-1, 1, num=10), 'gamma': np.logspace(-1, 1, num=10), 'kernel': ['rbf']},
## ]
#parameters = {'C': np.logspace(-1, 1, num=10)}
#svc = SVC(kernel='linear')
#clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=x, return_train_score=True)
#clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=x, return_train_score=True, scoring = 'f1_micro')
#clf.fit(feature, label)
#cv_result = pd.DataFrame(clf.cv_results_)

# optimize rgf kernel SVM

#data = pd.read_csv(Config['feature_path'])
#feature1 = data[['Class', 'user', 'ydev', 'zrange', 'zmax','ymax', 'missing']].sample(frac=1)
#label = feature1['Class']
#groups = feature1['user']
#feature = feature1.drop(columns=['Class','user'])
#gss = LeaveOneGroupOut()
#x = gss.split(feature, label, groups=groups)
#
#parameters = {'gamma':np.logspace(-1, 1, num=10), 'C':np.logspace(-1, 1, num=10)}
#svc = SVC(kernel='rbf')
#clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=x, return_train_score=True)
#clf.fit(feature, label)
#cv_result = pd.DataFrame(clf.cv_results_)
#cv_result.to_csv('result.csv', index=False)


df = pd.read_csv('result(1).csv')
accuracy = df.mean_test_score.to_numpy()
accuracy = accuracy.reshape((10,10))
fig, ax = plt.subplots()
im = ax.imshow(accuracy)
#label = [round(x, 2) for x in np.logspace(-1, 1, num=10)]
#ax.set_xticklabels(label)
#ax.set_yticklabels(label)
for i in range(10):
    for j in range(10):
        acc = round(accuracy[i, j],2)
        text = ax.text(j, i, acc,
                       ha="center", va="center", color="w")
ax.set_xlabel('gamma')
ax.set_ylabel('C')
ax.set_title('acc')