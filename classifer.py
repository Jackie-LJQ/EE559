# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import LeaveOneGroupOut, cross_validate, cross_val_score
from sklearn.metrics import classification_report as clfrepo
import pandas as pd
import numpy as np
from utils import Config

#accuracy if random predict 19.59 match intuition
df = pd.read_csv(Config['feature_path'])
#guess = np.random.randint(1,6,size = 13500)
#temp = np.equal(guess, df['Class'])
#guess_acc = temp.sum()/13500*100


datadf = df[Config['selected_features']]

## oigin class in order, shuffle it
if Config['debug']:
    data = datadf.sample(10)
else:
    data = datadf.sample(frac=1)
#    
feature1 = data[Config['selected_features']]
label = data['Class']
groups = data['user']
feature = feature1.drop(columns=['Class','user'])
gss = LeaveOneGroupOut()
x = gss.split(feature, label, groups=groups)

clf = Config['model']
score = cross_val_score(clf, X=feature, y=label, cv = x)
print('the average of c.v. score is:')
print(score.mean())
print()
print('the standard deviation of c.v. score is:')
print(np.std(score))


feature1 = data[Config['selected_features']]
label = data['Class']
groups = data['user']
feature = feature1.drop(columns=['Class','user'])
gss = LeaveOneGroupOut()
x = gss.split(feature, label, groups=groups)

clf = Config['model']
score = cross_val_score(clf, X=feature, y=label, cv = x, scoring='f1_micro')
print('the average of f1 score is:')
print(score.mean())
print()
print('the standard deviation of f1 score is:')
print(np.std(score))

