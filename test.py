# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:04:33 2020

@author: liu
"""
import pandas as pd
from utils import Config
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, cross_val_score
from sklearn.metrics import classification_report as clfrepo


# training feature
df = pd.read_csv(Config['feature_path'])
feature1 = df[Config['selected_features']].sample(frac=1)
label = feature1['Class']
feature = feature1.drop(columns=['Class','user'])


# test feature
dftest = pd.read_csv(Config['test_path'])
test = dftest[Config['selected_features']]
tfeature = test[Config['selected_features']].drop(columns=['Class','user'])
tlabel = test['Class']

clf = SVC(gamma=2.15, C=0.774)
#clf = SVC(kernel = 'linear', C=1)
clf.fit(X=feature, y=label)

tscore = clf.score(tfeature, tlabel)

print()
print('test data set score is:')
print(tscore)
tprediction = clf.predict(tfeature)
tcategory_report = clfrepo(tlabel, tprediction)
