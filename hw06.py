# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:45:24 2020

@author: liu
"""
#tolfloat, default=1e-3 The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol).


import csv
import numpy as np
from sklearn.linear_model import Perceptron as P
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.metrics import accuracy_score as acc


raw_data = np.zeros(14)
with open('wine_train.csv', newline = '') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        for j in range(len(row)):
            row[j] = float(row[j])
        raw_data = np.vstack((raw_data, row))
raw_data = raw_data[1:]
# reguliarize
data_mean = np.mean(raw_data, axis = 0)
print(data_mean)
print('mean')
data_var = np.std(raw_data, axis = 0)
print(data_var)
print('var')
data = np.copy(raw_data)
for i in range(89):
    data[i,: -1] = (raw_data[i, : -1] - data_mean[: -1]) / data_var[: -1]

# build and fit model
f1_f2, label = data[:,:2], data[:,-1]
clf = P()
clf.fit(f1_f2, label)
weights = clf.coef_
print('The weights of 2 features traing model is {0}'.format(weights))
# pred and calculate acc rate
pred_label = clf.predict(f1_f2)     
acc_rate = acc(data[:,-1], pred_label)
print('The training acc using 2 feature is {0}'.format(acc_rate))

# get test data set
raw_test_data = np.zeros(14)
with open('wine_test.csv', newline = '') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        for j in range(14):
            row[j] = float(row[j])
        raw_test_data = np.vstack((raw_test_data, row))
raw_test_data = raw_test_data[1:]
# regularize test data
test_data = np.copy(raw_test_data)
for i in range(89):
    test_data[i,: -1] = (raw_test_data[i, : -1] - data_mean[: -1]) / data_var[: -1]
pred_test_label = clf.predict(test_data[:, :2])
acc_test_rate = acc(test_data[:,-1], pred_test_label)
print('The testing acc using 2 feature is {0}'.format(acc_test_rate))

# use 13 features
clf13 = P()
clf13.fit(data[:,: -1], data[:,-1])
weights = clf13.coef_
print('The weights of 13 features traing model is {0}'.format(weights))

#calculate acc rate, 13 feature
pred_13_label = clf13.predict(data[:,: -1])
acc_13_rate = acc(data[:,-1], pred_13_label)
print('The training acc using 13 features is {0}'.format(acc_13_rate))
pred_13_test_label = clf13.predict(test_data[:,: -1])
acc_13_test_rate = acc(test_data[:,-1], pred_13_test_label)
print('The testing acc using 13 features is {0}'.format(acc_13_test_rate))


print('\n')
print('random starting weights')

# for two features
acc_save = 0
test_acc_save = 0
# Perceptron use one vs rest by default ref:https://scikit-learn.org/stable/modules/multiclass.html
# so the code below wont't use OvsR and inherit Perceptron, will user Perceptron directly   
weights = np.zeros((3,2))     
intercept = np.zeros(3) 
clf = P()
for _ in range(100):
    init_w = np.random.rand(3,2)
    init_b = np.random.rand(3)
    clf.fit(data[:, :2], data[:,-1], init_w, init_b)
    pred = clf.predict(data[:, :2])
    acc_rate = acc(data[:,-1], pred)
    test_pred = clf.predict(test_data[:, :2])
    test_acc = acc(test_data[:,-1], test_pred)
    if acc_save < acc_rate:
        acc_save = acc_rate
        test_acc_save = test_acc
        weights = clf.coef_
        intercept = clf.intercept_
#    print('acc rate is {0}'.format(acc_rate))
        
        
print('the weighte vector is {0}'.format(weights))
print('the intercept is {0}'.format(intercept))

print('best performance on the training set, 2 feature is {0}'.format(acc_save))
print('Performance on the test set, 2 feature is {0}'.format(test_acc_save))
print('\n')

# for 13 features

acc_save = 0
test_acc_save = 0
# Perceptron use one vs rest by default ref:https://scikit-learn.org/stable/modules/multiclass.html
# so the code below wont't use OvsR and inherit Perceptron, will user Perceptron directly   
      
clf = P()
acc_save = 0
test_acc_save = 0
weights = np.zeros((3,13))
intercept = np.zeros(3)
for _ in range(100):
    init_w = np.random.rand(3,13)
    init_b = np.random.rand(3)
    clf.fit(data[:, :-1], data[:,-1], init_w, init_b)
    pred = clf.predict(data[:, :-1])
    acc_rate = acc(data[:,-1], pred)
    test_pred = clf.predict(test_data[:, :-1])
    test_acc = acc(test_data[:,-1], test_pred)
    if acc_save < acc_rate:
        acc_save = acc_rate
        test_acc_save = test_acc
        test_acc_save = test_acc
        weights = clf.coef_
        intercept = clf.intercept_
#    print('acc rate is {0}'.format(acc_rate))
        
        

print('the weighte vector is {0}'.format(clf.coef_))
print('the intercept is {0}'.format(clf.intercept_))
print('best performance on the training set, 13 feature is {0}'.format(acc_save))
print('Performance on the test set, 13 feature is {0}'.format(test_acc_save))
print('\n')




#g
# row data is unnormalized data, data is standerd data
from sklearn.linear_model import LinearRegression as LR
class MSE_binary(LR):
    def __init__(self):
#        print('Call newly created MSE binary function')
        super(MSE_binary, self).__init__()
    def predict(self, X):
#        print('Call newly created MSE binary function')
        thr = 0.5
        y = self._decision_function(X)
        pred = np.zeros(y.shape)
        pred[y>thr] = 1
        return pred



binary_model = MSE_binary()
clf = OVR(binary_model)

def report_acc(num_feature):
    clf.fit(raw_data[:, :num_feature], raw_data[:,-1])
    pred = clf.predict(raw_data[:, :num_feature])
    acc_rate = acc(raw_data[:,-1], pred)
    print('The acc rate for training data({1}, raw data) is {0}'.format(acc_rate, num_feature))
    test_pred = clf.predict(raw_test_data[:, :num_feature])
    acc_rate = acc(raw_test_data[:,-1], test_pred)
    print('The acc rate for testing data({1}, raw data) is {0}'.format(acc_rate,num_feature))
    
    
    clf.fit(data[:, :num_feature], data[:,-1])
    pred = clf.predict(data[:, :num_feature])
    acc_rate = acc(data[:,-1], pred)
    print('The acc rate for training data({1}, standard) is {0}'.format(acc_rate,num_feature))
    test_pred = clf.predict(test_data[:, :num_feature])
    acc_rate = acc(test_data[:,-1], test_pred)
    print('The acc rate for testing data({1}, standard) is {0}'.format(acc_rate, num_feature))


report_acc(2)
report_acc(13)








#
#
