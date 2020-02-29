# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:56:24 2020

@author: liu
"""
# =============================================================================
# problem 01
# =============================================================================
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import scipy.io

# reflect data points of class 2
def reflect(feature, label):
    if label == 2:
        return -1 * feature
    return feature

# w(i+1) = w(i) + zn*xn -- perceptron GD
def update(w, data, F):
    if w @ data <= 0:
        F = 0
        w = w + data
    return w, F

# ruffle, use random data point order in every epoch
def ruffle(n):
    index = []
    for _ in range(n):
        index.append(random.randrange(0, n))
    return index

# design augment space
def augment(feature):
    l_one = np.ones((len(feature), 1))
    return np.hstack((feature, l_one))

# the total error over all training data when parameter is w
def error(w, feature, label):
    total = 0
    for i in range(len(label)):
        data = reflect(feature[i], label[i])
        if w @ data < 0:
            total += w @ data / np.linalg.norm(w)
    return total         
            
def perceptron(feature, label, m=1000):
    i = 0
    w = [0.1, 0.1, 0.1]
    while i < m - 1:
        F = 1
        index = ruffle(len(label))
        for j in index:
            data = reflect(feature[j], label[j])
            w, F = update(w, data, F)
        if F:
            break
        i += 1
    if i == m-1:
        err = []
        index = ruffle(len(label))
        for j in index:
            data = reflect(feature[j], label[j])
            w = update(w, data)
            cost = error(w, feature, label)
            err.append(cost)
        w = min(err)
    return w

#plot boundary and data points
def plotboundary(training, label_train, w):
    
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1
    x_plot = np.arange(min_x, max_x, 0.01)
    plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'rx')
    plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'go')
    l = plt.legend(('Class 1', 'Class 2'), loc=2)
    plt.plot( x_plot, (-1 * w[0] *  x_plot - w[2]) / w[1], c='b', label = 'boundary' )
#    plt.legend()
    axes = plt.gca()
    axes.add_artist(l)
    axes.legend(loc = 4)
    axes.set_ylim([min_y, max_y])    
    axes.set_xlim([min_x, max_x])
    plt.show()

#calculate the classification error rate
def err_rate(w, feature, label):
    j = 0
    for i in range(len(label)):
        if w @ feature[i] <= 0 and label[i]==1:
            j += 1
        elif w @ feature[i] >= 0 and label[i]==2:
            j += 1
    result = j/len(label)
    return round(result, 4) *100

def extract_data(filename):
    feature1, feature2, label = [],[],[]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            feature1 = np.append(feature1, float(row[0]))
            feature2 = np.append(feature2, float(row[1]))
            label = np.append(label, float(row[2]))
    return feature1, feature2, label

# classify training data 1
feature1, feature2, label = extract_data('synthetic1_train.csv')
feature = np.vstack((feature1, feature2))
feature = augment(feature.T)    
w = perceptron(feature, label)
tr_err = err_rate(w, feature, label)
print('The error rate of training-data-1 is {}%'.format(tr_err))
f = feature[:, 0:2]
plotboundary(f, label, w)

# test data 1
feature1, feature2, label = extract_data('synthetic1_test.csv')
feature = np.vstack((feature1, feature2))
feature = augment(feature.T)
tr_err = err_rate(w, feature, label)
print('The error rate of testing-data-1 is {}%'.format(tr_err))
f = feature[:, 0:2]
plotboundary(f, label, w)
#


# get data set from synthetic2.csv
feature1, feature2, label = extract_data('synthetic2_train.csv')
feature = np.vstack((feature1, feature2))
feature = augment(feature.T)    
w = perceptron(feature, label)
tr_err = err_rate(w, feature, label)
print('The error rate of training-data-2 is {}%'.format(tr_err))
f = feature[:, 0:2]
plotboundary(f, label, w)

#test data 2
feature1, feature2, label = extract_data('synthetic2_test.csv')
feature = np.vstack((feature1, feature2))
feature = augment(feature.T) 
tr_err = err_rate(w, feature, label)
print('The error rate of testing-data-2 is {}%'.format(tr_err))
f = feature[:, 0:2]
plotboundary(f, label, w)



# get data set from synthetic3.csv
mat = scipy.io.loadmat('synthetic3.mat')
feature = mat['feature_train']
label1 = mat['label_train']
label = []
for i in range(len(label1)):
    label.append(label1[i][0])
label = np.array(label)
feature = augment(feature)    
w = perceptron(feature, label)
tr_err = err_rate(w, feature, label)
print('The error rate of training-data-3 is {}%'.format(tr_err))
f = feature[:, 0:2]
plotboundary(f, label, w)
#
#test data 2
feature = mat['feature_test']
label1 = mat['label_test']
label = []
for i in range(len(label1)):
    label.append(label1[i][0])
label = np.array(label)
feature = augment(feature) 
tr_err = err_rate(w, feature, label)
print('The error rate of testing-data-3 is {}%'.format(tr_err))
f = feature[:, 0:2]
plotboundary(f, label, w)
