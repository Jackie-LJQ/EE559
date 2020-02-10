# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:59:03 2020

@author: liu
"""
# =============================================================================
# problem 1
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
x1 = np.linspace(-8, 8, 1000)
x2_1 = [i*-1+5 for i in x1]
x2_3 = [i+1 for i in x1]
fig, ax = plt.subplots()
ax.plot(x1, x2_1, color = 'b', linewidth=0.01)
ax.plot(x1, x2_3, color = 'b', linewidth=0.01)
ax.axvline(x=3, color = 'b', linewidth=0.01)
ax.fill_between(x1, x2_3[0], 15, facecolor = 'white', label = 'indeterminate')
ax.fill_between(x1,x2_3, 15, facecolor = 'purple', label = 'class 2')
ax.fill_between(x1, x2_3[0], x2_1, where=x1 < 3, facecolor = 'blue', label = 'class 1')
ax.fill_between(x1, x2_3[0], x2_3, where=x1 >= 3, facecolor = 'green', label = 'class 3')
ax.legend(loc = 'upper left',bbox_to_anchor=(0.03, 0.98))
plt.show()
def classfy(x):
    x1, x2 = x[0], x[1]
    g_12 = -x1 - x2 + 5
    g_13 = -x1 + 3 
    g_23 = -x1 + x2 - 3
    if g_12 > 0 and g_13 > 0:
        return 1
    if g_23 > 0 and g_12 < 0:
        return 2
    if g_13 < 0 and g_23 < 0:
        return 3
    return 'indeterminate'
x = [4,1]
result = classfy(x)
print('The point ({0},{1}) belongs to class{2}.'.format(x[0], x[1], result))        
x = [1,5]
result = classfy(x)
print('The point ({0},{1}) belongs to class{2}.'.format(x[0], x[1], result)) 
x = [0,0]
result = classfy(x)
print('The point ({0},{1}) belongs to class{2}.'.format(x[0], x[1], result)) 
x = [2.5,3]
result = classfy(x)
print('The point ({0},{1}) is {2}.'.format(x[0], x[1], result))     


# =============================================================================
# problem 2
# =============================================================================
import csv
from scipy.spatial.distance import cdist
fetr_1, fetr_2, label = [], [], []
with open('wine_train.csv', newline='') as csvfile:
    fieldnames=['feature{0}'.format(i) for i in range(1,14)] + ['decision']
    reader = csv.DictReader(csvfile,fieldnames)
    for row in reader:
        fetr_1.append(float(row['feature1']))
        fetr_2.append(float(row['feature2']))
        label.append(int(row['decision']))
#compute the sample mean of class n, feature i
def find_mean(n, i, feature, label):
#the sum of data set in class n and out of class n respectively
    sum_n, sum_not_n, k = 0, 0, 0
    for j in range(len(label)):
        if label[j] == n:
            sum_n += feature[j]
            k += 1
        else:
            sum_not_n += feature[j]
    mean_n, mean_not_n = sum_n/k, sum_not_n/(len(label)-k)
    return mean_n, mean_not_n
#compute different pairs of features for each class
c1 = np.zeros((2,2))
# the first row is the class_i's coordinate 
# the second row is the coordinate of data that not in class i
c1[0, 0], c1[1, 0] = find_mean(1,1,fetr_1,label)
c1[0, 1], c1[1, 1] = find_mean(1,2,fetr_2,label)
c2 = np.zeros((2,2))
c2[0, 0], c2[1, 0] = find_mean(2,1,fetr_1,label)
c2[0, 1], c2[1, 1] = find_mean(2,2,fetr_2,label)
c3 = np.zeros((2,2))
c3[0, 0], c3[1, 0] = find_mean(3,1,fetr_1,label)
c3[0, 1], c3[1, 1] = find_mean(3,2,fetr_2,label)
training = np.column_stack((fetr_1,fetr_2))
# create label of class i as 1 and the rest of class i as 2
def change_label(label, j):
    i = 0
    label_list = np.copy(label)
    while i < len(label):
        if label_list[i] == j:
            label_list[i] = 1
        else:
            label_list[i] = 2
        i+=1
    return label_list
label_1 = change_label(label, 1)
label_2 = change_label(label, 2)
label_3 = change_label(label, 3)
label = np.array(label)
sample_mean = np.row_stack((c1,c2,c3))
total = len(label_1)
# =============================================================================
# problem 2(b)
# =============================================================================
def myplotboundary_2(training, c_x, k):
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1
    xrange = (min_x, max_x)
    yrange = (min_y, max_y)
    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005
    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))
    
    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) 
    # make (x,y) pairs as a bunch of row vectors.
    
    dist_1_r = cdist(xy, c_x)
    pred_label_1 = np.argmin(dist_1_r, axis=1)
    dec = pred_label_1.reshape(image_size, order='F')
    plt.imshow(dec, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
#    plt.plot(training[label_x == 1, 0],training[label_x == 1, 1], 'rx')
#    plt.plot(training[label_x == 2, 0],training[label_x == 2, 1], 'go')
#    l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
#    plt.gca().add_artist(l)
    m1, = plt.plot(c_x[0,0], c_x[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(c_x[1,0], c_x[1,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    l1 = plt.legend([m1,m2],['class{0}'.format(k), 'rest{0}'.format(k)], loc='lower right')  
    plt.title('Class{0} and the rest'.format(k))
    plt.gca().add_artist(l1)
    plt.show()
myplotboundary_2(training, c1, 1)
myplotboundary_2(training, c2, 2)
myplotboundary_2(training, c3, 3)

# =============================================================================
# problem 2(c)
# =============================================================================
def myplotboundary(training, c1, c2, c3):
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1
    xrange = (min_x, max_x)
    yrange = (min_y, max_y)
    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005
    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))
    
    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) 
    # make (x,y) pairs as a bunch of row vectors.
    
    dist_1_r = cdist(xy, c1)
    dist_2_r = cdist(xy, c2)
    dist_3_r = cdist(xy, c3)
    pred_label_1 = np.argmin(dist_1_r, axis=1)
    pred_label_2 = np.argmin(dist_2_r, axis=1)
    pred_label_3 = np.argmin(dist_3_r, axis=1)
    j = len(pred_label_1)
    pred_label = np.zeros(j).astype(int)
    for i in range(j):
        if not pred_label_1[i] and pred_label_2[i] and pred_label_3[i]:
            pred_label[i] = 1
        elif pred_label_1[i] and not pred_label_2[i] and pred_label_3[i]:
            pred_label[i] = 2
        elif pred_label_1[i] and pred_label_2[i] and not pred_label_3[i]:
            pred_label[i] = 3
    dec = pred_label.reshape(image_size, order='F')
    plt.imshow(dec, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
#    plt.plot(training[label == 1, 0],training[label == 1, 1], 'rx')
#    plt.plot(training[label == 2, 0],training[label == 2, 1], 'go')
#    plt.plot(training[label == 3, 0],training[label == 3, 1], 'b*')
#    l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
#    plt.gca().add_artist(l)
    m1, = plt.plot(c1[0,0], c1[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(c2[0,0], c2[0,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    m3, = plt.plot(c3[0,0], c3[0,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')
    l1 = plt.legend([m1,m2,m3],['Class 1', 'Class 2', 'Class 3'], loc='lower right')   
    plt.gca().add_artist(l1)
    plt.show()
myplotboundary(training, c1, c2, c3)
# =============================================================================
# problem 2 (a)
# =============================================================================
def distance(x,y):
        return (x[0]-y[0])**2 + (x[1]-y[1])**2
def accuracy(feature, mean):
    k = 0
    for i in range(total):
        dist = [distance(feature[i], mean[j]) for j in range(6)]
        if dist[0]<dist[1] and dist[2]>dist[3] and dist[4]>dist[5]:
            f = 1
        elif dist[0]>dist[1] and dist[2]<dist[3] and dist[4]>dist[5]:
            f = 2
        elif dist[0]>dist[1] and dist[2]>dist[3] and dist[4]<dist[5]:
            f = 3
        else:
            f = 4
        if label[i] == f:
            k+=1
    return k
total_right = accuracy(training, sample_mean)
accurate_rate = round(total_right/total, 5) *100
print('Classification accuracy on training set is {0}%'.format(accurate_rate))

# accuracy rate of testing data
fetr_1, fetr_2, label = [], [], []
with open('wine_test.csv', newline='') as csvfile:
    fieldnames=['feature{0}'.format(i) for i in range(1,14)] + ['decision']
    reader = csv.DictReader(csvfile,fieldnames)
    for row in reader:
        fetr_1.append(float(row['feature1']))
        fetr_2.append(float(row['feature2']))
        label.append(int(row['decision']))
testing = np.column_stack((fetr_1,fetr_2))
total_right = accuracy(testing, sample_mean)
accurate_rate = round(100*total_right/total, 3) 
print('Classification accuracy on testinging set is {0}%'.format(accurate_rate))
# =============================================================================
# problem3 (b)
# =============================================================================
sample_mean = np.array([[0,0],[-2,1]])
min_value, max_value = np.min(sample_mean), np.max(sample_mean) 
ploting = np.array([[min_value-1, min_value-1],[max_value+1, max_value+1]])
myplotboundary_2(ploting, sample_mean, 1)
# =============================================================================
# problem3 (d)
# =============================================================================
ploting = np.array([[min_value-1, min_value-1],[max_value+1, max_value+1]])
def mod_ploting(training, sample_mean):
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1
    xrange = (min_x, max_x)
    yrange = (min_y, max_y)
    inc = 0.005
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) )
    dist_mat = cdist(xy, sample_mean)
    pred_label = np.argmin(dist_mat, axis=1)
    decisionmap = pred_label.reshape(image_size, order='F')
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
    # plot the class mean vector.
    m1, = plt.plot(sample_mean[0,0], sample_mean[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(sample_mean[1,0], sample_mean[1,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    m3, = plt.plot(sample_mean[2,0], sample_mean[2,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')
    # include legend for class mean vector
    l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
    plt.gca().add_artist(l1)
    plt.show()
mean = np.array([[0, -2],[0, 1],[2, 0]])
mod_ploting(ploting, mean)