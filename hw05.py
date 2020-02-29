# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:05:33 2020

@author: liu
"""
import numpy as np
import matplotlib.pyplot as plt
w_0 = np.array([[-1] * 5, [1] * 5, [0] * 5])
## s1, s2, s2, s3
x = np.array([[1, 0, 1, -1, 2], [1] * 5, [1, 2, 1, 1, 1], [1, -1, 1, 0, -1]])
label = np.array([0,1,1,2])
def multi_class_perceptron(x, w_0, label):
    w = np.copy(w_0)
    F = 1 # if there is no update
    for i in range(len(x)):
        print(w)
        result = []
        for j in range(3):
            result.append(w[j] @ x[i])
        max_index = np.where(result == max(result))
        print('The maximum value is function of class {0}'.format(max_index[0][0] + 1))
        print('The correct label is {0}'.format(label[i] + 1))
        w0, w1, w2 = update(x, w, label[i], max_index[0][0], i)
        if not (np.array_equal(w0, w_0[0]) and np.array_equal(w1, w_0[1]) and np.array_equal(w2, w_0[2])):
            F = 0
    if F == 1:
        return w
    else:
        return multi_class_perceptron(x, w, label)
    
    
def update(x, w, i, j, k):
#    kth data point, the correct class label is i, the maximum value is wj @ x[k]
    w[i] = w[i] + x[k]
    w[j] = w[j] - x[k]
    print('\n')
    return w[0], w[1], w[2]



w_n = multi_class_perceptron(x, w_0, label)


# project on to 2-D
x_t = np.copy(x)
x_t[:,3] = 0
x_t[:,4] = 0
value = x_t @ w_n.T













