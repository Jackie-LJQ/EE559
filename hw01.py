import csv
import numpy as np
from plotDecBoundaries import plotDecBoundaries

def find_feature_class_mean(file_name):
    # find the three variables(the features of data points, the class label and
    # the sample mean) which needed to pass in plotDecBoundaries
    #
    # file_name: the file_name of data point. the file type is .csv

    # read row data from csv file. fetr_1 and fetr_2 are the first and second feature
    # respectly. dec is the class label.
    fetr_1, fetr_2, label = [], [], []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile,fieldnames=['feature1', 'feature2','decision'])
        for row in reader:
            fetr_1.append(float(row['feature1']))
            fetr_2.append(float(row['feature2']))
            label.append(int(row['decision']))

    # combine two features of data points in one row, change type to numpy array
    feature = np.column_stack((fetr_1,fetr_2))
    label = np.array(label)

    # calculate the sample mean of class1's features and class2's features
    # cl1_f1 denotes the feature1 belongs to class1
    # cl1_f2 denotes the feature2 belongs to class1
    # cl2_f1 denotes the feature1 belongs to class2
    # cl2_f2 denotes the feature2 belongs to class2
    cl1_f1, cl1_f2, cl2_f1, cl2_f2 = [], [], [], []
    a, b, c, d = [], [], [], []
    i = 0
    while i < len(label):
        if label[i] == 1:
            cl1_f1.append(fetr_1[i])
            cl1_f2.append(fetr_2[i])
        if label[i] == 2:
            cl2_f1.append(fetr_1[i])
            cl2_f2.append(fetr_2[i])
        i += 1
    mean = np.array([[np.mean(cl1_f1), np.mean(cl1_f2)],
                     [np.mean(cl2_f1), np.mean(cl2_f2)]])
    return [feature, label, mean]

def error_rate(fetr, label, mean, d = 2):
    # calculate the error rate
    # fstr is the feature of data points
    # dec is the class label of corresponding data points
    # mean is the sample mean
    #
    def distance(point_1, point_2):
        # calculate the distance between two arbitrary data points
        return (point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2
    num_err = 0
    for i in range(len(label)):
        tr_distance = [distance(fetr[i], mean[j]) for j in range(d)]
        # regarding to the decision rule, the class of a data point should be the
        # same as the closest sample mean
        cl_label = tr_distance.index(min(tr_distance))
        if cl_label + 1 != label[i]:
            num_err += 1
    return round(num_err/len(label)*100,2)
# ******************************question(a)*************************************
# plot the decision boundaries, region and data points of synthetic1_train
[tr1_feature, tr1_label, tr1_mean] = find_feature_class_mean('synthetic1_train.csv')
plotDecBoundaries(tr1_feature, tr1_label, tr1_mean)
# the error rate on the trained set one(synthetic1_train)
error_tr1 = error_rate(tr1_feature, tr1_label, tr1_mean)

# read the feature and class label of test points
[test1_feature, test1_label, _] = find_feature_class_mean('synthetic1_test.csv')
plotDecBoundaries(test1_feature, test1_label, tr1_mean)
# calculate the error rate of test data coresponding to the trained data rule
error_test1 = error_rate(test1_feature, test1_label, tr1_mean)

# plot and calculate the error rate of data set 2
[tr2_feature, tr2_label, tr2_mean] = find_feature_class_mean('synthetic2_train.csv')
plotDecBoundaries(tr2_feature, tr2_label, tr2_mean)
error_tr2 = error_rate(tr2_feature, tr2_label, tr2_mean)
[test2_feature, test2_label, _] = find_feature_class_mean('synthetic2_test.csv')
error_test2 = error_rate(test2_feature, test2_label, tr2_mean)

# ******************************question(c)*************************************
def find_feature_class_mean_3(file_name, f_pair):
    # find the three variables which needed to pass in plotDecBoundaries when
    # there are 3 classes in data set
    fetr_1, fetr_2, label = [], [], []
    with open(file_name, newline='') as csvfile:
        fieldnames=['feature{0}'.format(i) for i in range(1,14)] + ['decision']
        reader = csv.DictReader(csvfile,fieldnames)
        for row in reader:
            fetr_1.append(float(row['feature'+str(f_pair[0])]))
            fetr_2.append(float(row['feature'+str(f_pair[1])]))
            label.append(int(row['decision']))
    feature = np.column_stack((fetr_1,fetr_2))
    label = np.array(label)
    cl1_f1, cl1_f2, cl2_f1, cl2_f2, cl3_f1, cl3_f2 = [], [], [], [], [], []
    a, b, c, d = [], [], [], []
    i = 0
    while i < len(label):
        if label[i] == 1:
            cl1_f1.append(fetr_1[i])
            cl1_f2.append(fetr_2[i])
        if label[i] == 2:
            cl2_f1.append(fetr_1[i])
            cl2_f2.append(fetr_2[i])
        if label[i] == 3:
            cl3_f1.append(fetr_1[i])
            cl3_f2.append(fetr_2[i])
        i += 1
    mean = np.array([[np.mean(cl1_f1), np.mean(cl1_f2)],
                     [np.mean(cl2_f1), np.mean(cl2_f2)],
                     [np.mean(cl3_f1), np.mean(cl3_f2)]])
    return [feature, label, mean]


#
[w_tr_feature, w_tr_label, w_tr_mean] = find_feature_class_mean_3('wine_train.csv',[1,2])
plotDecBoundaries(w_tr_feature, w_tr_label, w_tr_mean)
error_wine_tr = error_rate(w_tr_feature, w_tr_label, w_tr_mean, 3)
[w_t_feature, w_t_label, _] = find_feature_class_mean_3('wine_test.csv',[1,2])
plotDecBoundaries(w_t_feature, w_t_label, w_tr_mean)
error_wine_t = error_rate(w_t_feature, w_t_label, w_tr_mean, 3)

# ******************************question(d)*************************************
# choose every pair of features, find the minimum error rate
# use total_err record all the error rate of choosing different pair of features
error, f1, f2 = 100, 0, 0
total_err_tr = []
for i in range(1,13):
    for j in range(i+1,14):
        [w_feature, wlabel, w_mean] = find_feature_class_mean_3('wine_train.csv',[i,j])
        error_i_j = error_rate(w_feature, wlabel, w_mean, 3)
        total_err_tr.append(error_i_j)
        if error > error_i_j:
            error, f1, f2 = error_i_j, i, j
# f1 and f2 are chosen features
[min_tr_feature, min_tr_label, min_tr_mean] = find_feature_class_mean_3('wine_train.csv',[f1,f2])
plotDecBoundaries(min_tr_feature, min_tr_label, min_tr_mean)
error_min_train = error_rate(min_tr_feature, min_tr_label, min_tr_mean, 3)
[min_test_feature, min_test_label, _] = find_feature_class_mean_3('wine_test.csv',[f1,f2])
error_min_test = error_rate(min_test_feature, min_test_label, min_tr_mean, 3)
# ******************************question(e)*************************************
# calculate the standard deviation
# deviation_tr and deviation_test are standard deviation of training and testing
# data respectly
total_err_tr = np.array(total_err_tr)
deviation_tr = np.std(total_err_tr)
total_err_test = []
for i in range(1,13):
    for j in range(i+1,14):
        [w_feature, wlabel, _] = find_feature_class_mean_3('wine_test.csv',[i,j])
        [_, _, w_mean] = find_feature_class_mean_3('wine_train.csv',[i,j])
        total_err_test.append(error_rate(w_feature, wlabel, w_mean, 3))
total_err_test = np.array(total_err_test)
deviation_test = np.std(total_err_test)
