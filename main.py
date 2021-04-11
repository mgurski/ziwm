import numpy as numpy
from sklearn.feature_selection import SelectKBest, chi2, f_classif

import os

def load_data(data_path):
    dataset = []

    for file in os.listdir(data_path):
        data = numpy.loadtxt(data_path+"/"+file, dtype=int)
        data = numpy.swapaxes(data, 0, 1)
        dataset.append(data)

    X = dataset[0]
    dataset.pop(0)
    Y = X.shape[0]*[0]

    for i in range(len(dataset)):
        X = numpy.concatenate((X, dataset[i]))
        Y = Y + dataset[i].shape[0]*[i+1]

    Y = numpy.array(Y, dtype=int)

    return X, Y

def feature_ranking(X, Y):
    test = SelectKBest(chi2, k=59)
    best_features = test.fit(X, Y)

    selected = best_features.get_support(True)

    print(numpy.argsort(-1*best_features.scores_))

    sorted_best_features = numpy.sort(-1*best_features.scores_)*-1
    print(sorted_best_features)


X, Y = load_data('./dane')
feature_ranking(X, Y)