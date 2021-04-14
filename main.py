import numpy as numpy
import os
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(data_path):
    dataset = []

    for file in os.listdir(data_path):
        data = numpy.loadtxt(data_path + "/" + file, dtype=int)
        data = numpy.swapaxes(data, 0, 1)
        dataset.append(data)
        print(data.shape)

    X = dataset[0]
    dataset.pop(0)
    Y = X.shape[0] * [0]

    for i in range(len(dataset)):
        X = numpy.concatenate((X, dataset[i]))
        Y = Y + dataset[i].shape[0] * [i + 1]

    Y = numpy.array(Y, dtype=int)

    return X, Y


def get_k_best_features(X, Y, k):
    test = SelectKBest(chi2, k=k)
    best_features = test.fit(X, Y)

    indexes = numpy.argsort(-1 * best_features.scores_)
    sorted_best_features = numpy.sort(-1 * best_features.scores_) * -1

    print(indexes[:k])
    print(sorted_best_features[:k])

    return X[:, indexes[:k]], Y


def experimental_loop():
    X, Y = load_data('./dane')
    X, Y = get_k_best_features(X, Y, 59)

    layer_sizes = [50, 200, 400]
    momentum_list = [0, 0.9]

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=3232)

    for number_of_features in range(1, 59):
        for layer_size in layer_sizes:
            for momentum in momentum_list:

                clf = MLPClassifier(hidden_layer_sizes=layer_size, solver='sgd', max_iter=800, batch_size=100,
                                    momentum=momentum, tol=1e-6, n_iter_no_change=20)

                scores = []

                for train_index, test_index in rskf.split(X[:, 1:number_of_features], Y):
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    clf.fit(X_train, Y_train)

                    predict = clf.predict(X_test)
                    scores.append(accuracy_score(Y_test, predict))

                mean_score = numpy.mean(scores)
                std_score = numpy.std(scores)

                print(str(layer_size) + "      " + str(momentum) + "      " + str(number_of_features) + "       " + str(mean_score) + " +- " + str(std_score))


experimental_loop()
