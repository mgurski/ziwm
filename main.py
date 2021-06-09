import numpy as numpy
from numpy import savetxt
import os
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import csv
from scipy.stats import ttest_ind
from tabulate import tabulate


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

def initial_learning_rate_test():
    X, Y = load_data('./dane')
    X, Y = get_k_best_features(X, Y, 11)

    learning_rate_init_values = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    momentum_list = [0, 0.9]

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=3232)
    with open('wyniki/MPLClassificationInitialLearningRate.csv', mode='w') as csv_file:
        fieldnames = ['layer_size', 'momentum', 'learning_rate_init', 'number_of_features', 'mean_score', 'std_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for larning_rate_init_id, learning_rate_init in enumerate(learning_rate_init_values):
            for momentum_id, momentum in enumerate(momentum_list):

                clf = MLPClassifier(hidden_layer_sizes=300, solver='sgd', max_iter=600,
                                    momentum=momentum, learning_rate_init=learning_rate_init)

                scores = []

                for fold_id, (train_index, test_index) in enumerate(rskf.split(X, Y)):
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    clf.fit(X_train, Y_train)

                    predict = clf.predict(X_test)
                    scores.append(accuracy_score(Y_test, predict))

                mean_score = numpy.mean(scores)
                std_score = numpy.std(scores)

                writer.writerow({'momentum': str(momentum),
                                'learning_rate_init' : str(learning_rate_init),
                                'number_of_features': str(11), 'mean_score': str(mean_score),
                                'std_score': str(std_score)})
                print(str(momentum) + "      " +
                      str(learning_rate_init) + "     " + str(11) + "       " +
                      str(
                      mean_score) + " +- " + str(std_score))

def experimental_loop():
    X, Y = load_data('./dane')
    X, Y = get_k_best_features(X, Y, 59)

    layer_sizes = [50, 300, 600]
    momentum_list = [0, 0.9]

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=3232)
    with open('wyniki/MPLClassification.csv', mode='w') as csv_file:
        fieldnames = ['layer_size', 'momentum', 'number_of_features', 'mean_score', 'std_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        result_scores = numpy.zeros((6, 59, 10))

        for number_of_features in range(1, 60):
            for layer_size_id, layer_size in enumerate(layer_sizes):
                for momentum_id, momentum in enumerate(momentum_list):

                    clf = MLPClassifier(hidden_layer_sizes=layer_size, solver='sgd', max_iter=600,
                                        momentum=momentum, learning_rate_init=0.0001)

                    scores = []

                    for fold_id, (train_index, test_index) in enumerate(rskf.split(X[:, 0:number_of_features], Y)):
                        X_train, X_test = X[train_index], X[test_index]
                        Y_train, Y_test = Y[train_index], Y[test_index]

                        clf.fit(X_train, Y_train)

                        predict = clf.predict(X_test)
                        scores.append(accuracy_score(Y_test, predict))

                        result_scores[layer_size_id * 2 + momentum_id, number_of_features - 1, fold_id] = scores[
                            fold_id]

                    mean_score = numpy.mean(scores)
                    std_score = numpy.std(scores)

                    writer.writerow({'layer_size': str(layer_size), 'momentum': str(momentum),
                                     'number_of_features': str(number_of_features), 'mean_score': str(mean_score),
                                     'std_score': str(std_score)})
                    print(str(layer_size) + "      " + str(momentum) + "      " + str(
                        number_of_features) + "       " + str(
                        mean_score) + " +- " + str(std_score))
    numpy.save('wyniki', result_scores)


def ttest_results():
    scores = numpy.load('wyniki.npy')
    for index in range(6):
        savetxt('wyniki{}.csv'.format(index), scores[index], delimiter=',')

    best_classifier_results = numpy.zeros((6, 10))
    for index in range(6):
        mean_score = 0.
        for number_of_features in range(59):
            if mean_score < numpy.mean(scores[index, number_of_features]):
                mean_score = numpy.mean(scores[index, number_of_features])
                best_classifier_results[index] = scores[index, number_of_features]

    alfa = .05
    t_statistic = numpy.zeros((6, 6))
    p_value = numpy.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            t_statistic[i, j], p_value[i, j] = ttest_ind(best_classifier_results[i], best_classifier_results[j])

    headers = ["NoMom50", "Mom50", "NoMom300", "Mom300", "NoMom600", "Mom600"]
    names_column = numpy.array([["NoMom50"], ["Mom50"], ["NoMom300"], ["Mom300"], ["NoMom600"], ["Mom600"]])
    t_statistic_table = numpy.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = numpy.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = numpy.zeros((6, 6))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(numpy.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = numpy.zeros((6, 6))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(numpy.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(numpy.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)


def save_classifier_results(scores):
    for index in range(6):
        savetxt('wyniki{}.csv'.format(index), scores[index], delimiter=',')


experimental_loop()
#initial_learning_rate_test()
#ttest_results()
