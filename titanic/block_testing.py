# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

from common.testing_models import *
from features import *


def test_using_data(X_train, Y_train, X_test, Y_test):
    # Logistic Regression
    acc_log = logistic_regression(X_train, Y_train, X_test, Y_test)
    # Support Vector Machines
    acc_svc = support_vector_machines(X_train, Y_train, X_test, Y_test)

    acc_knn = KNeighbors(X_train, Y_train, X_test, Y_test)
    # Gaussian Naive Bayes
    acc_gaussian = gaussian_naive_bayes(X_train, Y_train, X_test, Y_test)
    # Perceptron
    acc_perceptron = perceptron(X_train, Y_train, X_test, Y_test)
    # Linear SVC
    acc_linear_svc = linear_svm(X_train, Y_train, X_test, Y_test)
    # Stochastic Gradient Descent
    acc_sgd = stochastic_gradient_descent(X_train, Y_train, X_test, Y_test)
    # Decision Tree
    acc_decision_tree = decision_tree(X_train, Y_train, X_test, Y_test)
    # Random Forest
    acc_random_forest = random_forest(X_train, Y_train, X_test, Y_test)

    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
    return models


def get_test_indexes(size, block, max_blocks):
    size_of_block = size / max_blocks
    start_index = block * size_of_block
    end_index = start_index + size_of_block
    return [{'start': start_index, 'end': end_index}]


def get_train_indexes(size, block, max_blocks):
    test_indexes = get_test_indexes(size, block, max_blocks)
    start = 0
    end = size
    train_indexes = []
    start_test = 0
    end_test = 0
    for test_block in test_indexes:
        start_test = test_block['start']
        end_test = test_block['end']
        if start < start_test:
            train_indexes.append({'start': start, 'end': start_test})
        start = end_test
    if end_test < end:
        train_indexes.append({'start': end_test, 'end': end})
    return train_indexes


def get_train_block(size, block, max_blocks, all_data):
    train_indexes = get_train_indexes(size, block, max_blocks)
    train_data = None
    for index in train_indexes:
        if train_data is None:
            train_data = all_data[index['start']:index['end']]
        else:
            train_data = train_data.append(all_data[index['start']:index['end']])
    return train_data


def get_test_block(size, block, max_blocks, all_data):
    test_indexes = get_test_indexes(size, block, max_blocks)
    test_data = None
    for index in test_indexes:
        if test_data is None:
            test_data = all_data[index['start']:index['end']]
        else:
            test_data = test_data.append(all_data[index['start']:index['end']])
    return test_data


def testing_on_train_set(all_data):
    MAX_BLOCK_NUMBER = 5
    size = len(all_data)
    all_results = []
    for i in range(MAX_BLOCK_NUMBER):
        all_test = get_test_block(size, i, MAX_BLOCK_NUMBER, all_data)
        all_train = get_train_block(size, i, MAX_BLOCK_NUMBER, all_data)

        X_train = all_train.drop("Survived", axis=1)
        Y_train = all_train["Survived"]
        X_test = all_test.drop("Survived", axis=1)
        Y_test = all_test["Survived"]

        results = test_using_data(X_train, Y_train, X_test, Y_test)
        print(results.sort_values(by='Score', ascending=False))
        all_results.append(results)
        print ('*'*20)

    all_results_combined = all_results[0].copy()
    all_results_combined['TotalScore'] = all_results_combined['Score']
    for i in range(1, MAX_BLOCK_NUMBER):
        all_results_combined['TotalScore'] += all_results[i]['Score']
    all_results_combined['TotalScore'] /= MAX_BLOCK_NUMBER
    all_results_combined = all_results_combined.drop('Score', axis=1)

    print(all_results_combined.sort_values(by='TotalScore', ascending=False))

