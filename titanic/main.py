# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

import common.models as model
from features import *
import block_testing

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]


def feature_extracting():
    global combine

    work_on_title(combine)

    sex_to_int(combine)
    fill_missed_age(combine)
    combine = age_to_categories(combine)
    add_family_size(combine)
    add_is_alone(combine)
    add_age_class(combine)
    fill_missed_port(combine)
    port_to_int(combine)
    fill_missed_fare(combine)
    fare_to_int(combine)
    combine = drop_unused_columns(combine)


def test():
    global combine
    train_df = combine[0]
    test_df = combine[1]
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()

    # Logistic Regression
    acc_log, Y_log = model.logistic_regression(X_train, Y_train, X_test)
    # Support Vector Machines
    acc_svc, Y_svc = model.support_vector_machines(X_train, Y_train, X_test)
    acc_knn, Y_knn = model.KNeighbors(X_train, Y_train, X_test)
    # Gaussian Naive Bayes
    acc_gaussian, Y_gaussian = model.gaussian_naive_bayes(X_train, Y_train, X_test)
    # Perceptron
    acc_perceptron, Y_perceptron = model.perceptron(X_train, Y_train, X_test)
    # Linear SVC
    acc_linear_svc, Y_linear_svc = model.linear_svm(X_train, Y_train, X_test)
    # Stochastic Gradient Descent
    acc_sgd, Y_sgd = model.stochastic_gradient_descent(X_train, Y_train, X_test)
    # Decision Tree
    acc_decision_tree, Y_decision_tree = model.decision_tree(X_train, Y_train, X_test)
    # Random Forest
    acc_random_forest, Y_random_forest = model.random_forest(X_train, Y_train, X_test)

    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
    print(models.sort_values(by='Score', ascending=False))


def make_submission():
    global combine
    train_df = combine[0]
    test_df = combine[1]
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    acc_svc, Y_svc = model.support_vector_machines(X_train, Y_train, X_test)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_svc
    })

    submission.to_csv('output/submission.csv', index=False)


feature_extracting()
test()
make_submission()
#block_testing.testing_on_train_set(combine[0])