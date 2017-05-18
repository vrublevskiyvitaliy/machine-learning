# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import time

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

import common.models as model
from features import *
from common.testing_models import *
import block_testing

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]


def feature_extracting():
    global combine

    work_on_title(combine)
    work_on_cabine(combine)
    add_name_size(combine)
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
    acc_svc, Y_svc = model.random_forest(X_train, Y_train, X_test)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_svc
    })
    time_suffix = time.strftime("%H_%M_%S%d_%m_%Y")
    submission.to_csv('output/submission_' + time_suffix + '.csv', index=False)


def make_submission_using_test_module():
    global combine
    train_df = combine[0]
    test_df = combine[1]
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()

    testModel = TestingModels()
    testModel.run(X_train, Y_train, X_train, Y_train)

    Y_svc = testModel.second_level_predict(X_test)


    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_svc
    })
    time_suffix = time.strftime("%H_%M_%S%d_%m_%Y")
    submission.to_csv('output/submission_' + time_suffix + '.csv', index=False)

feature_extracting()


#print(train_df.head(10))
#print train_df.describe(include=['O'])
#print train_df[['CabineChar', 'Survived']].groupby(['CabineChar'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print train_df[['CabineD', 'Survived']].groupby(['CabineD'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print train_df[['CabineE', 'Survived']].groupby(['CabineE'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print train_df[['CabineB', 'Survived']].groupby(['CabineB'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print train_df[['CabineT', 'Survived']].groupby(['CabineT'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#test()
#make_submission()
#make_submission_using_test_module()
block_testing.testing_on_train_set(combine[0])