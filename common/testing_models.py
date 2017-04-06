from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def logistic_regression(X_train, Y_train, X_test, Y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
    return acc_log


def support_vector_machines(X_train, Y_train, X_test, Y_test):
    svc = SVC()
    svc.fit(X_train, Y_train)
    acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
    return acc_svc


def KNeighbors(X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
    return acc_knn


def gaussian_naive_bayes(X_train, Y_train, X_test, Y_test):
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)
    return acc_gaussian


def perceptron(X_train, Y_train, X_test, Y_test):
    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)
    return acc_perceptron

def linear_svm(X_train, Y_train, X_test, Y_test):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)
    return acc_linear_svc


def stochastic_gradient_descent(X_train, Y_train, X_test, Y_test):
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)
    return acc_sgd


def decision_tree(X_train, Y_train, X_test, Y_test):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
    return acc_decision_tree


def random_forest(X_train, Y_train, X_test, Y_test):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
    return acc_random_forest
