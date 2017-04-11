from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


class TestingModels:
    def __init__(self):
        pass

    def run(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        acc_log = self.logistic_regression_train()
        acc_svc = self.support_vector_machines_train()
        acc_knn = self.KNeighbors_train()
        acc_gaussian = self.gaussian_naive_bayes_train()
        acc_perceptron = self.perceptron_train()
        acc_linear_svc = self.linear_svm()
        acc_sgd = self.stochastic_gradient_descent_train()
        acc_decision_tree = self.decision_tree_train()
        acc_random_forest = self.random_forest_train()
        acc_second = self.second_level()

        results = [acc_svc, acc_knn, acc_log,
                   acc_random_forest, acc_gaussian, acc_perceptron,
                   acc_sgd, acc_linear_svc, acc_decision_tree, acc_second]

        modelNames = ['Support Vector Machines', 'KNN', 'Logistic Regression',
                      'Random Forest', 'Naive Bayes', 'Perceptron',
                      'Stochastic Gradient Decent', 'Linear SVC',
                      'Decision Tree', '2 Level']
        return {
            'score': results,
            'models': modelNames,
        }

    def logistic_regression_train(self):
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.Y_train)
        acc_log = round(self.logreg.score(self.X_test, self.Y_test) * 100, 2)

        # coeff_df = pd.DataFrame(self.X_train.columns.delete(0))
        # coeff_df.columns = ['Feature']
        # coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
        #
        # print coeff_df.sort_values(by='Correlation', ascending=False)
        # print '*' * 20

        return acc_log

    def support_vector_machines_train(self):
        self.svc = SVC()
        self.svc.fit(self.X_train, self.Y_train)
        acc_svc = round(self.svc.score(self.X_test, self.Y_test) * 100, 2)
        return acc_svc

    def KNeighbors_train(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.X_train, self.Y_train)
        acc_knn = round(self.knn.score(self.X_test, self.Y_test) * 100, 2)
        return acc_knn

    def gaussian_naive_bayes_train(self):
        self.gaussian = GaussianNB()
        self.gaussian.fit(self.X_train, self.Y_train)
        acc_gaussian = round(self.gaussian.score(self.X_test, self.Y_test) * 100, 2)
        return acc_gaussian

    def perceptron_train(self):
        self.perceptron = Perceptron()
        self.perceptron.fit(self.X_train, self.Y_train)
        acc_perceptron = round(self.perceptron.score(self.X_test, self.Y_test) * 100, 2)
        return acc_perceptron

    def linear_svm(self):
        self.linear_svc = LinearSVC()
        self.linear_svc.fit(self.X_train, self.Y_train)
        acc_linear_svc = round(self.linear_svc.score(self.X_test, self.Y_test) * 100, 2)
        return acc_linear_svc

    def stochastic_gradient_descent_train(self):
        self.sgd = SGDClassifier()
        self.sgd.fit(self.X_train, self.Y_train)
        acc_sgd = round(self.sgd.score(self.X_test, self.Y_test) * 100, 2)
        return acc_sgd

    def decision_tree_train(self):
        self.decision_tree = DecisionTreeClassifier()
        self.decision_tree.fit(self.X_train, self.Y_train)
        acc_decision_tree = round(self.decision_tree.score(self.X_test, self.Y_test) * 100, 2)
        return acc_decision_tree

    def random_forest_train(self):
        self.random_forest = RandomForestClassifier(n_estimators=100)
        self.random_forest.fit(self.X_train, self.Y_train)
        acc_random_forest = round(self.random_forest.score(self.X_test, self.Y_test) * 100, 2)
        return acc_random_forest

    def second_level_predict(self, X_test):
        first_level_test_predictions = pd.DataFrame({
            "svm": self.svc.predict(X_test),
            #"logreg": self.logreg.predict(X_test),
            "random_forest": self.random_forest.predict(X_test),
            #"KNeighbors": self.knn.predict(X_test),
            #"gaussian": self.gaussian.predict(X_test),
            #"perceptron": self.perceptron.predict(X_test),
            #"linear_svc": self.linear_svc.predict(X_test),
            #"decision_tree": self.decision_tree.predict(X_test),
        })

        return self.second_level.predict(first_level_test_predictions)

    def second_level(self):
        #self.second_level = SVC()
        self.second_level = RandomForestClassifier(n_estimators=100)

        first_level_predictions = pd.DataFrame({
            "svm": self.svc.predict(self.X_train),
            #"logreg": self.logreg.predict(self.X_train),
            "random_forest": self.random_forest.predict(self.X_train),
            #"KNeighbors": self.knn.predict(self.X_train),
            #"gaussian": self.gaussian.predict(self.X_train),
            #"perceptron": self.perceptron.predict(self.X_train),
            #"linear_svc": self.linear_svc.predict(self.X_train),
            #"decision_tree": self.decision_tree.predict(self.X_train),
        })

        self.second_level.fit(first_level_predictions, self.Y_train)

        first_level_test_predictions = pd.DataFrame({
            "svm": self.svc.predict(self.X_test),
            #"logreg": self.logreg.predict(self.X_test),
            "random_forest": self.random_forest.predict(self.X_test),
            #"KNeighbors": self.knn.predict(self.X_test),
            #"gaussian": self.gaussian.predict(self.X_test),
            #"perceptron": self.perceptron.predict(self.X_test),
            #"linear_svc": self.linear_svc.predict(self.X_test),
            #"decision_tree": self.decision_tree.predict(self.X_test),
        })

        acc = round(self.second_level.score(first_level_test_predictions, self.Y_test) * 100, 2)
        return acc
