from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np


def hyperparam_tunning(features, labels, X_train, y_train):
    """ This function performs the classification of the given features using several classifiers. From the obtained results
    the classifier which best fits the data and gives the best result is chosen and the respective confusion matrix is
    showed.
    Parameters
    ----------
    features: array-like
      set of features
    labels: array-like
      features class labelsX_train: array-like
      train set features
    y_train: array-like
      train set labels
    Returns
    -------
    best_clf: best classifier
    best_acc: best accuracy score
    """

    # Classifiers
    print("USING GRID SEARCH")
    names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "SVM", "AdaBoost", "Naive Bayes", "QDA"]
    classifiers = [
        DecisionTreeClassifier(max_depth=5, min_samples_split=len(features) // 10),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
        svm.SVC(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    best_acc = 0
    best_classifier = None
    best_clf = None
    for n, c in zip(names, classifiers):
        counter = 0
        n_iter_search = 20
        print(n)

        if n == "Random Forest":
            # specify parameters and distributions to sample from
            param_dist = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, None],
                          "max_features": sp_randint(1, 20),
                          "min_samples_split": sp_randint(2, 18),
                          "min_samples_leaf": sp_randint(1, 18),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"],
                          "n_estimators": sp_randint(5, 20)}
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=10, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            grid = grid.best_estimator_

        elif n == 'SVM':
            Cs = [0.001, 0.01, 0.1, 1, 10]
            gammas = [0.001, 0.01, 0.1, 1]
            C_range = 10. ** np.arange(-3, 8)
            gamma_range = 10. ** np.arange(-5, 4)
            param_dist = {'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr'], 'C': C_range,
                          'gamma': gamma_range}
            # run randomized search
            grid = GridSearchCV(c, param_dist, cv=10, scoring='accuracy')
            grid.fit(X_train, y_train)
            grid = grid.best_estimator_
        elif n == 'Decision Tree':
            param_dist = {"criterion": ["gini", "entropy"],
                          'splitter': ['best', 'random'],
                          "min_samples_split": sp_randint(2, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "max_depth": sp_randint(1, 11)
                          }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            grid = grid.best_estimator_
        else:
            # Train the classifier
            grid = c
            grid.fit(X_train, y_train)

        # print grid.get_params()
        scores = cross_val_score(grid, features, labels, cv=5)

        print("Accuracy: " + str(np.mean(scores)) + '%')
        print(np.std(scores))
        print('-----------------------------------------')
        if np.mean(scores) > best_acc:
            best_classifier = n
            best_acc = np.mean(scores)
            best_clf = grid

    print('******** Best Classifier: ' + str(best_classifier) + ' ********')

    print(best_clf)
    return best_clf, best_acc
